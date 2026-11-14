# SPDX-FileCopyrightText: 2026 Marc Warne <marc@kernel-panic.im>
#
# SPDX-License-Identifier: MIT

"""
The last complete bootstrap, recent incoming texts and node state, kept so proxy clients can connect without the radio.

Everything is kept as the serialised FromRadio as received, never rebuilt from the node DB dictionaries,
so UI config, the file manifest and fields this code does not know about reach clients unchanged.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from google.protobuf.message import DecodeError

from .admin import changes_gateway, gateway_admin
from .protobuf import admin_pb2, mesh_pb2, portnums_pb2, telemetry_pb2, xmodem_pb2

LOGGER = logging.getLogger(__package__)

# firmware's special want_config_id values: config without other nodes, and nodes only
NONCE_ONLY_CONFIG = 69420
NONCE_ONLY_NODES = 69421

# bounded by count only, as the radio bounds its own store by size: a clock would empty it on a quiet mesh
HISTORY_SIZE = 256
# latest position, telemetry and status per node, which firmware 2.8 sends after the completion, not in it
NODE_STATE_SIZE = 2048
_NODE_STATE_PORTS = {portnums_pb2.PortNum.POSITION_APP, portnums_pb2.PortNum.NODE_STATUS_APP}
_NODE_STATE_TELEMETRY = {"device_metrics", "environment_metrics"}

# what the firmware may interleave with a bootstrap without it being part of the node's state
_NOT_STATE = frozenset({
    "packet", "queueStatus", "log_record", "clientNotification", "lockdown_status", "display_frame",
    "display_palette", "mqttClientProxyMessage", "xmodemPacket", "rebooted",
})  # fmt: skip
# firmware bootstrap order (PhoneAPI::getFromRadio); records of other kinds are kept but not ordered
_PHASE = {
    "my_info": 0,
    "deviceuiConfig": 1,
    "own_node_info": 2,
    "metadata": 3,
    "channel": 4,
    "config": 5,
    "moduleConfig": 6,
    "node_info": 7,
    "fileInfo": 8,
}
# never live: bootstrap records would mix generations, and a reboot closes every client instead; anything else,
# file transfers and MQTT proxy messages included, and kinds this code does not know, is passed on
_NOT_LIVE = frozenset({*_PHASE, "region_presets", "config_complete_id", "rebooted"})
# not deviceuiConfig: firmware before 2.6 does not send it
_REQUIRED = {"my_info", "own_node_info", "metadata", "channel", "config", "moduleConfig"}
_CHANNELS = 8  # MAX_NUM_CHANNELS
# sent unconditionally by every 2.5 to 2.8 firmware; later ones are optional so newer firmware is not refused
_CONFIGS = {"device", "position", "power", "network", "display", "lora", "bluetooth", "security"}
_MODULE_CONFIGS = {
    "mqtt", "serial", "external_notification", "store_forward", "range_test", "telemetry", "canned_message",
    "audio", "remote_hardware", "neighbor_info", "ambient_lighting", "detection_sensor", "paxcounter",
}  # fmt: skip

# admin that takes nodes out of the gateway's node DB, so their replayed state would bring them back in the app
_FORGETS_NODES = frozenset({"nodedb_reset", "factory_reset_config", "factory_reset_device"})
# a bootstrap whose my_info came this soon after a write may have been read before the radio applied it
_WRITE_SETTLE_S = 2.0

# called with each live record and its history sequence number (0 if it is not a text), or None when invalidated
Subscriber = Callable[[bytes | None, int], None]


@dataclass(frozen=True)
class Bound:
    snapshot: "Snapshot"
    # latest position, telemetry and status per node, then texts, in arrival order
    node_state: list[bytes]
    history: list[tuple[int, bytes]]


@dataclass(frozen=True)
class Snapshot:
    generation: int
    my_node_num: int
    # (raw FromRadio, is a node_info of another node), in the order the firmware sent them
    records: tuple[tuple[bytes, bool | None], ...]

    def records_for(self, nonce: int) -> Iterator[bytes]:
        """Yield what the firmware sends for this want_config_id, up to but not including its completion."""
        for raw, other_node in self.records:
            if nonce == NONCE_ONLY_NODES:
                if other_node is not None:
                    yield raw
            elif nonce != NONCE_ONLY_CONFIG or not other_node:
                yield raw


class ReplayCache:
    """
    Fed every FromRadio of the gateway's own session, in order, before anything else processes it.

    Subscribers are called synchronously from ingest() with each live record, or with None once the radio session
    they were bound to has ended. A write to the gateway only makes the snapshot stale: clients already bound keep
    their session, as on the radio itself, and new ones wait for the next download.
    """

    def __init__(self) -> None:
        self._logger = LOGGER.getChild(self.__class__.__name__)
        self.snapshot: Snapshot | None = None
        self.ready = asyncio.Event()
        self._generation = 0
        self._pending: list[tuple[str, bytes]] | None = None
        self._identity: tuple | None = None
        self.my_node_num: int | None = None
        # (sequence number, raw FromRadio)
        self._history: deque[tuple[int, bytes]] = deque(maxlen=HISTORY_SIZE)
        self._history_seq = 0
        # (sender, port, telemetry variant) -> raw FromRadio, oldest update first
        self._node_state: dict[tuple, bytes] = {}
        self._subscribers: set[Subscriber] = set()
        # why the last bootstrap was not kept, and who to tell
        self.failure: str | None = None
        self.on_failure: Callable[[], None] | None = None
        self._last_write = float("-inf")
        self._bootstrap_started = 0.0

    def ingest(self, from_radio: mesh_pb2.FromRadio, requested_config_id: int | None) -> None:
        kind = from_radio.WhichOneof("payload_variant")
        if kind == "rebooted":
            self.invalidate("upstream rebooted")
        elif kind == "my_info":
            # every full bootstrap starts with my_info; a new session invalidates before it, so this one may be a
            # refresh on the same session, which leaves bound clients alone
            self.stale("upstream started a bootstrap")
            self._pending = []
            self._bootstrap_started = time.monotonic()
            self.my_node_num = from_radio.my_info.my_node_num

        if self._pending is not None and kind not in _NOT_STATE:
            if kind == "config_complete_id":
                self._complete(from_radio.config_complete_id, requested_config_id)
            else:
                self._pending.append((kind or "", from_radio.SerializeToString()))
            return

        if kind in _NOT_LIVE:
            return
        raw = from_radio.SerializeToString()
        seq = 0
        if kind == "packet":
            if self._is_incoming_text(from_radio.packet):
                self._history_seq = seq = self._history_seq + 1
                self._history.append((seq, raw))
            elif key := self._node_state_key(from_radio.packet):
                self._node_state.pop(key, None)
                self._node_state[key] = raw
                if len(self._node_state) > NODE_STATE_SIZE:
                    del self._node_state[next(iter(self._node_state))]
        for subscriber in list(self._subscribers):
            subscriber(raw, seq)

    @property
    def bootstrapping(self) -> bool:
        return self._pending is not None

    @property
    def bootstrap_age(self) -> float:
        """Seconds since the bootstrap in progress began."""
        return time.monotonic() - self._bootstrap_started

    def frame_lost(self) -> None:
        """Note a frame from the gateway that arrived but could not be read."""
        self.discard("a frame from the gateway could not be read")

    def discard(self, reason: str) -> None:
        """Give up on a bootstrap in progress, if any."""
        if self._pending is not None:
            self._pending = None
            self._failed(reason)

    def note_write(self, to_radio: mesh_pb2.ToRadio) -> bool:
        """Mark the snapshot stale for a write that changes what a bootstrap carries; True if it is one."""
        if to_radio.WhichOneof("payload_variant") == "xmodemPacket":
            if to_radio.xmodemPacket.control != xmodem_pb2.XModem.Control.EOT:
                return False
            reason = "a file upload finished"  # the file list in the snapshot changed
        elif (admin := gateway_admin(to_radio, self.my_node_num)) is None or not changes_gateway(admin):
            return False
        else:
            reason = f"gateway admin {admin.WhichOneof('payload_variant') or 'not readable'}"
            self._forget(admin)
        self._last_write = time.monotonic()
        self.stale(reason)
        return True

    def _forget(self, admin: admin_pb2.AdminMessage) -> None:
        """Drop what would bring back a node the gateway was told to forget; called again after the send."""
        variant = admin.WhichOneof("payload_variant")
        if variant in _FORGETS_NODES:
            self._node_state.clear()
        elif variant in {"remove_by_nodenum", "set_ignored_node"}:
            num = getattr(admin, variant)
            self._node_state = {key: value for key, value in self._node_state.items() if key[0] != num}
            if variant == "set_ignored_node":  # firmware also drops an ignored node's stored messages
                kept = [entry for entry in self._history if _sender(entry[1]) != num]
                self._history = deque(kept, maxlen=HISTORY_SIZE)

    def stale(self, reason: str) -> None:
        """Forget the snapshot and any bootstrap in progress; bound clients stay on the session they joined."""
        self._pending = None
        if self.snapshot is not None:
            self._logger.debug("Bootstrap snapshot %d no longer current: %s", self.snapshot.generation, reason)
        self.snapshot = None
        self.ready.clear()

    def invalidate(self, reason: str) -> None:
        """Forget the snapshot as the radio session ends, and tell every bound client."""
        self.stale(reason)
        subscribers, self._subscribers = self._subscribers, set()
        for subscriber in subscribers:
            subscriber(None, 0)

    def bind(self, subscriber: Subscriber, after_seq: int = 0) -> Bound | None:
        """
        Bind a client to the current snapshot: what to replay, and every later live record to subscriber.

        Synchronous on purpose: no record can arrive between taking the history and subscribing, so each one
        is either replayed or delivered live, exactly once. Texts up to after_seq are left out.
        """
        if self.snapshot is None:
            return None
        self._subscribers.add(subscriber)
        history = [(seq, raw) for seq, raw in self._history if seq > after_seq]
        return Bound(self.snapshot, list(self._node_state.values()), history)

    def subscribe(self, subscriber: Subscriber) -> None:
        """Pass live records to a client that has not asked for config, until the radio session ends."""
        self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.discard(subscriber)

    def _is_incoming_text(self, packet: mesh_pb2.MeshPacket) -> bool:
        return (
            self.my_node_num is not None
            and packet.HasField("decoded")
            and packet.decoded.portnum == portnums_pb2.PortNum.TEXT_MESSAGE_APP
            and getattr(packet, "from") != self.my_node_num
        )

    def _node_state_key(self, packet: mesh_pb2.MeshPacket) -> tuple | None:
        sender, port = getattr(packet, "from"), packet.decoded.portnum
        if not packet.HasField("decoded") or sender in {0, self.my_node_num}:
            return None
        if port in _NODE_STATE_PORTS:
            return sender, port, None
        if port != portnums_pb2.PortNum.TELEMETRY_APP:
            return None
        try:
            variant = telemetry_pb2.Telemetry.FromString(packet.decoded.payload).WhichOneof("variant")
        except DecodeError:
            return None
        return (sender, port, variant) if variant in _NODE_STATE_TELEMETRY else None

    def _failed(self, reason: str) -> None:
        self._logger.warning("Discarding the gateway's configuration download, proxy clients must wait: %s", reason)
        self.failure = reason
        if self.on_failure is not None:
            self.on_failure()

    def _complete(self, config_complete_id: int, requested_config_id: int | None) -> None:
        pending, self._pending = self._pending, None
        if config_complete_id != requested_config_id:
            self._logger.debug(
                "Discarding bootstrap completed for %d, not our %s", config_complete_id, requested_config_id
            )
            return
        if requested_config_id in {NONCE_ONLY_CONFIG, NONCE_ONLY_NODES}:
            self._logger.debug("Not keeping partial bootstrap %d as a snapshot", requested_config_id)
            return
        if pending is None or self.my_node_num is None:
            return  # unreachable: a completion is only taken while a bootstrap begun by my_info is pending
        if self._bootstrap_started - self._last_write < _WRITE_SETTLE_S:
            # the radio answers a config request in the pass that reads it, and applies admin later
            self._failed("the gateway was changed as the download began")
            return

        result = _read_bootstrap(pending)
        if isinstance(result, str):
            self._failed(result)
            return
        records, identity = result

        if self._identity is not None and identity != self._identity:
            # what was received under another node identity, channel or key must not replay under the new one
            self._logger.info("Gateway identity, channels or keys changed, clearing %d texts", len(self._history))
            self._history.clear()
            self._node_state.clear()
        self._identity, self.failure = identity, None
        self._generation += 1
        self.snapshot = Snapshot(self._generation, self.my_node_num, tuple(records))
        self.ready.set()
        self._logger.debug("Bootstrap snapshot %d: %d records", self._generation, len(records))


def _sender(raw: bytes) -> int:
    return getattr(mesh_pb2.FromRadio.FromString(raw).packet, "from")


def _read_bootstrap(pending: list[tuple[str, bytes]]) -> tuple[list[tuple[bytes, bool | None]], tuple] | str:
    """Check a received bootstrap against firmware's order and contents; the records and identity, or why not."""
    parsed = [(kind, raw, mesh_pb2.FromRadio.FromString(raw)) for kind, raw in pending]
    my_info = parsed[0][2].my_info
    records: list[tuple[bytes, bool | None]] = []
    phases: list[str] = []
    identity: list[object] = [my_info.my_node_num]
    channels: list[int] = []
    variants: set[str | None] = set()
    last_phase = -1
    for kind, raw, from_radio in parsed:
        other_node = None
        phase_kind = kind
        if kind == "node_info":
            other_node = from_radio.node_info.num != my_info.my_node_num
            phase_kind = "node_info" if other_node else "own_node_info"
        elif kind == "channel":
            channel = from_radio.channel
            channels.append(channel.index)
            # what decides which channel a stored message belongs to; other channel settings do not
            identity.append((channel.index, channel.role, channel.settings.name, channel.settings.psk))
        elif kind in {"config", "moduleConfig"}:
            variants.add(getattr(from_radio, kind).WhichOneof("payload_variant"))
            if from_radio.config.HasField("security"):
                identity.append((from_radio.config.security.public_key, from_radio.config.security.private_key))
        phase = _PHASE.get(phase_kind)
        if phase is not None:
            if phase < last_phase or (phase_kind == "own_node_info" and phase_kind in phases):
                return f"{kind} out of firmware order"
            last_phase = phase
            phases.append(phase_kind)
        records.append((raw, other_node))
    if missing := _REQUIRED.difference(phases):
        return f"no {', '.join(sorted(missing))}"
    if channels != list(range(_CHANNELS)):
        # firmware sends every channel slot in index order; fewer, or zeroed ones, mean a broken bootstrap
        return f"channels {channels}"
    if missing := (_CONFIGS | _MODULE_CONFIGS).difference(variants):
        return f"no {', '.join(sorted(missing))} config"
    if my_info.nodedb_count > 1 and "node_info" not in phases:
        # the node count can change during a download, so only a wholly missing node list is detectable
        return "no other nodes"
    return records, tuple(identity)
