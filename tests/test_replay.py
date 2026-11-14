# SPDX-FileCopyrightText: 2026 Marc Warne <marc@kernel-panic.im>
#
# SPDX-License-Identifier: MIT

"""The replay cache on its own: which bootstraps become snapshots, and what the text history keeps."""

import asyncio

import pytest

from custom_components.meshtastic.aiomeshtastic import replay
from custom_components.meshtastic.aiomeshtastic.connection.streaming import StreamingClientTransport
from custom_components.meshtastic.aiomeshtastic.connection.tcp import TcpConnection
from custom_components.meshtastic.aiomeshtastic.protobuf import mesh_pb2, portnums_pb2, telemetry_pb2, xmodem_pb2
from custom_components.meshtastic.aiomeshtastic.protobuf.admin_pb2 import AdminMessage
from custom_components.meshtastic.aiomeshtastic.replay import ReplayCache

from .mesh_fixtures import OWN, FromRadio, bootstrap, channel, expected_for, not_history, packet

NONCE = 0x1234


@pytest.fixture(autouse=True)
def no_settle_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # these tests write and bootstrap within microseconds; the commit guard has its own test
    monkeypatch.setattr(replay, "_WRITE_SETTLE_S", 0)


def committed(nonce: int = NONCE, **kwargs: object) -> ReplayCache:
    cache = ReplayCache()
    for record in bootstrap(nonce, **kwargs):
        cache.ingest(record, nonce)
    return cache


def texts(cache: ReplayCache) -> list[bytes]:
    return [raw for _, raw in cache.bind(lambda *_: None).history]


@pytest.mark.parametrize("nonce", [NONCE, 0, 69420, 69421])
def test_snapshot_records_for_each_nonce(nonce: int) -> None:
    snapshot = committed().snapshot
    assert [*snapshot.records_for(nonce), expected_for(nonce)[-1]] == expected_for(nonce)


def without(records: list, kind: str) -> list:
    return [r for r in records if r.WhichOneof("payload_variant") != kind]


@pytest.mark.parametrize(
    ("records", "requested"),
    [
        pytest.param(bootstrap(NONCE), NONCE + 1, id="someone else's completion"),
        pytest.param(bootstrap(NONCE), None, id="no request of ours"),
        pytest.param(bootstrap(69420), 69420, id="our own config-only request"),
        pytest.param(without(bootstrap(NONCE), "channel"), NONCE, id="no channels"),
        pytest.param(without(bootstrap(NONCE), "moduleConfig"), NONCE, id="no module config"),
        pytest.param(bootstrap(NONCE)[1:], NONCE, id="no my_info"),
        pytest.param(bootstrap(NONCE)[:8] + bootstrap(NONCE)[:-1][-200:] + [bootstrap(NONCE)[-1]], NONCE, id="gap"),
        pytest.param([*bootstrap(NONCE)[:3], channel(0, "x"), *bootstrap(NONCE)[3:]], NONCE, id="out of order"),
        pytest.param([r for r in bootstrap(NONCE) if r.channel.index != 7], NONCE, id="a channel missing"),
        pytest.param([r for r in bootstrap(NONCE) if not r.config.HasField("lora")], NONCE, id="a config missing"),
        pytest.param(
            [r for r in bootstrap(NONCE) if not r.moduleConfig.HasField("audio")], NONCE, id="a module config missing"
        ),
        pytest.param(
            [r for r in bootstrap(NONCE) if not r.HasField("node_info") or r.node_info.num == OWN],
            NONCE,
            id="other nodes missing",
        ),
        pytest.param(
            [FromRadio(channel={}) if r.HasField("channel") else r for r in bootstrap(NONCE)],
            NONCE,
            id="zeroed channels",
        ),
    ],
)
def test_incomplete_or_foreign_bootstrap_is_never_a_snapshot(records: list, requested: int | None) -> None:
    cache = ReplayCache()
    for record in records:
        cache.ingest(record, requested)
    assert cache.snapshot is None
    assert not cache.ready.is_set()
    assert cache.bind(lambda *_: None) is None


def test_bootstrap_cut_short_by_a_new_one_keeps_only_the_new_one() -> None:
    cache = ReplayCache()
    for record in bootstrap(1)[:100]:
        cache.ingest(record, 1)
    for record in bootstrap(2):
        cache.ingest(record, 2)
    assert [*cache.snapshot.records_for(2)] == expected_for(2)[:-1]


def test_history_keeps_incoming_texts_as_received() -> None:
    cache = committed()
    assert texts(cache) == []  # nothing from before collection started

    received = [packet(5, 7), packet(6, 7), packet(9, 0)]  # two senders share an id, and an id of zero
    for record in [*not_history(), *received, *not_history()]:
        cache.ingest(record, NONCE)
    assert texts(cache) == [r.SerializeToString() for r in received]


def test_history_survives_invalidation_and_is_live_to_subscribers_once() -> None:
    cache = committed()
    seen = []
    cache.bind(lambda raw, seq: seen.append((raw, seq)))
    cache.ingest(packet(5, 1), NONCE)
    cache.invalidate("test")
    cache.ingest(packet(5, 2), NONCE)  # after invalidation nobody is subscribed
    assert seen == [(packet(5, 1).SerializeToString(), 1), (None, 0)]
    assert cache.bind(lambda *_: None) is None
    assert cache._history[-1] == (2, packet(5, 2).SerializeToString())


def test_a_write_or_a_refresh_leaves_bound_clients_on_their_session() -> None:
    cache = committed()
    seen = []
    cache.bind(lambda raw, _: seen.append(raw))
    cache.note_write(admin_to(OWN, AdminMessage(set_owner={}).SerializeToString()))
    assert cache.snapshot is None
    assert cache.bind(lambda *_: None) is None  # new clients wait for the next download
    for record in bootstrap(2):  # that download, on the same session
        cache.ingest(record, 2)
    cache.ingest(packet(5, 1), 2)
    assert seen == [packet(5, 1).SerializeToString()]  # live traffic, and none of the download's records
    assert cache.snapshot.generation == 2


def test_history_is_bounded_by_count() -> None:
    cache = committed()
    for packet_id in range(300):
        cache.ingest(packet(5, packet_id), NONCE)
    assert texts(cache) == [packet(5, i).SerializeToString() for i in range(44, 300)]


def test_history_and_node_state_older_than_a_day_are_still_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(replay.time, "monotonic", lambda: now[0])
    cache = committed()
    position = packet(6, 1, portnums_pb2.PortNum.POSITION_APP, b"")
    for record in (packet(5, 1), position):
        cache.ingest(record, NONCE)
    now[0] += 3 * 24 * 3600  # a quiet mesh: three days with nothing new
    bound = cache.bind(lambda *_: None)
    assert bound.history == [(1, packet(5, 1).SerializeToString())]
    assert bound.node_state == [position.SerializeToString()]


def test_history_is_cleared_when_gateway_identity_changes() -> None:
    cache = committed()
    cache.ingest(packet(5, 1), NONCE)
    for record in bootstrap(2):  # a normal reconnect, same settings
        cache.ingest(record, 2)
    assert len(texts(cache)) == 1

    uplink = [FromRadio.FromString(r.SerializeToString()) for r in bootstrap(3)]
    for record in uplink:
        if record.HasField("channel"):
            record.channel.settings.uplink_enabled = True
        cache.ingest(record, 3)
    assert len(texts(cache)) == 1  # a channel setting that does not decide which channel a text is on

    cache.ingest(packet(5, 3, portnums_pb2.PortNum.POSITION_APP, b""), NONCE)
    for record in bootstrap(4, channel_name="Private"):
        cache.ingest(record, 4)
    assert texts(cache) == []
    assert cache.bind(lambda *_: None).node_state == []
    assert cache.snapshot.generation == 4


def telemetry(sender: int, packet_id: int, **variant: object) -> FromRadio:
    payload = telemetry_pb2.Telemetry(time=packet_id, **variant).SerializeToString()
    return packet(sender, packet_id, portnums_pb2.PortNum.TELEMETRY_APP, payload)


def test_node_state_keeps_the_latest_position_telemetry_and_status_per_node() -> None:
    port = portnums_pb2.PortNum
    cache = committed()
    kept = [
        packet(6, 1, port.POSITION_APP, b""),  # replaced by 5 below
        telemetry(5, 2, device_metrics={"battery_level": 50}),
        telemetry(5, 3, environment_metrics={"temperature": 20}),
        packet(5, 4, port.NODE_STATUS_APP, b"ok"),
        packet(6, 5, port.POSITION_APP, b"newer"),
    ]
    ignored = [
        telemetry(5, 6, local_stats={}),
        packet(OWN, 7, port.POSITION_APP, b""),  # the gateway's own is in its node info
        packet(5, 8),  # a text, which is history
        FromRadio(packet=mesh_pb2.MeshPacket(id=9, encrypted=b"\x01")),
        packet(5, 10, port.TELEMETRY_APP, b"\xff"),
    ]
    for record in kept + ignored:
        cache.ingest(record, NONCE)
    assert cache.bind(lambda *_: None).node_state == [r.SerializeToString() for r in kept[1:]]


def test_node_state_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, "NODE_STATE_SIZE", 3)
    cache = committed()
    for sender in range(10, 15):
        cache.ingest(packet(sender, 1, portnums_pb2.PortNum.POSITION_APP, b""), NONCE)
    assert cache.bind(lambda *_: None).node_state == [
        packet(s, 1, portnums_pb2.PortNum.POSITION_APP, b"").SerializeToString() for s in (12, 13, 14)
    ]


def test_live_records_exclude_only_bootstrap_records() -> None:
    cache = committed()
    seen = []
    cache.bind(lambda raw, _: seen.append(FromRadio.FromString(raw).WhichOneof("payload_variant")))
    for record in [
        FromRadio(lockdown_status={}),
        FromRadio(display_frame={}),
        FromRadio(queueStatus={}),
        FromRadio(mqttClientProxyMessage={}),
        FromRadio(xmodemPacket={}),
        FromRadio(config={}),
        FromRadio(node_info={}),
        FromRadio(config_complete_id=1),
        FromRadio(region_presets={}),
    ]:
        cache.ingest(record, NONCE)
    unknown = FromRadio.FromString(bytes([0xBA, 0x3E, 0x00]))  # field 999, a kind this code does not know
    cache.ingest(unknown, NONCE)
    assert seen == ["lockdown_status", "display_frame", "queueStatus", "mqttClientProxyMessage", "xmodemPacket", None]


def test_a_discarded_bootstrap_is_reported() -> None:
    cache = committed()
    failures = []
    cache.on_failure = lambda: failures.append(cache.failure)
    for record in [r for r in bootstrap(2) if r.channel.index != 7]:
        cache.ingest(record, 2)
    records = bootstrap(3)
    for record in records[:50]:
        cache.ingest(record, 3)
    cache.frame_lost()
    assert failures == ["channels [0, 1, 2, 3, 4, 5, 6]", "a frame from the gateway could not be read"]
    for record in bootstrap(4):
        cache.ingest(record, 4)
    assert cache.failure is None


def test_reboot_invalidates_snapshot() -> None:
    cache = committed()
    seen = []
    cache.bind(lambda raw, _: seen.append(raw))
    cache.ingest(replay.mesh_pb2.FromRadio(rebooted=True), NONCE)
    assert cache.snapshot is None
    assert seen == [None]


def test_unreadable_frame_discards_only_a_bootstrap_in_progress() -> None:
    cache = committed()
    cache.frame_lost()
    assert cache.snapshot is not None

    records = bootstrap(2)
    for record in records[:50]:
        cache.ingest(record, 2)
    cache.frame_lost()
    for record in records[50:]:
        cache.ingest(record, 2)
    assert cache.snapshot is None


def admin_to(to: int, payload: bytes, port: int = portnums_pb2.PortNum.ADMIN_APP) -> mesh_pb2.ToRadio:
    return mesh_pb2.ToRadio(packet=mesh_pb2.MeshPacket(to=to, decoded=mesh_pb2.Data(portnum=port, payload=payload)))


def xmodem(control: int) -> mesh_pb2.ToRadio:
    return mesh_pb2.ToRadio(xmodemPacket=xmodem_pb2.XModem(control=control))


@pytest.mark.parametrize(
    ("to_radio", "stale"),
    [
        pytest.param(
            admin_to(OWN, AdminMessage(set_channel=channel(1, "x").channel).SerializeToString()), True, id="set"
        ),
        pytest.param(admin_to(OWN, AdminMessage(store_ui_config={}).SerializeToString()), True, id="ui"),
        pytest.param(admin_to(OWN, AdminMessage(delete_file_request="/x").SerializeToString()), True, id="file"),
        pytest.param(admin_to(OWN, AdminMessage(commit_edit_settings=True).SerializeToString()), True, id="commit"),
        pytest.param(admin_to(OWN, b"\xff"), True, id="unreadable"),
        pytest.param(admin_to(OWN, b""), True, id="empty"),
        pytest.param(admin_to(OWN, AdminMessage(get_channel_request=1).SerializeToString()), False, id="get"),
        pytest.param(admin_to(OWN, AdminMessage(set_time_only=1).SerializeToString()), False, id="time"),
        pytest.param(admin_to(OWN, AdminMessage(begin_edit_settings=True).SerializeToString()), False, id="begin"),
        pytest.param(admin_to(5, AdminMessage(set_owner={}).SerializeToString()), False, id="another node"),
        pytest.param(admin_to(0xFFFFFFFF, AdminMessage(set_owner={}).SerializeToString()), True, id="broadcast"),
        pytest.param(admin_to(OWN, b"hi", portnums_pb2.PortNum.TEXT_MESSAGE_APP), False, id="text"),
        pytest.param(mesh_pb2.ToRadio(heartbeat={}), False, id="heartbeat"),
        pytest.param(xmodem(xmodem_pb2.XModem.Control.EOT), True, id="file upload finished"),
        pytest.param(xmodem(xmodem_pb2.XModem.Control.SOH), False, id="file chunk"),
    ],
)
def test_writes_that_change_the_gateway_make_the_snapshot_stale(to_radio: mesh_pb2.ToRadio, *, stale: bool) -> None:
    cache = committed()
    seen = []
    cache.bind(lambda raw, _: seen.append(raw))
    assert cache.note_write(to_radio) is stale
    assert (cache.snapshot is None) is stale
    assert seen == []  # a bound client is never told to leave by a write


def node_state_of(cache: ReplayCache) -> set[int]:
    return {getattr(FromRadio.FromString(raw).packet, "from") for raw in cache.bind(lambda *_: None).node_state}


def text_senders(cache: ReplayCache) -> list[int]:
    return [getattr(FromRadio.FromString(raw).packet, "from") for raw in texts(cache)]


@pytest.mark.parametrize(
    ("admin", "state", "senders"),
    [
        pytest.param(AdminMessage(remove_by_nodenum=5), {6}, [5, 6], id="delete"),
        pytest.param(AdminMessage(set_ignored_node=5), {6}, [6], id="ignore"),
        pytest.param(AdminMessage(nodedb_reset=True), set(), [5, 6], id="node DB reset"),
        pytest.param(AdminMessage(factory_reset_config=1), set(), [5, 6], id="factory reset"),
        pytest.param(AdminMessage(set_favorite_node=5), {5, 6}, [5, 6], id="favourite"),
    ],
)
def test_deleted_or_ignored_nodes_are_not_brought_back_by_replay(
    admin: AdminMessage, state: set[int], senders: list[int]
) -> None:
    cache = committed()
    for sender in (5, 6):
        cache.ingest(packet(sender, 1, portnums_pb2.PortNum.POSITION_APP, b""), NONCE)
        cache.ingest(telemetry(sender, 2, device_metrics={"battery_level": 50}), NONCE)
        cache.ingest(packet(sender, 3), NONCE)
    write = admin_to(OWN, admin.SerializeToString())
    cache.note_write(write)
    cache.note_write(write)  # as after the send: the same again is harmless
    for record in bootstrap(2):
        cache.ingest(record, 2)
    assert node_state_of(cache) == state
    assert text_senders(cache) == senders


def test_a_bootstrap_begun_just_after_a_write_is_not_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, "_WRITE_SETTLE_S", 2.0)
    now = [1000.0]
    monkeypatch.setattr(replay.time, "monotonic", lambda: now[0])
    cache = committed()
    failures = []
    cache.on_failure = lambda: failures.append(cache.failure)
    cache.note_write(admin_to(OWN, AdminMessage(set_owner={}).SerializeToString()))
    now[0] += 0.1  # a config request in the same firmware pass may be answered before the write is applied
    for record in bootstrap(2):
        cache.ingest(record, 2)
    assert cache.snapshot is None
    assert failures == ["the gateway was changed as the download began"]
    now[0] += 5  # the refresh after the write
    for record in bootstrap(3):
        cache.ingest(record, 3)
    assert cache.snapshot is not None


@pytest.mark.parametrize("damage", [b"\x00", b"\x94\x00", b"\x94\xc3\xff\xff"], ids=["start1", "start2", "length"])
async def test_damaged_frame_in_a_bootstrap_is_not_committed(damage: bytes) -> None:
    connection = TcpConnection("radio")
    connection._reader = asyncio.StreamReader()
    frames = [StreamingClientTransport.build_frame(r.SerializeToString()) for r in bootstrap(NONCE)]
    frames[100] = damage + frames[100][len(damage) :]  # one other node's header
    connection._reader.feed_data(b"".join(frames))
    connection._reader.feed_eof()
    async for from_radio in connection._packet_stream():
        connection.replay_cache.ingest(from_radio, NONCE)
    assert connection.replay_cache.snapshot is None
    refresh = connection._replay_refresh  # the connection downloads its configuration again after a back-off
    assert refresh is not None
    connection.cancel_replay_refresh()
    await asyncio.wait([refresh])
