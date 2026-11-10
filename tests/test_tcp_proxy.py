# SPDX-FileCopyrightText: 2026 Marc Warne <marc@kernel-panic.im>
#
# SPDX-License-Identifier: MIT

"""
Offline tests for the TCP proxy.

Home Assistant's real MeshInterface runs over a fake radio that answers its config request with a recorded
bootstrap; proxy clients are fake streams, except in the loopback test. Nothing touches a real radio.
"""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable

import pytest

from custom_components.meshtastic import meshtastic_web
from custom_components.meshtastic.aiomeshtastic import MeshInterface, replay
from custom_components.meshtastic.aiomeshtastic import connection as aio_connection
from custom_components.meshtastic.aiomeshtastic.connection import (
    ClientApiConnection,
    ClientApiConnectionInterruptedError,
)
from custom_components.meshtastic.aiomeshtastic.connection.streaming import StreamingClientTransport
from custom_components.meshtastic.aiomeshtastic.protobuf import (
    admin_pb2,
    config_pb2,
    mesh_pb2,
    portnums_pb2,
    xmodem_pb2,
)
from custom_components.meshtastic.meshtastic_tcp import server

from .mesh_fixtures import OWN, UNKNOWN_FIELD, bootstrap, channel, completion, expected_for, not_history, packet

FRAME = StreamingClientTransport.build_frame
CLIENT_NONCE = 0x5EED


@pytest.fixture(autouse=True)
def no_settle_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # these tests write and bootstrap within milliseconds; the commit guard has its own tests
    monkeypatch.setattr(replay, "_WRITE_SETTLE_S", 0)


class FakeRadio(ClientApiConnection):
    """The gateway's own upstream session. Answers want_config_id with a recorded bootstrap unless held."""

    def __init__(self, *, answer: bool = True) -> None:
        super().__init__()
        self.sent: list[mesh_pb2.ToRadio] = []
        self.raw: list[bytes] = []  # what reached the radio, byte for byte
        self.emitted: list[bytes] = []  # what the radio sent besides its bootstraps
        self.incoming: asyncio.Queue[mesh_pb2.FromRadio] = asyncio.Queue()
        self.answer = answer
        self.channel_name = "LongFast"
        self.connected = False
        self.broken = 0  # how many of the next bootstraps lack a channel, so that the cache cannot keep them
        self.cut_short = 0  # how many of the next bootstraps stop after 50 records

    async def _connect(self) -> None:
        self.connected = True

    async def _disconnect(self) -> None:
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def _packet_stream(self) -> AsyncIterable[mesh_pb2.FromRadio]:
        while self.connected:
            record = await self.incoming.get()
            if isinstance(record, Exception):
                self.connected = False  # the link dropped
                raise record
            yield record

    async def _send_packet(self, packet: bytes) -> bool:
        to_radio = mesh_pb2.ToRadio.FromString(packet)
        self.sent.append(to_radio)
        self.raw.append(packet)
        if to_radio.HasField("want_config_id") and self.answer:
            self.answer_config(to_radio.want_config_id)
        return True

    def emit(self, record: mesh_pb2.FromRadio) -> None:
        self.emitted.append(record.SerializeToString())
        self.incoming.put_nowait(record)

    def answer_config(self, nonce: int) -> None:
        broken, self.broken = self.broken > 0, max(0, self.broken - 1)
        records = bootstrap(nonce, channel_name=self.channel_name)
        if self.cut_short:
            self.cut_short -= 1
            records = records[:50]
        for record in records:
            if not (broken and record.channel.index == 7):
                self.incoming.put_nowait(record)

    def config_requests(self) -> list[int]:
        return [p.want_config_id for p in self.sent if p.HasField("want_config_id")]

    def disconnects(self) -> list[mesh_pb2.ToRadio]:
        return [p for p in self.sent if p.HasField("disconnect")]


class FakeServer:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, writer: "FakeWriter") -> None:
        self._writer = writer

    def get_extra_info(self, name: str, default: object = None) -> object:
        return ("127.0.0.1", self._writer.port) if name == "peername" else default

    def abort(self) -> None:
        self._writer.aborted = True
        self._writer.closing = True


class FakeWriter:
    _port = 50000

    def __init__(
        self,
        *,
        drain_hangs: bool = False,
        drain_error: Exception | None = None,
        close_error: Exception | None = None,
        close_hangs: bool = False,
        on_drain: Callable[[], None] | None = None,
    ) -> None:
        FakeWriter._port += 1
        self.port = FakeWriter._port
        self.data = bytearray()
        self.closing = False
        self.aborted = False
        self.drain_hangs = drain_hangs
        self.drain_error = drain_error
        self.close_error = close_error
        self.close_hangs = close_hangs
        self.on_drain = on_drain
        self.transport = FakeTransport(self)

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        if self.on_drain is not None:
            self.on_drain()
        if self.drain_error is not None:
            raise self.drain_error
        if self.drain_hangs:
            await asyncio.Event().wait()
        await asyncio.sleep(0)  # a real drain may suspend, so give everything else a turn

    def is_closing(self) -> bool:
        return self.closing

    def close(self) -> None:
        self.closing = True

    async def wait_closed(self) -> None:
        if self.close_error is not None:
            raise self.close_error
        if self.close_hangs:
            await asyncio.Event().wait()

    def frames(self) -> list[bytes]:
        frames, data = [], bytes(self.data)
        while data:
            assert data[:2] == bytes([StreamingClientTransport.START1, StreamingClientTransport.START2])
            length = (data[2] << 8) + data[3]
            frames.append(data[4 : 4 + length])
            data = data[4 + length :]
        return frames

    def received(self) -> list[mesh_pb2.FromRadio]:
        return [mesh_pb2.FromRadio.FromString(frame) for frame in self.frames()]


async def wait_until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not condition():  # noqa: ASYNC110 - the fakes expose state, not events
            await asyncio.sleep(0.01)


def send(reader: asyncio.StreamReader, to_radio: mesh_pb2.ToRadio) -> None:
    reader.feed_data(FRAME(to_radio.SerializeToString()))


class Gateway:
    def __init__(self, proxy: server.MeshtasticTcpProxy, radio: FakeRadio, interface: MeshInterface) -> None:
        self.proxy, self.radio, self.interface = proxy, radio, interface
        self.cache = interface.replay_cache
        self.clients: list[asyncio.Task] = []

    def connect(self, writer: FakeWriter | None = None) -> tuple[asyncio.StreamReader, FakeWriter, asyncio.Task]:
        reader, writer = asyncio.StreamReader(), writer or FakeWriter()
        task = asyncio.create_task(self.proxy._handle_client(reader, writer))
        self.clients.append(task)
        return reader, writer, task

    async def bootstrapped(self, nonce: int = CLIENT_NONCE, writer: FakeWriter | None = None) -> tuple:
        reader, writer, task = self.connect(writer)
        send(reader, mesh_pb2.ToRadio(want_config_id=nonce))
        await wait_until(lambda: completion(nonce).SerializeToString() in writer.frames())
        return reader, writer, task

    async def mesh(self, *records: mesh_pb2.FromRadio) -> None:
        """Deliver records through Home Assistant's own ingress and wait until it has taken them."""
        marker = packet(0xDEAD, 0xFFFF, portnums_pb2.PortNum.PRIVATE_APP)
        for record in (*records, marker):
            self.radio.emit(record)
        await wait_until(self.radio.incoming.empty)
        await asyncio.sleep(0.05)


async def start_gateway(monkeypatch: pytest.MonkeyPatch, radio: FakeRadio) -> Gateway:
    async def fake_start_server(*_args: object, **_kwargs: object) -> FakeServer:
        return FakeServer()

    monkeypatch.setattr(server.asyncio, "start_server", fake_start_server)
    interface = MeshInterface(radio, enable_mqtt_proxy=False)
    await interface.start()
    proxy = server.MeshtasticTcpProxy(interface)
    await proxy.start()
    return Gateway(proxy, radio, interface)


@pytest.fixture
async def gateway(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Gateway]:
    gateway = await start_gateway(monkeypatch, FakeRadio())
    await gateway.interface.connected_node_ready()
    await wait_until(gateway.cache.ready.is_set)
    yield gateway
    await gateway.proxy.stop()
    await gateway.interface.stop()


def texts_after_last_completion(writer: FakeWriter) -> list[tuple[int, int]]:
    received = writer.received()
    last = max(i for i, r in enumerate(received) if r.HasField("config_complete_id"))
    return [
        (getattr(r.packet, "from"), r.packet.id)
        for r in received[last + 1 :]
        if r.HasField("packet") and r.packet.decoded.portnum == portnums_pb2.PortNum.TEXT_MESSAGE_APP
    ]


@pytest.mark.parametrize("nonce", [CLIENT_NONCE, 0, 69420, 69421])
async def test_client_bootstraps_from_snapshot_without_touching_the_radio(gateway: Gateway, nonce: int) -> None:
    reader, writer, task = gateway.connect()
    await asyncio.sleep(0.05)
    assert writer.frames() == []  # silent until asked

    send(reader, mesh_pb2.ToRadio(want_config_id=nonce))
    await wait_until(lambda: len(writer.frames()) == len(expected_for(nonce)))
    assert writer.frames() == expected_for(nonce)
    if nonce == 0:
        assert writer.frames()[-1] == bytes([0x38, 0x00])  # completion present, with a nonce of zero
    if nonce != 69420:  # it rides on another node's record
        assert any(frame.endswith(UNKNOWN_FIELD) for frame in writer.frames())

    reader.feed_eof()
    await task
    assert gateway.radio.config_requests() == [gateway.radio.sent[0].want_config_id]  # only Home Assistant's own
    assert gateway.radio.disconnects() == []


async def test_second_client_does_not_disturb_the_first_or_home_assistant(gateway: Gateway) -> None:
    ha_sent = list(gateway.radio.sent)
    ha_channels = [c.SerializeToString() for c in gateway.interface.connected_node_channels()]

    reader_a, writer_a, task_a = await gateway.bootstrapped(1111)
    a_bootstrap = len(writer_a.frames())
    reader_b, writer_b, task_b = await gateway.bootstrapped(2222)
    await gateway.mesh(packet(5, 1))

    assert writer_b.frames()[: len(expected_for(2222))] == expected_for(2222)
    assert [r.config_complete_id for r in writer_a.received() if r.HasField("config_complete_id")] == [1111]
    assert [r.config_complete_id for r in writer_b.received() if r.HasField("config_complete_id")] == [2222]
    assert writer_a.received()[a_bootstrap:][0] == packet(5, 1)  # A got B's arrival live, and nothing of B's
    assert not task_a.done()

    send(reader_a, mesh_pb2.ToRadio(disconnect=True))
    send(reader_b, mesh_pb2.ToRadio(disconnect=False))  # the value is irrelevant; either ends only this client
    await asyncio.gather(task_a, task_b)
    assert writer_a.closing
    assert writer_b.closing

    # the headline: clients cost the radio no config request and no disconnect
    assert gateway.radio.sent == ha_sent
    assert gateway.radio.config_requests() == [ha_sent[0].want_config_id]
    assert gateway.radio.disconnects() == []
    assert gateway.interface.is_running
    assert gateway.interface._connected_node_ready.is_set()
    assert [c.SerializeToString() for c in gateway.interface.connected_node_channels()] == ha_channels


async def test_client_waits_for_home_assistants_bootstrap_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = await start_gateway(monkeypatch, FakeRadio(answer=False))
    try:
        await wait_until(lambda: gateway.radio.config_requests() != [])
        ha_nonce = gateway.radio.config_requests()[0]
        reader, writer, task = gateway.connect()
        send(reader, mesh_pb2.ToRadio(want_config_id=CLIENT_NONCE))
        await asyncio.sleep(0.1)
        assert writer.frames() == []

        gateway.radio.answer_config(ha_nonce)
        await gateway.interface.connected_node_ready()
        await wait_until(lambda: writer.frames() == expected_for(CLIENT_NONCE))
        assert gateway.radio.config_requests() == [ha_nonce]
        assert gateway.radio.requested_config_id == ha_nonce
        reader.feed_eof()
        await task
    finally:
        await gateway.proxy.stop()
        await gateway.interface.stop()


async def test_client_is_refused_while_there_is_no_snapshot(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(server, "_SNAPSHOT_WAIT_S", 0.1)
    gateway = await start_gateway(monkeypatch, FakeRadio(answer=False))
    try:
        reader, writer, task = gateway.connect()
        send(reader, mesh_pb2.ToRadio(want_config_id=CLIENT_NONCE))
        async with asyncio.timeout(2):
            await task
        assert writer.closing
        assert writer.frames() == []
        assert "No complete gateway bootstrap within 0.1 s, refusing proxy client" in caplog.text
        assert len(gateway.radio.config_requests()) == 1  # Home Assistant's own, never one for the client
    finally:
        await gateway.proxy.stop()
        await gateway.interface.stop()


async def test_history_replays_after_completion_with_original_identity(gateway: Gateway) -> None:
    received = [packet(5, 7), packet(6, 7), packet(9, 0)]
    await gateway.mesh(*not_history(), *received)

    reader, writer, task = await gateway.bootstrapped()
    await asyncio.sleep(0.05)  # let the replay finish; nothing else arrives meanwhile
    frames = writer.frames()
    after = frames[frames.index(completion(CLIENT_NONCE).SerializeToString()) + 1 :]
    position = not_history()[2]  # the latest position per node comes first, as firmware 2.8 sends it
    assert after == [r.SerializeToString() for r in [position, *received]]  # same bytes: id, sender, rx time, hops

    # what is not history still passes live, the MQTT proxy and file transfer included
    await gateway.mesh(*not_history())
    kinds = [r.WhichOneof("payload_variant") for r in writer.received()[len(frames) :]]
    assert "mqttClientProxyMessage" in kinds
    assert "xmodemPacket" in kinds
    assert kinds.count("packet") >= 6
    reader.feed_eof()
    await task


def admin_packet(admin: admin_pb2.AdminMessage | bytes, to: int = OWN, packet_id: int = 78) -> mesh_pb2.MeshPacket:
    payload = admin if isinstance(admin, bytes) else admin.SerializeToString()
    return mesh_pb2.MeshPacket(
        id=packet_id, to=to, decoded=mesh_pb2.Data(portnum=portnums_pb2.PortNum.ADMIN_APP, payload=payload)
    )


async def test_mesh_traffic_and_heartbeats_reach_the_radio_as_sent(gateway: Gateway) -> None:
    ha_sent = list(gateway.radio.sent)
    reader, writer, task = await gateway.bootstrapped()
    bootstrap_frames = len(writer.frames())

    sent = [
        mesh_pb2.ToRadio(
            packet=mesh_pb2.MeshPacket(id=77, to=0xFFFFFFFF, decoded=mesh_pb2.Data(portnum=1, payload=b"hi"))
        ),
        mesh_pb2.ToRadio(
            packet=mesh_pb2.MeshPacket(id=79, to=5, channel=1, decoded=mesh_pb2.Data(portnum=1, payload=b"dm"))
        ),
        mesh_pb2.ToRadio(packet=mesh_pb2.MeshPacket(id=80, to=5, encrypted=b"\x13\x37")),  # no port to inspect
        mesh_pb2.ToRadio(packet=mesh_pb2.MeshPacket(id=81, to=5, decoded=mesh_pb2.Data(portnum=4000, payload=b"?"))),
        mesh_pb2.ToRadio(heartbeat=mesh_pb2.Heartbeat(nonce=2)),  # nonce 1 would make the radio broadcast its node info
        # admin for another node, and the gateway's own reads and clock
        mesh_pb2.ToRadio(packet=admin_packet(admin_pb2.AdminMessage(set_owner=mesh_pb2.User(long_name="x")), to=5)),
        mesh_pb2.ToRadio(packet=admin_packet(admin_pb2.AdminMessage(get_owner_request=True))),
        mesh_pb2.ToRadio(packet=admin_packet(admin_pb2.AdminMessage(get_config_request=5), to=0xFFFFFFFF)),
        mesh_pb2.ToRadio(packet=admin_packet(admin_pb2.AdminMessage(set_time_only=1_762_700_000))),
    ]
    for to_radio in sent:
        send(reader, to_radio)
    await wait_until(lambda: len(gateway.radio.sent) == len(ha_sent) + len(sent))
    assert [p.SerializeToString() for p in gateway.radio.sent[len(ha_sent) :]] == [p.SerializeToString() for p in sent]

    # the radio's acknowledgement reaches the client live, and no refusal was sent
    ack = packet(OWN, 200, portnums_pb2.PortNum.ROUTING_APP, mesh_pb2.Routing(error_reason=0).SerializeToString())
    ack.packet.decoded.request_id = 77
    await gateway.mesh(ack)
    replies = writer.received()[bootstrap_frames:]
    assert not any(r.HasField("clientNotification") for r in replies)
    assert ack in replies
    assert len(gateway.radio.config_requests()) == 1  # Home Assistant's own only
    assert gateway.radio.disconnects() == []
    assert gateway.cache.ready.is_set()  # none of it changed the gateway, so this client stays connected
    assert not task.done()

    # nor is the acknowledgement replayed to a client connecting later
    _, later, _ = await gateway.bootstrapped(4444)
    assert ack not in later.received()
    reader.feed_eof()
    await task


async def test_text_sent_before_asking_for_config_is_passed_on(gateway: Gateway) -> None:
    ha_sent = list(gateway.radio.sent)
    reader, writer, task = gateway.connect()
    text = mesh_pb2.ToRadio(packet=mesh_pb2.MeshPacket(id=9, decoded=mesh_pb2.Data(portnum=1, payload=b"x")))
    send(reader, text)
    await wait_until(lambda: gateway.radio.sent[len(ha_sent) :] == [text])
    send(reader, mesh_pb2.ToRadio(want_config_id=CLIENT_NONCE))
    await wait_until(lambda: completion(CLIENT_NONCE).SerializeToString() in writer.frames())
    await asyncio.sleep(0.05)
    assert not any(r.HasField("clientNotification") for r in writer.received())
    reader.feed_eof()
    await task


def variants(message_type: type) -> list[str]:
    return [f.name for f in message_type.DESCRIPTOR.oneofs_by_name["payload_variant"].fields]


def sample(message_type: type, variant: str) -> object:
    """Return a message with only this oneof field set, to a value that is not the default where it is a scalar."""
    message = message_type()
    field = message_type.DESCRIPTOR.fields_by_name[variant]
    if field.message_type is not None:
        getattr(message, variant).SetInParent()
    elif field.type == field.TYPE_BOOL:
        setattr(message, variant, True)
    elif field.type == field.TYPE_STRING:
        setattr(message, variant, "x")
    elif field.type == field.TYPE_BYTES:
        setattr(message, variant, b"x")
    else:
        setattr(message, variant, 1)
    return message


# the guard against taking anything away from clients: every kind but these two must reach the radio as sent,
# whatever a protobuf update adds. Not taken from the proxy, so that growing its list fails here
LOCAL = {"want_config_id", "disconnect"}


def test_only_config_requests_and_disconnect_stay_with_the_proxy() -> None:
    assert server._NOT_FORWARDED == LOCAL, "the proxy no longer passes every ToRadio kind on to the radio"
    assert set(variants(mesh_pb2.ToRadio)) > LOCAL


def from_client(gateway: Gateway, writer: FakeWriter, start: int) -> list[mesh_pb2.FromRadio]:
    """Return what this client received since start that the radio did not send: replies the proxy made up."""
    return [r for r in writer.received()[start:] if r.SerializeToString() not in gateway.radio.emitted]


@pytest.mark.parametrize("variant", sorted(set(variants(mesh_pb2.ToRadio)) - LOCAL))
async def test_every_to_radio_kind_reaches_the_radio_as_sent(gateway: Gateway, variant: str) -> None:
    reader, writer, task = await gateway.bootstrapped()
    start, sent = len(writer.frames()), len(gateway.radio.raw)
    data = sample(mesh_pb2.ToRadio, variant).SerializeToString()
    reader.feed_data(FRAME(data))
    await wait_until(lambda: len(gateway.radio.raw) > sent)
    await asyncio.sleep(0.05)
    assert gateway.radio.raw[sent:] == [data], f"a client's {variant} no longer reaches the radio as sent"
    assert from_client(gateway, writer, start) == []
    assert not task.done()
    reader.feed_eof()
    await task


async def test_a_kind_newer_than_this_code_reaches_the_radio_as_sent(gateway: Gateway) -> None:
    reader, writer, task = await gateway.bootstrapped()
    start, sent = len(writer.frames()), len(gateway.radio.raw)
    newer = bytes([0xBA, 0x3E, 0x00])  # field 999
    reader.feed_data(FRAME(newer))
    await wait_until(lambda: len(gateway.radio.raw) > sent)
    assert gateway.radio.raw[sent:] == [newer]
    assert from_client(gateway, writer, start) == []
    reader.feed_eof()
    await task


@pytest.mark.parametrize("to", [OWN, 0xFFFFFFFF, 5, None], ids=["gateway", "broadcast", "another node", "unknown"])
async def test_every_admin_message_reaches_the_radio_as_sent(gateway: Gateway, to: int | None) -> None:
    _, idle_writer, idle = await gateway.bootstrapped(1)
    reader, writer, task = await gateway.bootstrapped()
    if to is None:  # the gateway's node number is not known yet
        gateway.cache.my_node_num, to = None, OWN
    payloads = [sample(admin_pb2.AdminMessage, v).SerializeToString() for v in variants(admin_pb2.AdminMessage)]
    payloads += [b"", b"\xff\xff"]  # empty, and not readable
    frames = [
        mesh_pb2.ToRadio(packet=admin_packet(payload, to=to, packet_id=1000 + i)).SerializeToString()
        for i, payload in enumerate(payloads)
    ]
    start, idle_start, sent = len(writer.frames()), len(idle_writer.frames()), len(gateway.radio.raw)
    reader.feed_data(b"".join(FRAME(frame) for frame in frames))
    await wait_until(lambda: len(gateway.radio.raw) >= sent + len(frames))
    await asyncio.sleep(0.05)

    assert gateway.radio.raw[sent:] == frames
    assert from_client(gateway, writer, start) == []  # no routing error, queue status or notification of its own
    assert from_client(gateway, idle_writer, idle_start) == []
    assert not task.done()  # neither the writer nor anyone else is cut off by a write
    assert not idle.done()
    assert gateway.radio.config_requests() == gateway.radio.config_requests()[:1]  # no refresh yet: 5 s after
    reader.feed_eof()
    await task


# what the cache keeps for the bootstrap it serves itself, so never passes live
BOOTSTRAP_KINDS = {
    "my_info", "deviceuiConfig", "metadata", "channel", "config", "moduleConfig", "node_info", "fileInfo",
    "region_presets", "config_complete_id", "rebooted",
}  # fmt: skip


def test_file_transfer_and_mqtt_proxy_messages_pass_live() -> None:
    live = set(variants(mesh_pb2.FromRadio)) - BOOTSTRAP_KINDS
    assert {"packet", "xmodemPacket", "mqttClientProxyMessage", "queueStatus", "clientNotification"} <= live


@pytest.mark.parametrize("variant", sorted(set(variants(mesh_pb2.FromRadio)) - BOOTSTRAP_KINDS))
async def test_every_live_from_radio_kind_reaches_a_client(gateway: Gateway, variant: str) -> None:
    reader, writer, task = await gateway.bootstrapped()
    record = sample(mesh_pb2.FromRadio, variant)
    if variant == "queueStatus":
        record = mesh_pb2.FromRadio(queueStatus=mesh_pb2.QueueStatus(free=16, maxlen=16))  # empty would mean full
    await gateway.mesh(record)
    assert record in writer.received(), f"the radio's {variant} no longer reaches clients"
    reader.feed_eof()
    await task


def admin_write(admin: admin_pb2.AdminMessage, packet_id: int = 78, to: int = OWN) -> mesh_pb2.ToRadio:
    return mesh_pb2.ToRadio(packet=admin_packet(admin, to=to, packet_id=packet_id))


BEGIN = admin_pb2.AdminMessage(begin_edit_settings=True)
COMMIT = admin_pb2.AdminMessage(commit_edit_settings=True)
SET_LORA = admin_pb2.AdminMessage(set_config=config_pb2.Config(lora=config_pb2.Config.LoRaConfig(hop_limit=5)))


async def test_settings_changed_from_a_phone_reach_the_radio_and_keep_everyone_connected(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.2,))
    _, idle_writer, idle = await gateway.bootstrapped(1)
    reader, writer, task = await gateway.bootstrapped()
    sent = len(gateway.radio.raw)
    transaction = [
        admin_write(BEGIN, 101),
        admin_write(SET_LORA, 102),
        admin_write(admin_pb2.AdminMessage(set_channel=channel(1, "Second").channel), 103),
        admin_write(admin_pb2.AdminMessage(set_favorite_node=5), 104),
        admin_write(COMMIT, 105),
    ]
    for to_radio in transaction:
        send(reader, to_radio)
    await wait_until(lambda: len(gateway.radio.raw) >= sent + len(transaction))
    assert gateway.radio.raw[sent:] == [t.SerializeToString() for t in transaction]

    # one download refreshes the snapshot once the writes stop, and nobody connected was closed by it
    await wait_until(lambda: len(gateway.radio.config_requests()) == 2 and gateway.cache.ready.is_set())
    await gateway.mesh(packet(5, 1))
    for client, client_writer in ((task, writer), (idle, idle_writer)):
        assert not client.done()
        assert texts_after_last_completion(client_writer) == [(5, 1)]
        assert sum(r.HasField("my_info") for r in client_writer.received()) == 1  # none of the refresh's
    assert gateway.radio.disconnects() == []
    await gateway.bootstrapped(4444)  # a client connecting now is served the new snapshot
    assert len(gateway.radio.config_requests()) == 2
    reader.feed_eof()
    await task


async def test_a_transaction_across_connections_as_the_cli_makes_it(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05,))
    sent = len(gateway.radio.sent)
    for admin in (BEGIN, SET_LORA, COMMIT):  # meshtastic --begin-edit, --set, --commit-edit: one connection each
        reader, _, task = await gateway.bootstrapped()
        send(reader, admin_write(admin))
        send(reader, mesh_pb2.ToRadio(disconnect=True))
        async with asyncio.timeout(2):
            await task
    writes = [p for p in gateway.radio.sent[sent:] if p.HasField("packet")]
    assert [replay.gateway_admin(p, OWN) for p in writes] == [BEGIN, SET_LORA, COMMIT]  # and nothing of our own
    assert gateway.radio.disconnects() == []


async def test_file_transfer_and_mqtt_proxying_pass_both_ways(gateway: Gateway) -> None:
    reader, writer, task = await gateway.bootstrapped()
    sent = len(gateway.radio.raw)
    chunk = mesh_pb2.ToRadio(xmodemPacket=xmodem_pb2.XModem(control=xmodem_pb2.XModem.Control.SOH, seq=1, buffer=b"d"))
    uplink = mesh_pb2.ToRadio(mqttClientProxyMessage=mesh_pb2.MqttClientProxyMessage(topic="msh/x", data=b"x"))
    send(reader, chunk)
    send(reader, uplink)
    await wait_until(lambda: len(gateway.radio.raw) == sent + 2)
    assert gateway.radio.raw[sent:] == [chunk.SerializeToString(), uplink.SerializeToString()]
    assert gateway.cache.ready.is_set()  # a chunk changes nothing yet

    replies = [
        mesh_pb2.FromRadio(xmodemPacket=xmodem_pb2.XModem(control=xmodem_pb2.XModem.Control.ACK, seq=1)),
        mesh_pb2.FromRadio(mqttClientProxyMessage=mesh_pb2.MqttClientProxyMessage(topic="msh/y", data=b"y")),
    ]
    await gateway.mesh(*replies)
    assert all(reply in writer.received() for reply in replies)

    send(reader, mesh_pb2.ToRadio(xmodemPacket=xmodem_pb2.XModem(control=xmodem_pb2.XModem.Control.EOT)))
    await wait_until(lambda: not gateway.cache.ready.is_set())  # the upload finished: the file list changed
    assert not task.done()
    reader.feed_eof()
    await task


async def test_two_stage_handshake_as_android_does_it(gateway: Gateway) -> None:
    await gateway.mesh(*(packet(5, i) for i in range(1, 6)))
    ha_sent = list(gateway.radio.sent)
    reader, writer, task = gateway.connect()

    def admin(message: admin_pb2.AdminMessage, packet_id: int) -> mesh_pb2.ToRadio:
        return mesh_pb2.ToRadio(packet=admin_packet(message, packet_id=packet_id))

    send(reader, mesh_pb2.ToRadio(heartbeat=mesh_pb2.Heartbeat(nonce=0)))
    send(reader, mesh_pb2.ToRadio(want_config_id=replay.NONCE_ONLY_CONFIG))
    await wait_until(lambda: any(r.HasField("my_info") for r in writer.received()))
    send(reader, admin(admin_pb2.AdminMessage(set_time_only=1_762_700_000), 1001))  # on receiving my_info
    await wait_until(lambda: completion(replay.NONCE_ONLY_CONFIG).SerializeToString() in writer.frames())
    await gateway.mesh(packet(5, 6))  # between the stages
    send(reader, mesh_pb2.ToRadio(want_config_id=replay.NONCE_ONLY_NODES))
    await wait_until(lambda: completion(replay.NONCE_ONLY_NODES).SerializeToString() in writer.frames())
    # after the handshake: the owner, then a settings screen's reads
    send(reader, admin(admin_pb2.AdminMessage(get_owner_request=True), 1002))
    send(reader, admin(admin_pb2.AdminMessage(get_channel_request=1), 1003))
    send(reader, admin(admin_pb2.AdminMessage(get_config_request=admin_pb2.AdminMessage.LORA_CONFIG), 1004))
    await wait_until(lambda: len(gateway.radio.sent) == len(ha_sent) + 5)
    await gateway.mesh(packet(5, 7))

    received = writer.received()
    assert [r.config_complete_id for r in received if r.HasField("config_complete_id")] == [69420, 69421]
    texts = [r.packet.id for r in received if r.packet.decoded.portnum == portnums_pb2.PortNum.TEXT_MESSAGE_APP]
    assert texts == [1, 2, 3, 4, 5, 6, 7]  # each once over the whole handshake
    assert not any(r.HasField("clientNotification") or r.HasField("queueStatus") for r in received)
    assert [p.WhichOneof("payload_variant") for p in gateway.radio.sent[len(ha_sent) :]] == ["heartbeat"] + [
        "packet"
    ] * 4
    assert gateway.radio.config_requests() == [gateway.radio.config_requests()[0]]  # Home Assistant's own only
    assert gateway.radio.disconnects() == []
    assert not task.done()
    reader.feed_eof()
    await task


async def test_text_being_written_during_a_repeated_request_is_not_sent_again(gateway: Gateway) -> None:
    await gateway.mesh(packet(5, 700), packet(5, 701))
    release = asyncio.Event()
    held = packet(5, 700).SerializeToString()

    class HoldingWriter(FakeWriter):
        async def drain(self) -> None:
            if self.frames()[-1] == held and not release.is_set():
                await release.wait()  # written, not yet drained
            await super().drain()

    reader, writer, task = gateway.connect(HoldingWriter())
    send(reader, mesh_pb2.ToRadio(want_config_id=1))
    await wait_until(lambda: held in writer.frames())
    send(reader, mesh_pb2.ToRadio(want_config_id=2))
    await wait_until(lambda: completion(2).SerializeToString() in writer.frames())
    release.set()
    await asyncio.sleep(0.05)
    text = portnums_pb2.PortNum.TEXT_MESSAGE_APP
    assert [r.packet.id for r in writer.received() if r.packet.decoded.portnum == text] == [700, 701]
    reader.feed_eof()
    await task


async def test_live_record_arriving_while_a_repeated_request_rebinds_is_kept(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    ack = packet(OWN, 200, portnums_pb2.PortNum.ROUTING_APP, mesh_pb2.Routing(error_reason=0).SerializeToString())
    bind, binds = gateway.cache.bind, []

    def bind_after_an_arrival(*args: object) -> object:
        binds.append(1)
        if len(binds) == 2:  # between the old bootstrap stopping and the new one binding
            gateway.cache.ingest(ack, gateway.radio.requested_config_id)
        return bind(*args)

    monkeypatch.setattr(gateway.cache, "bind", bind_after_an_arrival)
    reader, writer, task = await gateway.bootstrapped(1)
    send(reader, mesh_pb2.ToRadio(want_config_id=2))
    await wait_until(lambda: ack in writer.received())
    assert writer.received().count(ack) == 1
    reader.feed_eof()
    await task


async def test_repeated_request_keeps_queued_live_records(gateway: Gateway) -> None:
    release = asyncio.Event()
    paused = [False]

    class PausingWriter(FakeWriter):
        async def drain(self) -> None:
            if paused[0]:
                await release.wait()
            await super().drain()

    reader, writer, task = await gateway.bootstrapped(writer=PausingWriter())
    paused[0] = True
    ack = packet(OWN, 200, portnums_pb2.PortNum.ROUTING_APP, mesh_pb2.Routing(error_reason=0).SerializeToString())
    await gateway.mesh(ack, packet(5, 1), packet(5, 2))  # the first is being written, the rest wait in its queue
    send(reader, mesh_pb2.ToRadio(want_config_id=4321))
    await asyncio.sleep(0.05)
    paused[0] = False
    release.set()
    await wait_until(lambda: completion(4321).SerializeToString() in writer.frames())
    await asyncio.sleep(0.05)

    received = writer.received()
    texts = [r.packet.id for r in received if r.packet.decoded.portnum == portnums_pb2.PortNum.TEXT_MESSAGE_APP]
    assert texts == [1, 2]
    assert received.count(ack) == 1
    reader.feed_eof()
    await task


async def test_every_text_crosses_the_replay_live_boundary_once(gateway: Gateway) -> None:
    ingested = list(range(10))
    await gateway.mesh(*(packet(5, i) for i in ingested))
    reader = asyncio.StreamReader()
    writes = [0]

    def arrive(packet_id: int) -> None:
        ingested.append(packet_id)
        gateway.cache.ingest(packet(5, packet_id), gateway.radio.requested_config_id)

    def on_drain() -> None:
        # an arrival at every await of both bootstraps, their history replays and the switch to live,
        # with the second request arriving part way through the first one's history replay
        writes[0] += 1
        if writes[0] <= 900:
            arrive(1000 + writes[0])
        if writes[0] == 300:
            send(reader, mesh_pb2.ToRadio(want_config_id=7002))

    writer = FakeWriter(on_drain=on_drain)
    task = asyncio.create_task(gateway.proxy._handle_client(reader, writer))
    arrive(999)  # just before the request
    send(reader, mesh_pb2.ToRadio(want_config_id=7001))
    await wait_until(lambda: writes[0] > 900)
    arrive(5000)  # after it all
    await wait_until(lambda: texts_after_last_completion(writer)[-1:] == [(5, 5000)])

    second = [packet_id for _, packet_id in texts_after_last_completion(writer)]
    assert second == ingested[-len(second) :]
    assert len(second) > replay.HISTORY_SIZE
    # over both requests: nothing twice, in arrival order. The second replays only what the first had not yet
    # written, so all that can be missing is one run of texts pushed out of the ring before either wrote them
    text = portnums_pb2.PortNum.TEXT_MESSAGE_APP
    ids = [r.packet.id for r in writer.received() if r.packet.decoded.portnum == text]
    assert ids == [i for i in ingested if i in ids]
    missing = [i for i in ingested if i not in ids]
    assert missing == ingested[ingested.index(missing[0]) :][: len(missing)]
    assert len(missing) < len(ingested) - replay.HISTORY_SIZE
    frames = writer.frames()
    first = writer.received()[
        frames.index(completion(7001).SerializeToString()) : frames.index(completion(7002).SerializeToString())
    ]
    first_ids = [r.packet.id for r in first if r.HasField("packet")]
    assert first_ids
    assert first_ids == ingested[ingested.index(first_ids[0]) :][: len(first_ids)]
    reader.feed_eof()
    await task


async def test_upstream_reboot_closes_clients_and_home_assistants_reconnect_reenables(gateway: Gateway) -> None:
    _, writer, task = await gateway.bootstrapped()
    await gateway.mesh(packet(5, 1))
    gateway.radio.incoming.put_nowait(mesh_pb2.FromRadio(rebooted=True))
    async with asyncio.timeout(2):
        await task
    assert writer.closing

    # Home Assistant's own restart bootstraps again, and clients are served from that
    await wait_until(lambda: len(gateway.radio.config_requests()) == 2 and gateway.cache.ready.is_set())
    _, writer, _ = await gateway.bootstrapped(4444)
    assert writer.frames()[: len(expected_for(4444))] == expected_for(4444)
    await gateway.mesh(packet(5, 2))
    assert texts_after_last_completion(writer) == [(5, 1), (5, 2)]  # same identity, so history survives a reboot


async def test_unloading_the_entry_closes_clients_and_drops_the_snapshot(gateway: Gateway) -> None:
    # reconfiguring an entry reloads it: unload stops the interface first, and the reload builds a new connection
    # with its own empty cache, so nothing from the old address can be served
    _, writer, task = await gateway.bootstrapped()
    await gateway.interface.stop()
    async with asyncio.timeout(2):
        await task
    assert writer.closing
    assert not gateway.cache.ready.is_set()
    assert gateway.radio._replay_refresh is None


async def test_reconnect_with_new_channels_clears_history(gateway: Gateway) -> None:
    await gateway.mesh(packet(5, 1))
    _, writer, task = await gateway.bootstrapped()
    gateway.radio.channel_name = "Private"
    gateway.radio.incoming.put_nowait(ClientApiConnectionInterruptedError("link dropped"))
    async with asyncio.timeout(2):
        await task  # closed as soon as the reconnect starts, not served a mixed generation

    # Home Assistant's own reconnect requests config again, and that bootstrap becomes the snapshot
    await wait_until(lambda: len(gateway.radio.config_requests()) == 2 and gateway.cache.ready.is_set())
    _, writer, _ = await gateway.bootstrapped()
    assert [r for r in writer.received() if r.HasField("channel")] == [channel(i, "Private") for i in range(8)]
    await gateway.mesh(packet(5, 2))
    assert texts_after_last_completion(writer) == [(5, 2)]


@pytest.mark.parametrize(
    ("admin", "stale"),
    [
        pytest.param(admin_pb2.AdminMessage(set_owner=mesh_pb2.User(long_name="x")), True, id="set_owner"),
        pytest.param(admin_pb2.AdminMessage(remove_by_nodenum=5), True, id="remove node"),
        pytest.param(admin_pb2.AdminMessage(set_time_only=1), False, id="set time"),
        pytest.param(admin_pb2.AdminMessage(get_owner_request=True), False, id="get owner"),
    ],
)
async def test_home_assistants_own_setting_writes_make_the_snapshot_stale(
    gateway: Gateway, admin: admin_pb2.AdminMessage, *, stale: bool
) -> None:
    reader, _, task = await gateway.bootstrapped()
    await gateway.interface.send_admin_message(OWN, admin, ack=False)
    await asyncio.sleep(0.05)
    assert not task.done()  # a client already connected keeps its session
    assert gateway.cache.ready.is_set() is not stale
    reader.feed_eof()
    await task


async def test_setting_write_downloads_a_fresh_snapshot_once_writes_stop(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    # as Home Assistant's own timezone write one second after connecting, when the radio's differs
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.2,))
    _, writer, task = await gateway.bootstrapped()
    for tz in ("GMT0", "CET-1CEST,M3.5.0,M10.5.0/3"):
        admin = admin_pb2.AdminMessage(set_config={"device": {"tzdef": tz}})
        await gateway.interface.send_admin_message(OWN, admin, ack=False)

    # a client arriving meanwhile waits for the fresh download instead of being refused until the next reconnect
    reader, waiting_writer, waiting = gateway.connect()
    send(reader, mesh_pb2.ToRadio(want_config_id=4444))
    await wait_until(lambda: completion(4444).SerializeToString() in waiting_writer.frames())
    assert len(gateway.radio.config_requests()) == 2  # Home Assistant's own, then one refresh for both writes
    _, _, other = await gateway.bootstrapped()
    assert len(gateway.radio.config_requests()) == 2
    await gateway.mesh(packet(5, 1))

    # the client connected before the writes kept its session, and saw none of the refresh's bootstrap
    assert not task.done()
    assert sum(r.HasField("my_info") for r in writer.received()) == 1
    assert texts_after_last_completion(writer) == [(5, 1)]
    assert gateway.radio.disconnects() == []
    assert not waiting.done()
    assert not other.done()


async def test_a_download_the_cache_cannot_keep_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05, 0.05, 0.05))
    radio = FakeRadio()
    radio.broken = 3  # Home Assistant's own download at startup, then two retries
    gateway = await start_gateway(monkeypatch, radio)
    try:
        await wait_until(gateway.cache.ready.is_set)
        assert len(radio.config_requests()) == 4
        _, _, task = await gateway.bootstrapped()
        assert not task.done()
        assert radio.disconnects() == []
    finally:
        await gateway.proxy.stop()
        await gateway.interface.stop()


async def test_a_download_that_never_completes_is_retried(gateway: Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05, 0.05, 0.05))
    monkeypatch.setattr(aio_connection, "_REPLAY_DOWNLOAD_TIMEOUT_S", 1.0)
    gateway.radio.cut_short = 2
    await gateway.interface.send_admin_message(OWN, admin_pb2.AdminMessage(set_owner={"long_name": "x"}), ack=False)
    await wait_until(gateway.cache.ready.is_set)
    assert len(gateway.radio.config_requests()) == 4  # Home Assistant's own, two cut short, then a whole one


async def test_retries_stop_with_an_actionable_warning(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05, 0.05))
    gateway.radio.broken = 100
    await gateway.interface.send_admin_message(OWN, admin_pb2.AdminMessage(set_owner={"long_name": "x"}), ack=False)
    await wait_until(lambda: "firmware 2.5 or later" in caplog.text)
    assert "channels [0, 1, 2, 3, 4, 5, 6]" in caplog.text
    await asyncio.sleep(0.2)
    assert len(gateway.radio.config_requests()) == 3  # Home Assistant's own, then one per retry, then none
    assert not gateway.cache.ready.is_set()


async def test_reconnect_supersedes_a_pending_refresh(gateway: Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.2,))
    await gateway.interface.send_admin_message(OWN, admin_pb2.AdminMessage(set_owner={"long_name": "x"}), ack=False)
    gateway.radio.answer = False  # the reconnect's download is still outstanding when the refresh would be due
    gateway.radio.incoming.put_nowait(ClientApiConnectionInterruptedError("link dropped"))
    await wait_until(lambda: len(gateway.radio.config_requests()) == 2)
    await asyncio.sleep(0.4)
    assert len(gateway.radio.config_requests()) == 2  # the reconnect's own download, never a competing one
    gateway.radio.answer_config(gateway.radio.config_requests()[-1])
    await wait_until(gateway.cache.ready.is_set)


async def test_slow_reader_is_disconnected_alone(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(server, "_CLIENT_QUEUE_SIZE", 4)
    wedged_reader, wedged_writer, wedged = gateway.connect(FakeWriter(drain_hangs=True))
    send(wedged_reader, mesh_pb2.ToRadio(want_config_id=1))
    reader, writer, task = await gateway.bootstrapped()
    await wait_until(lambda: wedged_writer.frames() != [])  # stuck on its first frame

    await gateway.mesh(*(packet(5, i) for i in range(10)))
    async with asyncio.timeout(2):
        await wedged
    assert wedged_writer.closing
    assert f"Proxy client 127.0.0.1:{wedged_writer.port} is not keeping up, disconnecting it" in caplog.text
    assert texts_after_last_completion(writer) == [(5, i) for i in range(10)]
    assert not task.done()
    reader.feed_eof()
    await task


@pytest.mark.parametrize(
    "writer",
    [
        pytest.param(FakeWriter(drain_hangs=True), id="slow peer"),
        pytest.param(FakeWriter(drain_error=ConnectionResetError()), id="dead peer write"),
    ],
)
async def test_unwritable_peer_is_disconnected(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, writer: FakeWriter
) -> None:
    monkeypatch.setattr(server, "_WRITE_TIMEOUT_S", 0.05)
    reader, writer, task = gateway.connect(writer)
    send(reader, mesh_pb2.ToRadio(want_config_id=1))
    async with asyncio.timeout(2):
        await task
    assert writer.closing
    assert gateway.cache._subscribers == set()
    if writer.drain_hangs:
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert warnings == [
            f"Proxy client 127.0.0.1:{writer.port} did not take a frame within 0.05 s, disconnecting it"
        ]


async def test_flooding_sender_holds_up_nobody_else(gateway: Gateway) -> None:
    flood_reader, _, flood = await gateway.bootstrapped(1)
    reader, writer, task = await gateway.bootstrapped(2)
    sent = len(gateway.radio.raw)
    text = mesh_pb2.ToRadio(packet=mesh_pb2.MeshPacket(id=9, to=0xFFFFFFFF, decoded=mesh_pb2.Data(portnum=1)))
    flood_reader.feed_data(FRAME(text.SerializeToString()) * 2000)  # passed on as upstream did; the radio decides

    await gateway.mesh(packet(5, 1))
    assert texts_after_last_completion(writer) == [(5, 1)]
    await wait_until(lambda: len(gateway.radio.raw) == sent + 2000)
    assert not task.done()
    assert not flood.done()
    reader.feed_eof()
    await task


async def test_dead_peer_eof_closes_handler(gateway: Gateway) -> None:
    reader, writer, task = await gateway.bootstrapped()
    reader.feed_eof()
    async with asyncio.timeout(2):
        await task
    assert writer.closing
    assert gateway.cache._subscribers == set()


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("bootstrapped", [True, False])
async def test_disconnect_is_never_forwarded(gateway: Gateway, value: bool, bootstrapped: bool) -> None:  # noqa: FBT001
    reader, writer, handler = await gateway.bootstrapped() if bootstrapped else gateway.connect()
    _, other_writer, other = await gateway.bootstrapped()

    to_radio = mesh_pb2.ToRadio(disconnect=value)
    assert to_radio.HasField("disconnect")
    send(reader, to_radio)
    async with asyncio.timeout(2):
        await handler

    assert writer.closing
    assert gateway.radio.disconnects() == []
    assert not other.done()
    assert not other_writer.closing


@pytest.mark.parametrize(
    "writer",
    [
        pytest.param(FakeWriter(close_error=ConnectionResetError()), id="broken close"),
        pytest.param(FakeWriter(close_hangs=True), id="hanging close"),
    ],
)
async def test_failed_close_falls_back_to_abort(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch, writer: FakeWriter
) -> None:
    monkeypatch.setattr(server, "_CLOSE_TIMEOUT_S", 0.05)
    reader, writer, task = gateway.connect(writer)
    reader.feed_eof()
    async with asyncio.timeout(2):
        await task
    assert writer.aborted


async def test_is_connected_reflects_transport_state() -> None:
    connection = server.ClientProxyTransport(asyncio.StreamReader(), FakeWriter())
    assert connection.is_connected
    await connection.disconnect()
    assert not connection.is_connected


async def test_stop_is_bounded_with_wedged_clients(gateway: Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_CLOSE_TIMEOUT_S", 0.05)
    writers = [FakeWriter(drain_hangs=True, close_hangs=True) for _ in range(3)]
    tasks = []
    for writer in writers:
        reader, _, task = gateway.connect(writer)
        send(reader, mesh_pb2.ToRadio(want_config_id=1))
        tasks.append(task)
    await wait_until(lambda: all(w.data for w in writers))

    async with asyncio.timeout(1):
        await gateway.proxy.stop()
    assert all(t.done() for t in tasks)
    assert all(w.aborted for w in writers)
    assert gateway.proxy._server.closed
    assert gateway.proxy._handlers == set()
    assert gateway.cache._subscribers == set()


async def test_client_arriving_during_stop_is_refused(gateway: Gateway) -> None:
    # accepted before stop(), but first runs while stop() is under way
    _, writer, task = gateway.connect()
    await gateway.proxy.stop()
    async with asyncio.timeout(1):
        await task
    assert writer.closing
    assert gateway.proxy._handlers == set()


@pytest.mark.usefixtures("socket_enabled")  # pytest-homeassistant-custom-component blocks sockets
async def test_loopback_bootstrap_history_live_and_disconnect() -> None:
    """Real asyncio server, socket and framing on loopback; only the radio is faked."""
    radio = FakeRadio()
    interface = MeshInterface(radio, enable_mqtt_proxy=False)
    await interface.start()
    await interface.connected_node_ready()
    proxy = server.MeshtasticTcpProxy(interface, host="127.0.0.1")
    proxy._port = 0  # the constructor maps 0 to the default port; let the OS choose one
    await proxy.start()
    try:
        radio.incoming.put_nowait(packet(5, 1))
        await wait_until(lambda: interface.replay_cache._history)
        port = proxy._server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(FRAME(mesh_pb2.ToRadio(want_config_id=7).SerializeToString()))
        await writer.drain()

        async def frame() -> bytes:
            header = await reader.readexactly(4)
            assert header[:2] == bytes([StreamingClientTransport.START1, StreamingClientTransport.START2])
            return await reader.readexactly((header[2] << 8) + header[3])

        expected = [*expected_for(7), packet(5, 1).SerializeToString(), packet(5, 2).SerializeToString()]
        received = []
        async with asyncio.timeout(5):
            while len(received) < len(expected):
                received.append(await frame())
                if len(received) == len(expected) - 1:
                    radio.incoming.put_nowait(packet(5, 2))  # live, after the history
        assert received == expected

        writer.write(FRAME(mesh_pb2.ToRadio(disconnect=True).SerializeToString()))
        await writer.drain()
        async with asyncio.timeout(5):
            assert await reader.read() == b""
        assert radio.config_requests() == [radio.sent[0].want_config_id]
        assert radio.disconnects() == []
        writer.close()
        await writer.wait_closed()
    finally:
        await proxy.stop()
        await interface.stop()


async def test_write_pipelined_behind_a_config_request_is_passed_on(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05,))
    sent = len(gateway.radio.raw)
    reader, writer, task = gateway.connect()
    write = admin_write(admin_pb2.AdminMessage(set_owner=mesh_pb2.User(long_name="x")), packet_id=9)
    config = mesh_pb2.ToRadio(want_config_id=CLIENT_NONCE)
    reader.feed_data(FRAME(config.SerializeToString()) + FRAME(write.SerializeToString()))  # one burst

    await wait_until(lambda: completion(CLIENT_NONCE).SerializeToString() in writer.frames())
    assert gateway.radio.raw[sent] == write.SerializeToString()
    assert writer.frames()[: len(expected_for(CLIENT_NONCE))] == expected_for(CLIENT_NONCE)  # the snapshot after it
    assert from_client(gateway, writer, 0) == [mesh_pb2.FromRadio.FromString(f) for f in expected_for(CLIENT_NONCE)]
    reader.feed_eof()
    await task


class FakeWebContext:
    def __init__(self, connection: ClientApiConnection) -> None:
        self.connection = connection

    def get_connection(self, _request: object) -> ClientApiConnection:
        return self.connection

    def add_session(self, _request: object, _config_id: str) -> None:
        pass


class FakeWebRequest:
    content_type = "application/x-protobuf"

    def __init__(self, to_radio: mesh_pb2.ToRadio) -> None:
        self.body = to_radio.SerializeToString()

    async def read(self) -> bytes:
        return self.body


async def test_bundled_web_client_writes_and_config_requests_reach_the_cache(gateway: Gateway) -> None:
    view = web_view(gateway)
    _, _, task = await gateway.bootstrapped()

    admin = admin_pb2.AdminMessage(set_owner=mesh_pb2.User(long_name="renamed"))
    set_owner = mesh_pb2.ToRadio(
        packet=mesh_pb2.MeshPacket(
            to=OWN, decoded=mesh_pb2.Data(portnum=portnums_pb2.PortNum.ADMIN_APP, payload=admin.SerializeToString())
        )
    )
    await view.put(FakeWebRequest(set_owner), "entry", "token")
    assert gateway.radio.sent[-1] == set_owner
    assert not gateway.cache.ready.is_set()
    assert not task.done()

    # the web client's own config request restarts the gateway's session; that bootstrap serves the next client
    await view.put(FakeWebRequest(mesh_pb2.ToRadio(want_config_id=31337)), "entry", "token")
    await wait_until(gateway.cache.ready.is_set)
    _, writer, _ = await gateway.bootstrapped()
    assert writer.frames()[: len(expected_for(CLIENT_NONCE))] == expected_for(CLIENT_NONCE)


async def test_snapshot_completed_while_an_admin_write_waits_for_the_radio_is_dropped(gateway: Gateway) -> None:
    await gateway.bootstrapped()
    radio = gateway.radio
    radio._queue_status = mesh_pb2.QueueStatus(free=0)
    admin = admin_pb2.AdminMessage(set_owner=mesh_pb2.User(long_name="renamed"))
    write = mesh_pb2.ToRadio(
        packet=mesh_pb2.MeshPacket(
            to=OWN, decoded=mesh_pb2.Data(portnum=portnums_pb2.PortNum.ADMIN_APP, payload=admin.SerializeToString())
        )
    )
    sending = asyncio.create_task(radio.send_packet(write))
    await asyncio.sleep(0)
    assert not gateway.cache.ready.is_set()

    for record in bootstrap(radio.requested_config_id):  # a download the radio answered before taking the write
        radio.replay_cache.ingest(record, radio.requested_config_id)
    assert gateway.cache.ready.is_set()
    radio._queue_status.free = 1
    radio._queue_status_update.set()
    await sending
    assert radio.sent[-1] == write
    assert not gateway.cache.ready.is_set()


def set_owner_write() -> mesh_pb2.ToRadio:
    admin = admin_pb2.AdminMessage(set_owner=mesh_pb2.User(long_name="renamed"))
    return mesh_pb2.ToRadio(
        packet=mesh_pb2.MeshPacket(
            to=OWN, decoded=mesh_pb2.Data(portnum=portnums_pb2.PortNum.ADMIN_APP, payload=admin.SerializeToString())
        )
    )


def web_view(gateway: Gateway) -> meshtastic_web.MeshtasticWebApiV1ToRadioView:
    view = meshtastic_web.MeshtasticWebApiV1ToRadioView(None, FakeWebContext(gateway.radio), None)
    view._check_access = lambda *_: None
    return view


def hold_sends(radio: FakeRadio) -> tuple[asyncio.Event, asyncio.Event]:
    """Make the radio's next sends wait, as a busy BLE write or transport lock does."""
    original, entered, release = radio._send_packet, asyncio.Event(), asyncio.Event()

    async def held(data: bytes) -> bool:
        entered.set()
        await release.wait()
        return await original(data)

    radio._send_packet = held
    return entered, release


async def test_refresh_waits_for_the_last_write_of_a_burst(gateway: Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.2,))
    await gateway.radio.send_packet(set_owner_write())
    await asyncio.sleep(0.15)
    await gateway.radio.send_packet(set_owner_write())
    await asyncio.sleep(0.1)
    assert len(gateway.radio.config_requests()) == 1  # 0.25 s after the first write, but not after the last
    await wait_until(lambda: len(gateway.radio.config_requests()) == 2 and gateway.cache.ready.is_set())


async def test_web_client_config_request_cancels_a_pending_refresh(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.2,))
    gateway.radio.answer = False  # its first bootstrap frame has not arrived when the refresh would be due
    await gateway.radio.send_packet(set_owner_write())
    await web_view(gateway).put(FakeWebRequest(mesh_pb2.ToRadio(want_config_id=31337)), "entry", "token")
    await asyncio.sleep(0.4)
    assert gateway.radio.config_requests()[1:] == [31337]
    gateway.radio.answer_config(31337)
    await wait_until(gateway.cache.ready.is_set)


async def test_web_client_write_waiting_for_the_radio_drops_a_snapshot_completed_meanwhile(
    gateway: Gateway,
) -> None:
    view = web_view(gateway)
    gateway.radio.answer = False
    await view.put(FakeWebRequest(mesh_pb2.ToRadio(want_config_id=31337)), "entry", "token")
    entered, release = hold_sends(gateway.radio)
    sending = asyncio.create_task(view.put(FakeWebRequest(set_owner_write()), "entry", "token"))
    await entered.wait()
    gateway.radio.answer_config(31337)  # the bootstrap requested before the write completes first
    await wait_until(gateway.cache.ready.is_set)
    release.set()
    await sending
    assert not gateway.cache.ready.is_set()


async def test_stop_cannot_be_followed_by_a_refresh_or_a_late_snapshot(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05,))
    original, entered = gateway.radio._send_packet, asyncio.Event()

    async def slow_disconnect(data: bytes) -> bool:
        if mesh_pb2.ToRadio.FromString(data).HasField("disconnect"):
            entered.set()
            await asyncio.sleep(0.3)  # as a busy BLE write; other sends still go through
        return await original(data)

    gateway.radio._send_packet = slow_disconnect
    await gateway.radio.send_packet(set_owner_write())
    stopping = asyncio.create_task(gateway.interface.stop())
    await entered.wait()
    await stopping
    assert len(gateway.radio.config_requests()) == 1  # Home Assistant's own only
    assert not gateway.cache.ready.is_set()


async def test_web_client_write_finishing_during_stop_does_not_refresh(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05,))
    original, write_release, disconnect_entered = gateway.radio._send_packet, asyncio.Event(), asyncio.Event()

    async def slow(data: bytes) -> bool:
        to_radio = mesh_pb2.ToRadio.FromString(data)
        if to_radio.HasField("packet"):
            await write_release.wait()
        elif to_radio.HasField("disconnect"):
            disconnect_entered.set()
            await asyncio.sleep(0.3)
        return await original(data)

    gateway.radio._send_packet = slow
    sending = asyncio.create_task(web_view(gateway).put(FakeWebRequest(set_owner_write()), "entry", "token"))
    await asyncio.sleep(0.01)
    stopping = asyncio.create_task(gateway.interface.stop())
    await disconnect_entered.wait()
    write_release.set()  # its late invalidation schedules a refresh while stop is still disconnecting
    await sending
    await stopping
    assert len(gateway.radio.config_requests()) == 1
    assert not gateway.cache.ready.is_set()


async def test_write_before_the_config_request_is_passed_on(gateway: Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05,))
    reader, writer, task = gateway.connect()
    send(reader, set_owner_write())
    await wait_until(lambda: gateway.radio.sent[-1] == set_owner_write())
    assert writer.frames() == []  # silent until it asks for config
    send(reader, mesh_pb2.ToRadio(want_config_id=CLIENT_NONCE))
    await wait_until(lambda: completion(CLIENT_NONCE).SerializeToString() in writer.frames())
    assert from_client(gateway, writer, len(expected_for(CLIENT_NONCE))) == []
    reader.feed_eof()
    await task


def reply_to(request_id: int) -> mesh_pb2.FromRadio:
    reply = packet(OWN, request_id + 1, portnums_pb2.PortNum.ADMIN_APP, admin_pb2.AdminMessage().SerializeToString())
    reply.packet.decoded.request_id = request_id
    return reply


async def test_a_reply_to_a_request_sent_before_asking_for_config_reaches_the_client(gateway: Gateway) -> None:
    reader, writer, task = gateway.connect()
    send(reader, admin_write(admin_pb2.AdminMessage(get_owner_request=True), packet_id=700))
    await wait_until(lambda: gateway.radio.sent[-1].packet.id == 700)
    await gateway.mesh(reply_to(700))
    assert reply_to(700) in writer.received()  # live from the start, as before
    send(reader, mesh_pb2.ToRadio(want_config_id=CLIENT_NONCE))
    await wait_until(lambda: completion(CLIENT_NONCE).SerializeToString() in writer.frames())
    reader.feed_eof()
    await task


async def test_the_reply_to_a_write_pipelined_with_a_config_request_follows_the_bootstrap(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.2,))
    reader, writer, task = gateway.connect()
    write = admin_write(admin_pb2.AdminMessage(set_owner=mesh_pb2.User(long_name="new")), packet_id=701)
    reader.feed_data(FRAME(mesh_pb2.ToRadio(want_config_id=123).SerializeToString()) + FRAME(write.SerializeToString()))
    await wait_until(lambda: gateway.radio.sent[-1] == write)
    await gateway.mesh(reply_to(701))  # while the client waits for the refresh
    await wait_until(lambda: completion(123).SerializeToString() in writer.frames())
    await asyncio.sleep(0.05)
    received = writer.received()
    assert received.index(reply_to(701)) > received.index(
        mesh_pb2.FromRadio.FromString(completion(123).SerializeToString())
    )
    reader.feed_eof()
    await task


async def test_a_web_client_download_that_never_completes_is_replaced(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.25,) * 8)
    monkeypatch.setattr(aio_connection, "_REPLAY_DOWNLOAD_TIMEOUT_S", 1.0)
    gateway.radio.cut_short = 1
    await web_view(gateway).put(FakeWebRequest(mesh_pb2.ToRadio(want_config_id=31337)), "entry", "token")
    await asyncio.sleep(0.6)
    assert gateway.radio.config_requests()[1:] == [31337]  # not abandoned for a download of our own too soon
    await wait_until(lambda: len(gateway.radio.config_requests()) == 3 and gateway.cache.ready.is_set())


class HoldingWriter(FakeWriter):
    """A peer that stops reading once it is sent the given frame."""

    def __init__(self, hold: mesh_pb2.FromRadio) -> None:
        super().__init__()
        self.hold, self.held = hold.SerializeToString(), asyncio.Event()

    async def drain(self) -> None:
        if self.data.endswith(FRAME(self.hold)):
            self.held.set()
            await asyncio.Event().wait()
        await super().drain()


def text_ids(writer: FakeWriter) -> list[int]:
    return [r.packet.id for r in writer.received() if r.packet.decoded.portnum == portnums_pb2.PortNum.TEXT_MESSAGE_APP]


async def test_texts_queued_before_asking_for_config_all_follow_the_bootstrap(gateway: Gateway) -> None:
    reader, writer, task = gateway.connect(HoldingWriter(reply_to(703)))
    await gateway.mesh(reply_to(703))
    await writer.held.wait()
    for packet_id in range(400):  # more than the history keeps, all still in this client's queue
        gateway.cache.ingest(packet(5, packet_id), gateway.radio.requested_config_id)
    send(reader, mesh_pb2.ToRadio(want_config_id=1002))
    await wait_until(lambda: len(text_ids(writer)) >= 400)
    await asyncio.sleep(0.05)
    assert text_ids(writer) == list(range(400))
    assert not task.done()
    reader.feed_eof()
    await task


async def test_a_reply_being_written_when_config_is_asked_for_is_sent_once(gateway: Gateway) -> None:
    reader, writer, task = gateway.connect(HoldingWriter(reply_to(702)))
    await gateway.mesh(packet(5, 100))
    await wait_until(lambda: text_ids(writer) == [100])
    gateway.radio.emit(reply_to(702))
    await writer.held.wait()
    gateway.radio.emit(packet(5, 101))
    send(reader, mesh_pb2.ToRadio(want_config_id=1000))
    await wait_until(lambda: completion(1000).SerializeToString() in writer.frames())
    await asyncio.sleep(0.05)
    assert text_ids(writer) == [100, 100, 101]  # sent live before config, then again as history: at least once
    assert writer.received().count(reply_to(702)) == 1
    send(reader, mesh_pb2.ToRadio(want_config_id=1001))
    await wait_until(lambda: completion(1001).SerializeToString() in writer.frames())
    await asyncio.sleep(0.05)
    assert text_ids(writer) == [100, 100, 101]
    assert writer.received().count(reply_to(702)) == 1
    reader.feed_eof()
    await task


async def test_repeated_config_requests_do_not_get_round_the_queue_bound(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "_CLIENT_QUEUE_SIZE", 8)
    _, other_writer, other = await gateway.bootstrapped(1)
    reader, writer, task = gateway.connect(HoldingWriter(bootstrap(2)[0]))  # stops reading at each my_info
    for round_id in range(2):
        send(reader, mesh_pb2.ToRadio(want_config_id=2))
        await wait_until(writer.held.is_set)
        writer.held.clear()
        for item in range(6):  # fits the queue each time, but not twice over
            gateway.cache.ingest(packet(5, round_id * 10 + item, 256), gateway.radio.requested_config_id)
    send(reader, mesh_pb2.ToRadio(want_config_id=2))
    async with asyncio.timeout(2):
        await task
    assert writer.closing
    await gateway.mesh(packet(5, 99))
    assert texts_after_last_completion(other_writer) == [(5, 99)]
    assert not other.done()


async def test_a_completed_web_client_download_does_not_hold_up_the_next_refresh(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = gateway.cache.snapshot.generation
    await web_view(gateway).put(FakeWebRequest(mesh_pb2.ToRadio(want_config_id=31337)), "entry", "token")
    await wait_until(lambda: gateway.cache.snapshot is not None and gateway.cache.snapshot.generation > generation)
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.05,))
    await gateway.radio.send_packet(set_owner_write())
    await wait_until(lambda: len(gateway.radio.config_requests()) == 3 and gateway.cache.ready.is_set())


async def test_a_web_client_download_begun_as_a_write_lands_is_not_kept(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay, "_WRITE_SETTLE_S", 0.15)
    monkeypatch.setattr(aio_connection, "_REPLAY_REFRESH_DELAYS_S", (0.3,))
    view = web_view(gateway)
    await view.put(FakeWebRequest(set_owner_write()), "entry", "token")
    await view.put(FakeWebRequest(mesh_pb2.ToRadio(want_config_id=31337)), "entry", "token")
    await wait_until(lambda: gateway.cache.failure == "the gateway was changed as the download began")
    assert not gateway.cache.ready.is_set()
    # the refresh after it is far enough from the write to be kept
    await wait_until(lambda: len(gateway.radio.config_requests()) == 3 and gateway.cache.ready.is_set())
    assert gateway.interface.is_running
    assert gateway.interface._connected_node_ready.is_set()
