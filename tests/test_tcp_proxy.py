# SPDX-FileCopyrightText: 2026 Marc Warne <marc@kernel-panic.im>
#
# SPDX-License-Identifier: MIT

"""Offline tests for the TCP proxy: fake upstream interface, fake client streams, no sockets."""

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest

from custom_components.meshtastic.aiomeshtastic.connection.streaming import StreamingClientTransport
from custom_components.meshtastic.aiomeshtastic.protobuf import mesh_pb2
from custom_components.meshtastic.meshtastic_tcp import server

FRAME = StreamingClientTransport.build_frame


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[mesh_pb2.ToRadio] = []

    async def _send_packet(self, packet: bytes) -> bool:
        to_radio = mesh_pb2.ToRadio()
        to_radio.ParseFromString(packet)
        self.sent.append(to_radio)
        return True


class FakeInterface:
    is_running = True

    def __init__(self) -> None:
        self._connection = FakeConnection()
        self.packets: asyncio.Queue[mesh_pb2.FromRadio] = asyncio.Queue()

    async def from_radio_stream(self) -> AsyncIterator[mesh_pb2.FromRadio]:
        while True:
            yield await self.packets.get()


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
        self, *, drain_hangs: bool = False, close_error: Exception | None = None, close_hangs: bool = False
    ) -> None:
        FakeWriter._port += 1
        self.port = FakeWriter._port
        self.data = bytearray()
        self.closing = False
        self.aborted = False
        self.drain_hangs = drain_hangs
        self.close_error = close_error
        self.close_hangs = close_hangs
        self.transport = FakeTransport(self)

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        if self.drain_hangs:
            await asyncio.Event().wait()

    def is_closing(self) -> bool:
        return self.closing

    def close(self) -> None:
        self.closing = True

    async def wait_closed(self) -> None:
        if self.close_error is not None:
            raise self.close_error
        if self.close_hangs:
            await asyncio.Event().wait()

    def received(self) -> list[mesh_pb2.FromRadio]:
        packets, data = [], bytes(self.data)
        while data:
            assert data[:2] == bytes([StreamingClientTransport.START1, StreamingClientTransport.START2])
            length = (data[2] << 8) + data[3]
            from_radio = mesh_pb2.FromRadio()
            from_radio.ParseFromString(data[4 : 4 + length])
            packets.append(from_radio)
            data = data[4 + length :]
        return packets


async def wait_until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not condition():  # noqa: ASYNC110 - the fakes expose state, not events
            await asyncio.sleep(0.01)


def client(writer: FakeWriter | None = None) -> tuple[asyncio.StreamReader, FakeWriter]:
    return asyncio.StreamReader(), writer or FakeWriter()


def send(reader: asyncio.StreamReader, to_radio: mesh_pb2.ToRadio) -> None:
    reader.feed_data(FRAME(to_radio.SerializeToString()))


@pytest.fixture
async def proxy(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[server.MeshtasticTcpProxy]:
    async def fake_start_server(*_args: object, **_kwargs: object) -> FakeServer:
        return FakeServer()

    monkeypatch.setattr(server.asyncio, "start_server", fake_start_server)
    instance = server.MeshtasticTcpProxy(FakeInterface())
    await instance.start()
    yield instance
    await instance.stop()


@pytest.mark.parametrize("value", [True, False])
async def test_disconnect_is_never_forwarded(proxy: server.MeshtasticTcpProxy, value: bool) -> None:  # noqa: FBT001
    reader, writer = client()
    other_reader, other_writer = client()
    handler = asyncio.create_task(proxy._handle_client(reader, writer))
    other = asyncio.create_task(proxy._handle_client(other_reader, other_writer))

    to_radio = mesh_pb2.ToRadio(disconnect=value)
    assert to_radio.HasField("disconnect")
    send(reader, to_radio)
    async with asyncio.timeout(2):
        await handler

    assert writer.closing
    assert proxy._interface._connection.sent == []
    assert not other.done()
    assert not other_writer.closing
    other_reader.feed_eof()
    await other
