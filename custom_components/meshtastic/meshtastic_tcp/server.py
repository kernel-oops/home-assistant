# SPDX-FileCopyrightText: 2024-2025 Pascal Brogle @broglep
#
# SPDX-License-Identifier: MIT

import asyncio
from asyncio import StreamReader, StreamWriter
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Self, cast

from google.protobuf import message

from custom_components.meshtastic.aiomeshtastic import MeshInterface
from custom_components.meshtastic.aiomeshtastic.connection import ClientApiConnection
from custom_components.meshtastic.aiomeshtastic.connection.streaming import StreamingClientTransport
from custom_components.meshtastic.aiomeshtastic.protobuf import mesh_pb2
from custom_components.meshtastic.aiomeshtastic.replay import NONCE_ONLY_CONFIG, ReplayCache
from custom_components.meshtastic.const import LOGGER

_LOGGER = LOGGER.getChild(__name__)

# a client that cannot take a frame within this time is not reading and gets disconnected
_WRITE_TIMEOUT_S = 30.0
# live FromRadio messages buffered per client while it is sent its bootstrap and history, and after
_CLIENT_QUEUE_SIZE = 1024
# how long a client asking for config waits for the gateway's own bootstrap to complete
_SNAPSHOT_WAIT_S = 30.0
# wait_closed() can hang on a peer that does not read, or raise on a broken one
_CLOSE_TIMEOUT_S = 5.0


class ClientProxyTransport(StreamingClientTransport):
    def __init__(self, reader: StreamReader, writer: StreamWriter) -> None:
        super().__init__()
        self._reader = reader
        self._writer = writer

    async def read_to_radio_packet(self) -> tuple[bytes, mesh_pb2.ToRadio] | None:
        packet = await self._read_packet_bytes()
        if packet is None:
            return None

        to_radio = mesh_pb2.ToRadio()
        try:
            to_radio.ParseFromString(packet)
        except message.DecodeError:
            self._logger.debug("Error while parsing ToRadio bytes %s", packet, exc_info=True)
        else:
            return packet, to_radio
        return None

    async def write_from_radio_bytes(self, from_radio: bytes) -> None:
        await self._write_bytes(StreamingClientTransport.build_frame(from_radio))

    async def _read_bytes(self, n: int = -1, *, exactly: int | None = None) -> bytes | None:
        if exactly is not None:
            return await self._reader.readexactly(n=exactly)
        return await self._reader.read(n=n)

    async def _write_bytes(self, data: bytes) -> bool:
        self._writer.write(data)
        await asyncio.wait_for(self._writer.drain(), _WRITE_TIMEOUT_S)
        return True

    def _can_read(self) -> bool:
        return not self._reader.at_eof()

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        await self._disconnect()

    async def _connect(self) -> None:
        pass

    async def _disconnect(self) -> None:
        self._writer.close()
        closed = False
        try:
            await asyncio.wait_for(self._writer.wait_closed(), _CLOSE_TIMEOUT_S)
            closed = True
        except Exception:
            self._logger.debug("Graceful close of proxy client failed, aborting", exc_info=True)
        finally:
            if not closed:
                self._writer.transport.abort()

    @property
    def is_connected(self) -> bool:
        return not self._writer.is_closing()


def _completion(nonce: int) -> bytes:
    from_radio = mesh_pb2.FromRadio()
    from_radio.config_complete_id = nonce  # assigned, so a nonce of 0 is still sent as present
    return from_radio.SerializeToString()


# the only ToRadio kinds a client's session answers itself: the bootstrap is the cache's, and disconnect would end
# the gateway's own session. Nothing may be added: clients could do everything else before, and still must
_NOT_FORWARDED = frozenset({"want_config_id", "disconnect"})


class _ProxyClient:
    """One client: silent until it asks for config, then its own bootstrap, recent texts and live traffic."""

    def __init__(
        self,
        cache: ReplayCache,
        connection: ClientProxyTransport,
        peer_name: str,
        forward: Callable[[bytes, mesh_pb2.ToRadio], Awaitable[bool]],
    ) -> None:
        self._cache = cache
        self._forward = forward
        self._connection = connection
        self._peer_name = peer_name
        self.closed = asyncio.Event()
        # (raw FromRadio, history sequence number or 0) for this client; live records stay here across a repeated
        # config request
        self._queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=_CLIENT_QUEUE_SIZE)
        self._stream_task: asyncio.Task | None = None
        # the latest text written to this client, so a repeated config request does not replay texts again
        self._text_seq = 0
        # taken from the queue for a bootstrap and not yet all written; a repeated request takes them over
        self._held: list[tuple[bytes, int]] = []

    async def run(self) -> None:
        # live traffic from the start, as before, so replies to what it sends before asking for config reach it
        self._cache.subscribe(self._on_record)
        self._stream_task = asyncio.create_task(self._stream(None), name=f"tcp_proxy_write_{self._peer_name}")
        task_read = asyncio.create_task(self._read(), name=f"tcp_proxy_read_{self._peer_name}")
        task_closed = asyncio.create_task(self.closed.wait(), name=f"tcp_proxy_disconnect_wait_{self._peer_name}")
        try:
            await asyncio.wait([task_read, task_closed], return_when=asyncio.FIRST_COMPLETED)
        finally:
            self._cache.unsubscribe(self._on_record)
            tasks = [task for task in (task_read, task_closed, self._stream_task) if task is not None]
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
        _LOGGER.debug(
            "Closed proxy connection %s: %s", self._peer_name, [r for r in results if isinstance(r, Exception)]
        )

    async def _read(self) -> None:
        while True:
            packet = await self._connection.read_to_radio_packet()
            if packet is None:
                continue
            packet_bytes, to_radio = packet
            _LOGGER.debug(
                "Proxy client %s sent %s",
                self._peer_name,
                ClientApiConnection._protobuf_log(to_radio),  # noqa: SLF001
            )
            variant = to_radio.WhichOneof("payload_variant")
            if variant not in _NOT_FORWARDED:
                # passed on as received, as the proxy always has: replies reach every client, not just this one
                await self._forward(packet_bytes, to_radio)
            elif variant == "want_config_id":
                await self._restart_stream(to_radio.want_config_id)
            else:
                # disconnect, whatever its value, ends only the client's own session, never the gateway's
                return

    async def _restart_stream(self, nonce: int) -> None:
        if self._stream_task is not None:
            # a request replaces only this client's own stream
            # the client stays subscribed meanwhile, so nothing live is missed; texts queued now that the new
            # bootstrap's history also holds are skipped when their turn comes
            self._stream_task.cancel()
            await asyncio.wait([self._stream_task])
        self._stream_task = asyncio.create_task(self._stream(nonce), name=f"tcp_proxy_write_{self._peer_name}")

    def _on_record(self, from_radio: bytes | None, seq: int) -> None:
        if from_radio is None:
            _LOGGER.info("Gateway session ended, disconnecting proxy client %s", self._peer_name)
            self.closed.set()
            return
        try:
            self._queue.put_nowait((from_radio, seq))
        except asyncio.QueueFull:
            if not self.closed.is_set():
                _LOGGER.warning("Proxy client %s is not keeping up, disconnecting it", self._peer_name)
                self.closed.set()

    async def _stream(self, nonce: int | None) -> None:
        """Write this client's bootstrap for nonce, if it asked for one, then its live traffic."""
        try:
            if nonce is not None and not await self._bootstrap(nonce):
                return
            while True:
                from_radio, seq = await self._queue.get()
                # texts before the first config request are not counted, so its history still has them
                if seq and nonce is not None:
                    if seq <= self._text_seq:
                        continue
                    self._text_seq = seq
                await self._connection.write_from_radio_bytes(from_radio)
        except TimeoutError:
            _LOGGER.warning(
                "Proxy client %s did not take a frame within %s s, disconnecting it", self._peer_name, _WRITE_TIMEOUT_S
            )
            self.closed.set()
        except Exception:
            _LOGGER.debug("Writing to proxy client %s failed", self._peer_name, exc_info=True)
            self.closed.set()

    async def _bootstrap(self, nonce: int) -> bool:
        try:
            async with asyncio.timeout(_SNAPSHOT_WAIT_S):
                while (bound := self._cache.bind(self._on_record, self._text_seq)) is None:
                    await self._cache.ready.wait()
        except TimeoutError:
            # never ask the radio instead: that would restart Home Assistant's own session
            _LOGGER.warning(
                "No complete gateway bootstrap within %s s, refusing proxy client %s", _SNAPSHOT_WAIT_S, self._peer_name
            )
            self.closed.set()
            return False

        # what was already queued for this client arrived before binding: its texts are merged with the history,
        # which may no longer hold the oldest of them, and the rest follows; later arrivals stay queued
        # texts are retired as they are written, the rest below, so _held is only what is still owed
        self._held += [self._queue.get_nowait() for _ in range(self._queue.qsize())]
        self._held = [(from_radio, seq) for from_radio, seq in self._held if not seq or seq > self._text_seq]
        if len(self._held) > _CLIENT_QUEUE_SIZE:  # repeated requests must not get round the queue's bound
            _LOGGER.warning("Proxy client %s is not keeping up, disconnecting it", self._peer_name)
            self.closed.set()
            return False
        texts = dict(bound.history)
        texts.update((seq, from_radio) for from_radio, seq in self._held if seq)

        write = self._connection.write_from_radio_bytes
        for from_radio in bound.snapshot.records_for(nonce):
            await write(from_radio)
        await write(_completion(nonce))
        if nonce != NONCE_ONLY_CONFIG:  # as firmware, which sends node state after the completion except here
            for from_radio in bound.node_state:
                await write(from_radio)
        # counted before writing: a frame is in the transport once write() has to wait for the peer
        for seq in sorted(texts):
            self._text_seq = seq
            await write(texts[seq])
        self._held = [(from_radio, seq) for from_radio, seq in self._held if not seq]
        while self._held:
            await write(self._held[0][0])
            del self._held[0]
        return True


class MeshtasticTcpProxy:
    def __init__(self, interface: MeshInterface, host: str | None = None, port: int | None = None) -> None:
        self._handlers: set[asyncio.Task] = set()
        self._interface = interface
        self._should_stop = False
        self._stopping = False
        self._host = host
        self._port = port or 4403
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        self._stopping = False
        if not self._interface.is_running:
            self._should_stop = True
            await self._interface.start()

        self._server = await asyncio.start_server(self._handle_client, self._host, self._port)

    async def stop(self) -> None:
        # a handler that first runs after the snapshot below would otherwise never be cancelled
        self._stopping = True
        if self._server:
            self._server.close()

        tasks = list(self._handlers)
        for task in tasks:
            task.cancel()
        if tasks:
            # each handler's close is itself bounded, so this only guards against the unexpected
            _, pending = await asyncio.wait(tasks, timeout=2 * _CLOSE_TIMEOUT_S)
            if pending:
                _LOGGER.warning("%d tcp proxy tasks did not stop in time", len(pending))

        if self._should_stop:
            await self._interface.stop()
            self._should_stop = False

    async def _handle_client(self, reader: StreamReader, writer: StreamWriter) -> None:
        client_connection = ClientProxyTransport(reader, writer)
        if self._stopping:
            await client_connection.disconnect()
            return

        handler = asyncio.current_task()
        self._handlers.add(cast("asyncio.Task", handler))  # a handler always runs inside a task
        try:
            peer_name = "{}:{}".format(*writer.transport.get_extra_info("peername", (None, None))[0:2])
            await _ProxyClient(
                self._interface.replay_cache,
                client_connection,
                peer_name,
                self._interface._connection.forward,  # noqa: SLF001
            ).run()
        except Exception:
            _LOGGER.warning("Failed handling proxy connection", exc_info=True)
        finally:
            try:
                await client_connection.disconnect()
            finally:
                self._handlers.discard(handler)
