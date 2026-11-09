# SPDX-FileCopyrightText: 2026 Marc Warne <marc@kernel-panic.im>
#
# SPDX-License-Identifier: MIT

"""A gateway's records as its firmware sends them, for the replay cache and TCP proxy tests."""

from custom_components.meshtastic.aiomeshtastic.protobuf import (
    admin_pb2,
    channel_pb2,
    config_pb2,
    device_ui_pb2,
    mesh_pb2,
    module_config_pb2,
    portnums_pb2,
    telemetry_pb2,
    xmodem_pb2,
)

OWN = 0x0AC9F6D4
FromRadio = mesh_pb2.FromRadio
# a field number no schema uses, so this record carries an unknown field end to end
UNKNOWN_FIELD = bytes([0x98, 0x06, 0x01])


def node(num: int) -> FromRadio:
    user = mesh_pb2.User(id=f"!{num:08x}", long_name=f"node {num:08x}", short_name=f"{num:08x}"[-4:])
    return FromRadio(node_info=mesh_pb2.NodeInfo(num=num, user=user, last_heard=1_762_700_000))


def channel(index: int, name: str) -> FromRadio:
    role = channel_pb2.Channel.Role.PRIMARY if index == 0 else channel_pb2.Channel.Role.DISABLED
    settings = channel_pb2.ChannelSettings(name=name if index == 0 else "")
    return FromRadio(channel=channel_pb2.Channel(index=index, role=role, settings=settings))


def every_variant(message: type, **values: object) -> list:
    """One message per payload variant in field order, as firmware sends config and module config."""
    fields = message.DESCRIPTOR.oneofs_by_name["payload_variant"].fields
    return [message(**{f.name: values.get(f.name, f.message_type._concrete_class())}) for f in fields]


def bootstrap(nonce: int, *, nodes: int = 250, channel_name: str = "LongFast") -> list[FromRadio]:
    """Build a full bootstrap in firmware order (PhoneAPI::getFromRadio), its completion last."""
    Config, ModuleConfig = config_pb2.Config, module_config_pb2.ModuleConfig  # noqa: N806
    unknown = FromRadio.FromString(node(OWN + 1).SerializeToString() + UNKNOWN_FIELD)
    configs = every_variant(
        Config,
        device=Config.DeviceConfig(tzdef="GMT0BST,M3.5.0/1,M10.5.0"),
        lora=Config.LoRaConfig(region=Config.LoRaConfig.RegionCode.EU_868),
        security=Config.SecurityConfig(public_key=b"k" * 32),
    )
    module_configs = every_variant(ModuleConfig, telemetry=ModuleConfig.TelemetryConfig(device_update_interval=900))
    return [
        FromRadio(my_info=mesh_pb2.MyNodeInfo(my_node_num=OWN, nodedb_count=nodes)),
        FromRadio(deviceuiConfig=device_ui_pb2.DeviceUIConfig(screen_brightness=5)),
        node(OWN),
        FromRadio(metadata=mesh_pb2.DeviceMetadata(firmware_version="2.8.2")),
        FromRadio(region_presets=mesh_pb2.LoRaRegionPresetMap()),  # firmware after 2.8.2 sends this here
        *(channel(index, channel_name) for index in range(8)),
        *(FromRadio(config=c) for c in configs),
        *(FromRadio(moduleConfig=m) for m in module_configs),
        unknown,
        *(node(OWN + 1 + i) for i in range(1, nodes - 1)),
        FromRadio(fileInfo=mesh_pb2.FileInfo(file_name="/prefs/config.proto", size_bytes=120)),
        FromRadio(fileInfo=mesh_pb2.FileInfo(file_name="/prefs/nodes.proto", size_bytes=9000)),
        completion(nonce),
    ]


def completion(nonce: int) -> FromRadio:
    from_radio = FromRadio()
    from_radio.config_complete_id = nonce
    return from_radio


def expected_for(nonce: int) -> list[bytes]:
    """List what firmware sends a client asking with this nonce, worked out independently of the code under test."""
    records = bootstrap(nonce)[:-1]
    if nonce == 69421:
        records = [r for r in records if r.HasField("node_info")]
    elif nonce == 69420:
        records = [r for r in records if not r.HasField("node_info") or r.node_info.num == OWN]
    return [r.SerializeToString() for r in records] + [completion(nonce).SerializeToString()]


def packet(
    sender: int, packet_id: int, port: int = portnums_pb2.PortNum.TEXT_MESSAGE_APP, payload: bytes | None = None
) -> FromRadio:
    mesh_packet = mesh_pb2.MeshPacket(
        to=0xFFFFFFFF,
        id=packet_id,
        rx_time=1_762_700_000 + packet_id,
        rx_snr=6.25,
        rx_rssi=-90,
        hop_start=3,
        hop_limit=2,
        decoded=mesh_pb2.Data(portnum=port, payload=payload or f"hello {packet_id} from {sender:x}".encode()),
    )
    setattr(mesh_packet, "from", sender)
    return FromRadio(packet=mesh_packet)


def not_history() -> list[FromRadio]:
    """List records that may pass live but must never be replayed as history."""
    port = portnums_pb2.PortNum
    return [
        packet(
            5, 100, port.ROUTING_APP, mesh_pb2.Routing(error_reason=mesh_pb2.Routing.Error.NONE).SerializeToString()
        ),
        packet(5, 101, port.ADMIN_APP, admin_pb2.AdminMessage(get_owner_response=mesh_pb2.User()).SerializeToString()),
        packet(5, 102, port.POSITION_APP, mesh_pb2.Position(latitude_i=541_000_000).SerializeToString()),
        packet(5, 103, port.TELEMETRY_APP, telemetry_pb2.Telemetry(time=1).SerializeToString()),
        packet(OWN, 104),  # our own text, looped back
        FromRadio(packet=mesh_pb2.MeshPacket(id=105, encrypted=b"\x01\x02")),
        FromRadio(queueStatus=mesh_pb2.QueueStatus(free=8, maxlen=16)),
        FromRadio(mqttClientProxyMessage=mesh_pb2.MqttClientProxyMessage(topic="msh/x", data=b"y")),
        FromRadio(xmodemPacket=xmodem_pb2.XModem(seq=1)),
    ]
