# SPDX-FileCopyrightText: 2026 Marc Warne <marc@kernel-panic.im>
#
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock

from custom_components.meshtastic import _setup_meshtastic_device


async def test_model_id_is_a_string_for_an_unnamed_hardware_model() -> None:
    # MessageToDict leaves hwModel as an int when the enum value is newer than the bundled protobuf
    registry = MagicMock()
    registry.async_get_device.return_value = None
    node = {"num": 1, "user": {"id": "!00000001", "longName": "Node", "hwModel": 999}}

    await _setup_meshtastic_device(MagicMock(), {}, registry, MagicMock(), node, node, 1)

    assert registry.async_get_or_create.call_args.kwargs["model_id"] == "999"
