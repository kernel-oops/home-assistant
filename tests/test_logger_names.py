# SPDX-FileCopyrightText: 2026 Marc Warne <marc@kernel-panic.im>
#
# SPDX-License-Identifier: MIT

from custom_components.meshtastic import api, config_flow, meshtastic_tcp, meshtastic_web
from custom_components.meshtastic.meshtastic_tcp import server


def test_module_loggers_are_named_after_their_module() -> None:
    for module in (api, config_flow, meshtastic_tcp, meshtastic_web, server):
        assert module._LOGGER.name == module.__name__
