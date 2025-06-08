# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

"""Pretty-printing utilities"""

BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
LIGHTGREY = "\033[37m"
RESET = "\033[0m"
RIGHT_ARROW = "\u2192"


def pprint_fancy(prefix: str, system, with_links=True) -> str:
    """Helper to pretty-print a system with colored output."""
    s = f"{prefix}\u2502\u2500\u2500"
    s += f" {BOLD}{GREEN}{system.name}{RESET}"
    s += f" <{BOLD}{system.__class__.__name__}{RESET}>"

    links = []
    if with_links and system.parent:
        for dest, source in system.parent.connection_map.items():
            if source[0] is not system:
                continue
            src_blk_name = system.output_ports[source[1]].name
            dst_blk_name = dest[0].name
            dst_port_name = dest[0].input_ports[dest[1]].name
            link = (
                f"{CYAN}{src_blk_name}{RESET} {RIGHT_ARROW} "
                f"{GREEN}{dst_blk_name}.{BLUE}{dst_port_name}{RESET}"
            )
            links.append(link)

    if not links:
        return f"{s}\n"

    return f"{s} [{', '.join(links)}]\n"
