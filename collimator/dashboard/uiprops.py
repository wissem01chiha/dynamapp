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

from collimator.dashboard.serialization import model_json


BlockNamePortIdPair = tuple[str, int]


def _find_link(
    diagram: model_json.Diagram, src: BlockNamePortIdPair, dst: BlockNamePortIdPair
):
    src_block = _find_node_by_name(diagram, src[0])
    dst_block = _find_node_by_name(diagram, dst[0])
    if not src_block or not dst_block:
        return None

    for link in diagram.links:
        node_match = src_block.uuid == link.src.node and dst_block.uuid == link.dst.node
        port_match = src[1] == link.src.port and dst[1] == link.dst.port
        if node_match and port_match:
            return link


def _find_node_by_name(diagram: model_json.Diagram, name: str):
    for node in diagram.nodes:
        if node.name == name:
            return node


def copy(from_model: model_json.Model, to_model: model_json.Model):
    for link1 in from_model.diagram.links:
        if link1.uiprops is None:
            continue

        src_node = from_model.diagram.find_node(link1.src.node)
        dst_node = from_model.diagram.find_node(link1.dst.node)
        link2 = _find_link(
            to_model.diagram,
            (src_node.name, link1.src.port),
            (dst_node.name, link1.dst.port),
        )
        if link2 is not None:
            link2.uiprops = link1.uiprops

    for node1 in from_model.diagram.nodes:
        if node1.uiprops is None:
            continue
        node2 = _find_node_by_name(to_model.diagram, node1.name)
        if node2 is not None:
            node2.uiprops = node1.uiprops

    for group_uuid1, group_diagram1 in from_model.subdiagrams.diagrams.items():
        group_diagram2 = to_model.subdiagrams.diagrams.get(group_uuid1)
        if group_diagram2 is not None:
            copy(group_diagram1, group_diagram2)
