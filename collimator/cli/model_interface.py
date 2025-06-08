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

from __future__ import annotations

import glob
import json
import os
import traceback
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

from collimator.logging import logger
from collimator.dashboard.serialization import (
    model_json,
    from_model_json,
    ui_types,
)
from collimator.dashboard.serialization.time_mode import (
    time_mode_node,
    time_mode_port,
    time_mode_node_with_ports,
)
from ..experimental import AcausalSystem
from ..framework import Diagram, IntegerTime, SystemBase, ErrorCollector
from ..simulation import SimulatorOptions, simulate

if TYPE_CHECKING:
    from ..framework import ContextBase
    from ..backend.typing import Array

__all__ = [
    "load_model",
    "load_model_from_dir",
]


def load_model_from_dir(
    modeldir: str,
    model: str = "model.json",
    block_overrides: dict[str, SystemBase] = None,
    parameter_overrides: dict[str, model_json.Parameter] = None,
) -> from_model_json.SimulationContext:
    # register reference submodels
    file_pattern = os.path.join(modeldir, "submodel-*-latest.json")
    submodel_files = glob.glob(file_pattern)
    for submodel_file in submodel_files:
        ref_id = os.path.basename(submodel_file).split("-")[1:-1]
        ref_id = "-".join(ref_id)
        with open(submodel_file, "r", encoding="utf-8") as f:
            submodel = model_json.Model.from_json(f.read())
            from_model_json.register_reference_submodel(ref_id, submodel)

    with open(os.path.join(modeldir, model), "r", encoding="utf-8") as f:
        model_dict = json.load(f)

    return from_model_json.load_model(
        model_dict,
        block_overrides=block_overrides,
        model_parameter_overrides=parameter_overrides,
    )


def load_model(
    modeldir: str = ".",
    model: str = "model.json",
    logsdir: str = None,
    npydir: str = None,
    block_overrides=None,
    parameter_overrides: dict[str, model_json.Parameter] = None,
) -> AppInterface:
    model = load_model_from_dir(
        modeldir,
        model=model,
        block_overrides=block_overrides,
        parameter_overrides=parameter_overrides,
    )

    return AppInterface(
        model,
        logsdir=logsdir,
        npydir=npydir,
    )


def get_acausal_signal_types(
    namepath: list[str],
    uuidpath: list[str],
    node: AcausalSystem,
    context: ContextBase,
):
    signal_types: list[ui_types.Node] = []

    # FIXME these two are quite the hack
    outports_maps = node.outports_maps
    acausal_network: from_model_json.AcausalNetwork = node.acausal_network

    phleaf_outport_to_block_port: dict[int, tuple[str, int]] = {}
    for block_id, outport_map in outports_maps.items():
        for block_port_index, phleaf_outport_index in outport_map.items():
            phleaf_outport_to_block_port[phleaf_outport_index] = (
                block_id,
                block_port_index,
            )

    phleaf_outport_values: dict[int, Any] = {}
    phleaf_outport_tms: dict[int, ui_types.TimeMode] = {}
    for phleaf_outport_idx, phleaf_outport in enumerate(node.output_ports):
        val = phleaf_outport.eval(context)
        phleaf_outport_values[phleaf_outport_idx] = val
        phleaf_outport_tms[phleaf_outport_idx] = time_mode_port(phleaf_outport, None)

    all_ports_tms = []
    for block_id, phleaf_outport_map in outports_maps.items():
        block_spec = next(
            (blk for blk in acausal_network.blocks.values() if blk.uuid == block_id),
            None,
        )

        if block_spec is None:
            logger.error("Block with id %s not found in acausal diagram.", block_id)
            continue

        blk_namepath = namepath + [block_spec.name]
        blk_uuidpath = uuidpath + [block_spec.uuid]
        blk_outports = []
        blk_port_tms = []

        for block_port_index, phleaf_outport_index in phleaf_outport_map.items():
            outport_spec: model_json.IOPort = block_spec.outputs[block_port_index]
            val = phleaf_outport_values[phleaf_outport_index]

            port_tm = phleaf_outport_tms[phleaf_outport_index]
            all_ports_tms.append(port_tm)
            blk_port_tms.append(port_tm)

            blk_outport = ui_types.Port(
                index=block_port_index,
                dtype=str(np.array(val).dtype),
                dimension=np.shape(val),
                time_mode=port_tm,
                discrete_interval=None,
                name=outport_spec.name,
            )
            blk_outports.append(blk_outport)

        # @am. assign time_mode.ACAUSAL
        # blk_tm = time_mode_node_with_ports(blk_port_tms)
        nd = ui_types.Node(
            namepath=blk_namepath,
            uuidpath=blk_uuidpath,
            outports=blk_outports,
            time_mode=ui_types.TimeMode.ACAUSAL,
            discrete_interval=None,
        )
        signal_types.append(nd)

    tm = time_mode_node_with_ports(all_ports_tms)
    return signal_types, tm


def get_signal_types(
    namepath: list[str],
    uuidpath: list[str],
    signal_type_nodes: list[ui_types.Node],
    nodes: list[SystemBase],
    context: ContextBase,
    dep_graph,
):
    # iterate over the nested diagram and:
    #   1] determine each signal's dtype and dimension
    #   2] determine time_mode of nodes and signals
    #   3] generate the signal_types.json given to the frontend

    nodes_tm = []
    for node in nodes:
        if isinstance(node, AcausalSystem):
            try:
                acausal_signal_types, acausal_tm = get_acausal_signal_types(
                    namepath,
                    uuidpath,
                    node,
                    context,
                )
                nodes_tm.append(acausal_tm)
                signal_type_nodes.extend(acausal_signal_types)
            except Exception as exc:
                # Not raising an exception because this is very experimental
                logger.error(
                    "Failed to extract signal type information for acausal blocks in "
                    "'%s' due to exception: %s. Signals may not be visualized and the "
                    "coloring of the diagram may be incorrect.",
                    ".".join(namepath) or "root",
                    exc,
                )
            continue

        node_cls_tm = time_mode_node(node)
        blk_namepath = namepath + [node.name]
        blk_uuidpath = uuidpath + [node.ui_id]
        ports = []

        if isinstance(node, Diagram):
            signal_type_nodes, subdiagram_tm = get_signal_types(
                blk_namepath,
                blk_uuidpath,
                signal_type_nodes,
                node.nodes,
                context,
                dep_graph,
            )

        ports_tm = []
        for port_idx, out_port in enumerate(node.output_ports):
            val = out_port.eval(context)

            tm = time_mode_port(out_port, node_cls_tm)
            ports_tm.append(tm)

            # this data is returned to the UI
            port = ui_types.Port(
                index=port_idx,
                dtype=str(np.array(val).dtype),
                dimension=np.shape(val),
                time_mode=tm,
                discrete_interval=None,
                name=node.output_ports[port_idx].name,
            )
            ports.append(port.__dict__)

        # node time mode
        if isinstance(node, Diagram):
            node_tm = subdiagram_tm
        else:
            node_tm = time_mode_node_with_ports(ports_tm)

        nodes_tm.append(node_tm)

        # this data is returned to the UI
        nd = ui_types.Node(
            namepath=blk_namepath,
            uuidpath=blk_uuidpath,
            outports=ports,
            time_mode=node_tm,
            discrete_interval=None,
        )
        signal_type_nodes.append(nd.__dict__)

    # diagram time mode
    diagram_tm = time_mode_node_with_ports(nodes_tm)

    return signal_type_nodes, diagram_tm


class AppInterface:
    def __init__(
        self,
        sim_context: from_model_json.SimulationContext,
        logsdir: str = None,
        npydir: str = None,
    ):
        self.context: ContextBase = None
        self.sim_context = sim_context
        self.logsdir = logsdir
        self.npydir = npydir

        # track whether "check" method has been run on this system
        self.static_analysis_complete = False

        # called here to maintain behavior expected by some tests,
        # i.e. some data created in statatic analysis is made available
        # in the object after only calling __init__.
        self.check()

    def check(self):
        # execute the all static analysis operations, raising errors/warnings
        # as appropriate.

        # TODO: ensure no top level inports/outports

        # initialized context and verify internal consistency
        try:
            error_collector = ErrorCollector()
            with error_collector:
                self.context = self.sim_context.diagram.create_context(
                    check_types=True,
                    error_collector=error_collector,
                )
                logger.debug("Context created")
        except Exception as exc:
            # try / catch here only to provide a breakpoint site for inspection
            # from within the debugger.
            # user model related errors found during context creation/type checking
            # are all collected in error_collector.
            # wildcat internal errors should be raised, not collected.
            raise exc

        # Write 'signal_types.json'
        if self.logsdir is not None:
            try:
                os.makedirs(self.logsdir, exist_ok=True)
                context = self.context
                signal_type_nodes, _root_time_modes = get_signal_types(
                    namepath=[],
                    uuidpath=[],
                    signal_type_nodes=[],
                    nodes=self.sim_context.diagram.nodes,
                    context=context,
                    dep_graph=self.sim_context.diagram._dependency_graph,
                )

                # signal types json to be returned to UI
                signal_types_file = os.path.join(self.logsdir, "signal_types.json")
                signal_types = ui_types.SignalTypes(nodes=signal_type_nodes)
                signal_types_dict = signal_types.to_api(omit_none=True)
                with open(signal_types_file, "w", encoding="utf-8") as outfile:
                    json.dump(signal_types_dict, outfile, indent=2, sort_keys=False)

            except Exception as exc:
                traceback.print_exc()
                warnings.warn(
                    f"Failed to generate signal_types.json due to exception: {exc}."
                )

        if error_collector.errors:
            # log all errors
            logger.debug("Type Errors collected during context creation:")
            for error in error_collector.errors:
                logger.debug(error)
            # for now, we just raise/return the first type error we found.
            raise error_collector.errors[0]

        self.static_analysis_complete = True

    @property
    def diagram(self):
        return self.sim_context.diagram

    @property
    def simulator_options(self):
        return self.sim_context.simulator_options

    @property
    def results_options(self):
        return self.sim_context.results_options

    def simulate(
        self,
        start_time: float = None,
        stop_time: float = None,
        simulator_options: SimulatorOptions = None,
    ) -> dict[str, Array]:
        if not self.static_analysis_complete:
            self.check()

        if start_time is None:
            start_time = 0.0
        if stop_time is None:
            stop_time = 10.0

        start_time = float(start_time)
        stop_time = float(stop_time)

        options = self.sim_context.simulator_options
        if simulator_options is not None:
            options = simulator_options

        # Usually the default integer time scale will work (up to ~0.3 years), but if
        # a longer simulation was requested, we need to use a larger integer time scale.
        # Note that this is also configurable via SimulatorOptions, but this is a more
        # robust automatic solution (though not amenable to JAX tracing).
        while stop_time > IntegerTime.max_float_time:
            IntegerTime.set_scale(1000 * IntegerTime.time_scale)
            logger.info(
                "Increasing integer time scale by a factor of 1000x to allow for "
                "representation of the specified end time."
            )

        results = simulate(
            self.sim_context.diagram,
            self.context,
            (start_time, stop_time),
            options=options,
            results_options=self.sim_context.results_options,
            recorded_signals=options.recorded_signals,
        )

        # Calls to model.simulate are expecting a dict of outputs including time.
        results.outputs["time"] = results.time

        if self.npydir is not None:
            for name, val in results.outputs.items():
                if val is None:
                    logger.error(f"Output '{name}' is None, not writing npy file")
                    continue
                with open(os.path.join(self.npydir, f"{name}.npy"), "wb") as f:
                    np.lib.format.write_array(f, val, allow_pickle=False)
        else:
            logger.warning("npydir is None, not writing npy files")

        return results.outputs
