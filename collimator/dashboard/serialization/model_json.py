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

"""
This module contains types that map directly to Collimator's JSON model format.
"""

import dataclasses
from typing import Optional
from uuid import uuid4

from dataclasses_json import DataClassJsonMixin, dataclass_json

from collimator.simulation import ResultsOptions, SimulatorOptions


# The use of uuid.UUID causes issues with json [de]serialization.
UUID = str


@dataclass_json
@dataclasses.dataclass
class Port:
    node: UUID
    port: int
    port_side: Optional[str] = None


@dataclass_json
@dataclasses.dataclass
class Link:
    uuid: Optional[UUID] = None
    src: Optional[Port] = None
    dst: Optional[Port] = None
    uiprops: Optional[dict] = None


@dataclass_json
@dataclasses.dataclass
class Parameter:
    value: str
    is_string: Optional[bool] = False


@dataclass_json
@dataclasses.dataclass
class IOPortVariant:
    variant_kind: Optional[str] = None  # acausal
    acausal_domain: Optional[str] = None  # electrical, rotational, translational


@dataclass_json
@dataclasses.dataclass
class IOPort:
    name: str
    record: Optional[bool] = None  # record_mode='all' ignores this
    parameters: Optional[dict[str, Parameter]] = dataclasses.field(default_factory=dict)

    # Maybe be enhanced at load time (WIP)
    # index: int = None

    # UI props
    kind: Optional[str] = "static"  # static or dynamic
    variant: Optional[IOPortVariant] = dataclasses.field(default_factory=dict)


@dataclass_json
@dataclasses.dataclass
class Node:
    name: str
    type: str
    uuid: UUID
    inputs: list[IOPort] = dataclasses.field(default_factory=list)
    outputs: list[IOPort] = dataclasses.field(default_factory=list)
    parameters: dict[str, Parameter] = dataclasses.field(default_factory=dict)
    submodel_reference_uuid: Optional[UUID] = None
    state_machine_diagram_id: Optional[UUID] = None
    time_mode: Optional[str] = "agnostic"

    # UI props
    uiprops: Optional[dict] = None


@dataclass_json
@dataclasses.dataclass
class Diagram:
    links: list[Link]
    nodes: list[Node]
    uuid: UUID = dataclasses.field(default_factory=lambda: str(uuid4()))
    annotations: Optional[list[dict]] = None

    def find_node(self, uuid: str):
        for node in self.nodes:
            if node.uuid == uuid:
                return node


@dataclass_json
@dataclasses.dataclass
class Reference:
    diagram_uuid: UUID


@dataclass_json
@dataclasses.dataclass
class ParameterDefinition:
    name: str
    default_value: str
    uuid: Optional[UUID] = dataclasses.field(default_factory=lambda: str(uuid4()))
    # display_name: Optional[str] = None
    # description: Optional[str] = None


@dataclass_json
@dataclasses.dataclass
class WorkspaceDataFile:
    file_name: str


@dataclass_json
@dataclasses.dataclass
class Workspace:
    init_scripts: Optional[list[WorkspaceDataFile]] = dataclasses.field(
        default_factory=list
    )


@dataclass_json
@dataclasses.dataclass
class SolverConfig:
    method: Optional[str] = "non-stiff"
    max_step: Optional[float] = 1e6
    min_step: Optional[float] = 0
    relative_tolerance: Optional[float] = 1e-3
    absolute_tolerance: Optional[float] = 1e-6


@dataclass_json
@dataclasses.dataclass
class Configuration:
    stop_time: Optional[float] = 10.0
    start_time: Optional[float] = 0.0
    sample_time: Optional[float] = 0.1
    numerical_backend: Optional[str] = "auto"  # one of "auto", "numpy", "jax"
    solver: Optional[SolverConfig] = dataclasses.field(default_factory=SolverConfig)
    max_major_steps: Optional[int] = None
    sim_output_mode: Optional[str] = "auto"
    max_results_interval: Optional[float] = None
    fixed_results_interval: Optional[float] = None
    record_mode: Optional[str] = "all"  # "all" or "selected"
    workspace: Optional[Workspace] = dataclasses.field(default_factory=Workspace)

    @staticmethod
    def from_wildcat_config(
        sim_options: SimulatorOptions,
        results_options: ResultsOptions,
        stop_time: float = 10.0,
        sample_time: float = 0.1,
        workspace: Workspace = None,
    ):
        return Configuration(
            record_mode="all",
            solver=SolverConfig(
                method=sim_options.ode_solver_method,
                absolute_tolerance=sim_options.atol,
                relative_tolerance=sim_options.rtol,
                min_step=sim_options.min_minor_step_size,
                max_step=sim_options.max_minor_step_size,
            ),
            sample_time=sample_time,
            stop_time=stop_time,
            max_major_steps=sim_options.max_major_steps,
            sim_output_mode=results_options.mode.name,
            max_results_interval=results_options.max_results_interval,
            fixed_results_interval=results_options.fixed_results_interval,
            numerical_backend=sim_options.math_backend,
            workspace=workspace or Workspace(),
        )


@dataclass_json
@dataclasses.dataclass
class Subdiagrams:
    diagrams: dict[UUID, Diagram]
    references: dict[UUID, Reference]

    def get_diagram(self, group_block_uuid: UUID) -> Diagram:
        diagram_uuid = self.references[group_block_uuid].diagram_uuid
        return self.diagrams[diagram_uuid]


@dataclass_json
@dataclasses.dataclass
class StateMachineState:
    name: str = None
    uuid: str = None
    exit_priority_list: list[str] = None
    uiprops: Optional[dict] = None


@dataclass_json
@dataclasses.dataclass
class StateMachineTransition:
    uuid: str = None
    guard: str = None
    actions: list[str] = None
    destNodeId: str = None  # pylint: disable=invalid-name
    sourceNodeId: str = None  # pylint: disable=invalid-name
    uiprops: Optional[dict] = None


@dataclass_json
@dataclasses.dataclass
class StateMachineEntryPoint:
    actions: list[str] = None
    dest_id: str = None

    # UI props
    dest_coord: int = None
    dest_side: str = None


@dataclass_json
@dataclasses.dataclass
class StateMachine:
    uuid: str = None
    links: list[StateMachineTransition] = None
    nodes: list[StateMachineState] = None
    entry_point: StateMachineEntryPoint = None


@dataclasses.dataclass
class Model(DataClassJsonMixin):  # explicit inheritance makes pylint happy
    # intended to load model.json and submodel-uuid.ver.json
    diagram: Diagram
    subdiagrams: Subdiagrams = dataclasses.field(
        default_factory=lambda: Subdiagrams({}, {})
    )
    state_machines: Optional[dict[UUID, StateMachine]] = dataclasses.field(
        default_factory=dict
    )
    parameters: Optional[dict[str, Parameter]] = dataclasses.field(default_factory=dict)
    parameter_definitions: Optional[list[ParameterDefinition]] = dataclasses.field(
        default_factory=list
    )
    configuration: Optional[Configuration] = dataclasses.field(
        default_factory=Configuration
    )

    kind: Optional[str] = "Model"
    uuid: Optional[str] = dataclasses.field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = dataclasses.field(default_factory=lambda: "root")
    version: Optional[int] = 1  # used by models
    edit_id: Optional[str] = None  # used by ref submodels

    # TODO:
    # submodel_configuration

    # FIXME API still uses "submodels" instead of "subdiagrams"
    @classmethod
    def from_dict(cls, kvs, *, infer_missing=False, legacy_subdiagrams_name=True):
        if legacy_subdiagrams_name:
            if "submodels" in kvs and "subdiagrams" not in kvs:
                kvs["subdiagrams"] = kvs.pop("submodels")
        return super().from_dict(kvs, infer_missing=infer_missing)

    def to_dict(self, encode_json=False, legacy_subdiagrams_name=True):
        d = super().to_dict(encode_json=encode_json)
        if legacy_subdiagrams_name:
            if "subdiagrams" in d and "submodels" not in d:
                d["submodels"] = d.pop("subdiagrams")
        return d
