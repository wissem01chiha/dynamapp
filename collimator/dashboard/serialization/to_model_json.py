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

"""This file contains conversion logic from wildcat to collimator model.json.
Collimator model.json to wildcat conversion is done in model_interface.py."""

import logging
from uuid import uuid4

from collimator.dashboard.serialization.model_json import (
    Configuration,
    Diagram,
    Link,
    Model,
    Node,
    Parameter,
    ParameterDefinition,
    Port,
    Reference,
    Subdiagrams,
)
from collimator.framework import (
    Diagram as WildcatDiagram,
    SystemBase,
    SystemCallback,
)
from collimator.library import (
    Sindy,
    CustomJaxBlock,
    CustomPythonBlock,
    Demultiplexer,
    DiscreteClock,
    PIDDiscrete,
    PyTorch,
    TensorFlow,
    IOPort,
    LTISystem,
    Multiplexer,
    MJX,
    PID,
    ReduceBlock,
    ReferenceSubdiagram,
    Logarithm,
    Sine,
    TransferFunction,
    TransferFunctionDiscrete,
)

logger = logging.getLogger(__name__)

_CL_LIBRARY = {
    "Abs",
    "Adder",
    "Arithmetic",
    "BatteryCell",
    "Chirp",
    "Clock",
    "Comparator",
    "Constant",
    "CoordinateRotation",
    "CoordinateRotationConversion",
    "CosineWave",
    "CrossProduct",
    "DataSource",
    "DeadZone",
    "Demux",
    "Derivative",
    "DerivativeDiscrete",
    "DiscreteInitializer",
    "DotProduct",
    "EdgeDetection",
    "Exponent",
    "FilterDiscrete",
    "Gain",
    "IfThenElse",
    "Inport",
    "Integrator",
    "IntegratorDiscrete",
    "KalmanFilter",
    "InfiniteHorizonKalmanFilter",
    "UnscentedKalmanFilter",
    "ExtendedKalmanFilter",
    "LogicalOperator",
    "LogicalReduce",
    "LookupTable1d",
    "LookupTable2d",
    "Log",
    "MatrixConcatenation",
    "MatrixInversion",
    "MatrixMultiplication",
    "MatrixTransposition",
    "MinMax",
    "MJX",
    "MuJoCo",
    "MLP",
    "ModelicaFMU",
    "Mux",
    "Offset",
    "Outport",
    "PID",
    "PID_Discrete",
    "Power",
    "PyTorch",
    "Product",
    "ProductOfElements",
    "Pulse",
    "PythonScript",
    "PyTwin",
    "Quantizer",
    "Relay",
    "RandomNumber",
    "Ramp",
    "RateLimiter",
    "Reciprocal",
    "RigidBody",
    "Saturate",
    "Sawtooth",
    "ScalarBroadcast",
    "SignalDatatypeConversion",
    "SineWave",
    "SINDy",
    "Slice",
    "SquareRoot",
    "Stack",
    "StateSpace",
    "Step",
    "Stop",
    "SumOfElements",
    "TensorFlow",
    "TransferFunction",
    "TransferFunctionDiscrete",
    "Trigonometric",
    "UnitDelay",
    "VideoSink",
    "VideoSource",
    "WhiteNoise",
    "ZeroOrderHold",
}

# Map of wildcat block param names to collimator block param names.
_PARAM_NAME_MAP = {
    "Integrator": {
        "initial_state": "initial_states",
    },
    "IntegratorDiscrete": {
        "initial_state": "initial_states",
    },
    "PIDDiscrete": {
        "kp": "Kp",
        "ki": "Ki",
        "kd": "Kd",
    },
    "LTISystem": {
        "initialize_states": "initial_states",
    },
}


class IOPortExportError(Exception):
    pass


def _param_name(block, param_name):
    block_cls = block.__class__.__name__
    if block_cls in _PARAM_NAME_MAP:
        if param_name in _PARAM_NAME_MAP[block_cls]:
            return _PARAM_NAME_MAP[block_cls][param_name]
    return param_name


def _wc_to_cl_model_parameters(block: WildcatDiagram) -> dict[str, Parameter]:
    """Converts top-level model parameters to API format"""
    # Only top-level models are supported here. Blocks that contain subdiagrams
    # (that is: core.Group) don't have parameters. Submodels serialize to
    # reference submodels and have ParameterDefinition lists.
    # NOTE: Hopefully one day we can migrate model parameters to align with
    # submodels ParameterDefinitions.
    assert isinstance(block, WildcatDiagram)

    all_parameters = {
        param_name: block.parameters[param_name]
        for param_name in (block.instance_parameters or [])
    }

    params: dict[str, Parameter] = {}
    for name, param in all_parameters.items():
        # Top-level model parameters must be fully defined by their expression
        # and can not be marked as 'is_string'.
        expr, _ = param.value_as_api_param(
            allow_param_name=False, allow_string_literal=False
        )
        params[name] = Parameter(value=expr, is_string=False)

    return params


def _wc_to_cl_parameters(block: SystemBase) -> dict[str, Parameter]:
    """Parse parameters of a wildcat block to a Collimator-compatible format"""

    if isinstance(block, WildcatDiagram):
        return _wc_to_cl_model_parameters(block)

    # Following blocks are instances of LTISystem so we want to ignore A, B, C, D
    # TFDiscrete is not an instance of LTISystem, but needs the same treatment as TF.
    if isinstance(block, PID):
        kp = str(block.dynamic_parameters["kp"])
        ki = str(block.dynamic_parameters["ki"])
        kd = str(block.dynamic_parameters["kd"])
        n = str(block.dynamic_parameters["n"])
        initial_state = str(block.dynamic_parameters["initial_state"])
        return {
            "Kp": Parameter(value=kp),
            "Ki": Parameter(value=ki),
            "Kd": Parameter(value=kd),
            "N": Parameter(value=n),
            "initial_state": Parameter(value=initial_state),
        }
    elif isinstance(block, TransferFunction) or isinstance(
        block, TransferFunctionDiscrete
    ):
        num = str(block.parameters["num"])
        den = str(block.parameters["den"])
        return {
            "numerator_coefficients": Parameter(value=num),
            "denominator_coefficients": Parameter(value=den),
        }

    all_parameters = {**block.dynamic_parameters, **block.static_parameters}

    params = {}
    for k, p in all_parameters.items():
        pname = _param_name(block, k)
        is_string = isinstance(p.value, str) and not p.is_python_expr
        params[pname] = Parameter(value=str(p), is_string=is_string)

    return params


def _wc_to_cl_ports(
    ports: list[SystemCallback], kind, params: dict = None
) -> list[dict]:
    """Parse input ports of a Wildcat block to a Collimator-compatible format"""
    if params is None:
        params = {}
    return [
        {
            "name": port.name,
            "kind": kind,
            "parameters": params[port.name] if port.name in params else {},
        }
        for port in ports
    ]


def _wc_to_cl_iports(node: SystemBase) -> list[dict]:
    """Parse input ports of a Wildcat block to a Collimator-compatible format"""
    dyn_blocks = (ReduceBlock, PyTorch, TensorFlow)
    kind = "dynamic" if isinstance(node, dyn_blocks) else "static"
    return _wc_to_cl_ports(node.input_ports, kind)


def _wc_to_cl_oports(node: SystemBase) -> list[dict]:
    """Parse input ports of a Wildcat block to a Collimator-compatible format"""
    dyn_blocks = (Demultiplexer, CustomJaxBlock, PyTorch, TensorFlow)
    kind = "dynamic" if isinstance(node, dyn_blocks) else "static"
    return _wc_to_cl_ports(node.output_ports, kind)


def _get_block_type(node: SystemBase) -> str:
    if isinstance(node, Logarithm):
        return "core.Log"
    if isinstance(node, Sine):
        return "core.SineWave"
    elif isinstance(node, Multiplexer):
        return "core.Mux"
    elif isinstance(node, Demultiplexer):
        return "core.Demux"
    elif isinstance(node, PIDDiscrete):
        return "core.PID_Discrete"
    elif isinstance(node, DiscreteClock):
        return "core.Clock"
    elif type(node) is LTISystem:
        return "core.StateSpace"
    elif isinstance(node, WildcatDiagram):
        if node.ref_id is not None:
            return "core.ReferenceSubmodel"
        else:
            return "core.Group"
    elif isinstance(node, Sindy):
        return "core.SINDy"
    elif isinstance(node, CustomPythonBlock):
        return "core.PythonScript"
    elif isinstance(node, CustomJaxBlock):
        return "core.PythonScript"
    elif isinstance(node, MJX):
        return "core.MuJoCo"
    elif node.__class__.__name__ in _CL_LIBRARY:
        return f"core.{node.__class__.__name__}"

    return None


def _get_ref_submodel_uuid(node: WildcatDiagram) -> str:
    if not isinstance(node, WildcatDiagram):
        return None
    return node.ref_id


def _wc_to_cl_block(node: SystemBase) -> Node:
    block_type = _get_block_type(node)

    if block_type is None:
        raise NotImplementedError(
            f"Block type {node.__class__.__name__} is not supported"
        )

    parameters = _wc_to_cl_parameters(node)
    time_mode = time_mode = (
        parameters.pop("time_mode").value if "time_mode" in parameters else None
    )

    return Node(
        name=node.name,
        type=_get_block_type(node),
        inputs=_wc_to_cl_iports(node),
        outputs=_wc_to_cl_oports(node),
        parameters=parameters,
        uuid=str(uuid4()),
        submodel_reference_uuid=_get_ref_submodel_uuid(node),
        time_mode=time_mode,
    )


def _wc_to_cl_links(
    diagram: WildcatDiagram, nodes: dict[SystemBase, Node]
) -> list[dict[Link]]:
    links = []
    for iport, oport in diagram.connection_map.items():
        input_sys, input_idx = iport
        output_sys, output_idx = oport
        logger.debug(
            "Connecting %s:%s to %s:%s",
            output_sys.name,
            output_idx,
            input_sys.name,
            input_idx,
        )
        links.append(
            Link(
                uuid=str(uuid4()),
                src=Port(node=nodes[output_sys].uuid, port=output_idx),
                dst=Port(node=nodes[input_sys].uuid, port=input_idx),
            )
        )
    return links


def _make_inport(name: str) -> Node:
    return Node(
        name=name,
        type="core.Inport",
        outputs=[
            {
                "name": "out_0",
                "kind": "static",
                "parameters": {},
            }
        ],
        parameters={
            "description": Parameter(is_string=True, value=""),
            "port_id": Parameter(value="0"),
        },
        uuid=str(uuid4()),
    )


def _make_outport(name: str) -> Node:
    return Node(
        name=name,
        type="core.Outport",
        inputs=[
            {
                "name": "in_0",
                "kind": "static",
                "parameters": {},
            }
        ],
        parameters={
            "description": Parameter(is_string=True, value=""),
            "port_id": Parameter(value="0"),
        },
        uuid=str(uuid4()),
    )


def _make_inport_and_link(name: str, node_uuid: str, idx: int) -> tuple[Node, Link]:
    inport = _make_inport(name)
    link = Link(
        uuid=str(uuid4()),
        src=Port(node=inport.uuid, port=0),
        dst=Port(node=node_uuid, port=idx),
    )
    return inport, link


def _make_outport_and_link(name: str, node_uuid: str, idx: int) -> tuple[Node, Link]:
    outport = _make_outport(name)
    link = Link(
        uuid=str(uuid4()),
        src=Port(node=node_uuid, port=idx),
        dst=Port(node=outport.uuid, port=0),
    )
    return outport, link


def _check_ioports_export_error(diagram: WildcatDiagram) -> bool:
    for node in diagram:
        if isinstance(node, IOPort):
            loc = (node, 0)
            is_input_port = loc in diagram.exported_input_ports
            is_output_port = loc in diagram.exported_output_ports

            if not is_input_port and not is_output_port:
                raise IOPortExportError(
                    f"IOPort {node.name} is not exported as input or output"
                )
            elif is_input_port and is_output_port:
                raise IOPortExportError(
                    f"IOPort {node.name} is exported as both input and output"
                )


RefId = str


def _wc_to_cl_diagram(diagram: WildcatDiagram) -> tuple[Model, dict[RefId, Model]]:
    _check_ioports_export_error(diagram)
    groups = {node for node in diagram if isinstance(node, WildcatDiagram)}

    # TODO: check that the diagrams corresponding to the same submodel references
    # are the same

    # Convert all non IOPort nodes
    nodes = {
        node: _wc_to_cl_block(node) for node in diagram if not isinstance(node, IOPort)
    }

    ioports_links = []
    ioports = []
    # create Inports and Outports for exported ports
    for node, idx in diagram.exported_input_ports:
        if isinstance(node, IOPort):
            nodes[node] = _make_inport(node.name)
        else:
            inport, link = _make_inport_and_link(
                f"{node.name}_inport_{idx}", nodes[node].uuid, idx
            )
            ioports.append(inport)
            ioports_links.append(link)

    for node, idx in diagram.exported_output_ports:
        if isinstance(node, IOPort):
            nodes[node] = _make_outport(node.name)
        else:
            outport, link = _make_outport_and_link(
                f"{node.name}_outport_{idx}", nodes[node].uuid, idx
            )
            ioports.append(outport)
            ioports_links.append(link)

    # process reference submodels
    ref_subdiagrams = {}
    for node in diagram:
        if (
            not isinstance(node, WildcatDiagram)
            or node.ref_id is None
            or node.ref_id in ref_subdiagrams
        ):
            continue
        ref_subdiagram, ref_ref_subdiagrams = _wc_to_cl_diagram(node)
        ref_subdiagrams.update(ref_ref_subdiagrams)
        ref_subdiagrams[node.ref_id] = ref_subdiagram

    subdiagrams = {}
    references = {}
    for group in groups:
        subdiagram, ref_ref_subdiagrams = _wc_to_cl_diagram(group)
        ref_subdiagrams.update(ref_ref_subdiagrams)
        references[nodes[group].uuid] = Reference(diagram_uuid=subdiagram.diagram.uuid)
        subdiagrams[subdiagram.diagram.uuid] = subdiagram.diagram
        subdiagrams.update(subdiagram.subdiagrams.diagrams)
        references.update(subdiagram.subdiagrams.references)

    root_diagram = Diagram(
        uuid=str(uuid4()),
        links=_wc_to_cl_links(diagram, nodes) + ioports_links,
        nodes=list(nodes.values()) + ioports,
        annotations=None,
    )
    parameter_definitions = None
    if diagram.ref_id is not None:
        params = ReferenceSubdiagram.get_parameter_definitions(diagram.ref_id)
        parameter_definitions = [
            ParameterDefinition(name=param.name, default_value=str(param.value))
            for param in params
        ]

    parameters = _wc_to_cl_model_parameters(diagram)

    return (
        Model(
            uuid=diagram.ui_id,
            name=diagram.name,
            diagram=root_diagram,
            subdiagrams=Subdiagrams(
                diagrams=subdiagrams,
                references=references,
            ),
            state_machines={},  # FIXME.
            parameters=parameters,
            parameter_definitions=parameter_definitions,
        ),
        ref_subdiagrams,
    )


def convert(
    wc_diagram: WildcatDiagram,
    configuration: Configuration = None,
) -> tuple[Model, dict[str, Model]]:
    """Convert a Wildcat diagram to a Collimator model.json."""
    model, reference_submodels = _wc_to_cl_diagram(wc_diagram)
    if configuration is not None:
        model.configuration = configuration
    return model, reference_submodels
