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

import dataclasses
import json
import math
import traceback
from typing import Any
from uuid import uuid4

import jax.numpy as jnp
import numpy as np

import collimator
from collimator import backend
from collimator.logging import logger
from collimator.experimental import AcausalCompiler, AcausalDiagram, EqnEnv
from collimator.framework import (
    Diagram,
    DiagramBuilder,
    LeafSystem,
    Parameter,
    ParameterCache,
    SystemBase,
    SystemCallback,
)
from collimator.framework.error import (
    BlockInitializationError,
    BlockParameterError,
    CollimatorError,
    LegacyBlockConfigurationError,
    ModelInitializationError,
    StaticError,
)
from collimator.framework.parameter import resolve_parameters
from collimator.lazy_loader import LazyLoader
from collimator.library import ReferenceSubdiagram
from collimator.simulation import ResultsMode, ResultsOptions, SimulatorOptions

from . import block_interface, model_json

control = LazyLoader("control", globals(), "control")


"""
Layer between the Collimator app JSON output and the sim engine.
Presently, this works for model.json+list[submodel-uuid-ver.json] content
from both simworkerpy(wildcat) and simworker(go)(cmlc).

Processing steps:
1] The classes in types.py are used to "read" the json, packaging convinently
Not ideal, but there is some transformation that happen here.
Specifically, model.json.parameters dict is transformed to match List[ParameterDefinition]
so that the top level diagram and all submodel diagrams can be treated identically
w.r.t. parameter namespace handling at later loading stages.

2] visit all the model subdiagrams by depth first search through the json.
create node name_path ids, and a dict diagrams[subdiagram_node_id]=LoadCtxtDiagram (e.g. most the raw json content).

3] visit all the elements of diagrams dict, and build a wildcat diagram object for each, composing them as we go.

"""


@dataclasses.dataclass
class SimulationContext:
    model_uuid: str
    diagram: Diagram
    results_options: ResultsOptions
    recorded_signals: dict[str, SystemCallback]
    simulator_options: SimulatorOptions = None
    start_time: float = 0.0
    stop_time: float = 10.0
    model_parameters: dict[str, Parameter] = None
    init_script: str = None


@dataclasses.dataclass
class AcausalNetwork:
    blocks: dict[str, model_json.Node]
    acausal_links: dict[str, model_json.Link]
    causal_inlinks: dict[str, model_json.Link]
    causal_outlinks: dict[str, model_json.Link]


def loads_model(
    model_json_str: str,
    namespace: dict[str, Any] = None,
    block_overrides: dict[str, SystemBase] = None,
    model_parameter_overrides: dict[str, model_json.Parameter] = None,
) -> SimulationContext:
    """Load a model from a JSON string.

    Reference submodels must be registered before calling this function.
    """
    model_dict = json.loads(model_json_str)
    return load_model(
        model_dict,
        namespace=namespace,
        block_overrides=block_overrides,
        model_parameter_overrides=model_parameter_overrides,
    )


def load_model(
    model_json_obj: dict,
    namespace: dict[str, Any] = None,
    block_overrides: dict[str, SystemBase] = None,
    model_parameter_overrides: dict[str, model_json.Parameter] = None,
) -> SimulationContext:
    """Load a model from a parsed JSON object.

    Reference submodels must be registered before calling this function.
    """
    if namespace is None:
        namespace = {}

    model = model_json.Model.from_dict(model_json_obj)

    model_parameters_json = model.parameters
    if model_parameter_overrides:
        model_parameters_json.update(model_parameter_overrides)

    # evaluate model params
    model_parameters_py = eval_parameters(
        default_parameters=model.parameter_definitions,
        instance_parameters=model_parameters_json,
        name_path=[],
        ui_id_path=[],
    )

    # NOTE: model parameters are available in init scripts and output of init scripts
    # are accessible in block parameters (not model parameters).
    init_script_namespace = {}
    init_script = None
    if model.configuration.workspace:
        # if it's not an empty dict, it should have ref to initscript
        # @am. this is not great but works for now.
        if model.configuration.workspace.init_scripts:
            init_script = model.configuration.workspace.init_scripts[0].file_name
            init_script_namespace = eval_init_script(
                init_script,
                model_parameters_py,
                name_path=[],
                ui_id_path=[],
            )
            # makes it easier to debug when printed and
            # we don't need those in the namespace
            init_script_namespace.pop("__builtins__")

    model_parameters = {}
    for param_name, param in model_parameters_py.items():
        if (
            param_name in model_parameters_json
            and model_parameters_json[param_name].is_string
        ):
            model_parameters[param_name] = model_parameters_json[param_name].value
        else:
            model_parameters[param_name] = Parameter(name=param_name, value=param)

    root_namespace = namespace
    root_namespace.update(model_parameters)
    root_namespace.update(init_script_namespace)

    root_id = "root"

    # MUST be set before diagram creation
    if model.configuration.numerical_backend == "auto":
        model.configuration.numerical_backend = backend.active_backend
    collimator.set_backend(model.configuration.numerical_backend)

    # traverse the entire model, and extract the blocks/links/etc.
    # of each acausal network.
    # NOTE: when isolating AcausalNetworks, we need to identify them
    # with name_path that starts with 'root' because otherwise the
    # name_path for a AcausalNetwork at the root level would have empty
    # string as id.
    acausal_networks = identify_acausal_networks(
        name=root_id,
        diagram=model.diagram,
        subdiagrams=model.subdiagrams,
        parent_ui_id_path=[root_id],
        parent_path=[root_id],
        ui_id=root_id,
    )

    recorded_signals = {}
    diagram = make_subdiagram(
        root_id,
        model.diagram,
        model.subdiagrams,
        model.state_machines,
        acausal_networks,
        parent_path=[],
        parent_ui_id_path=[],
        ui_id=model.diagram.uuid,
        namespace_params=root_namespace,
        block_overrides=block_overrides,
        global_discrete_interval=model.configuration.sample_time,
        record_mode=model.configuration.record_mode,
        recorded_signals=recorded_signals,
        start_time=float(model.configuration.start_time),
        model_parameters=model_parameters,
    )

    results_options = None
    if model.configuration:
        results_options, simulator_options = simulation_settings(
            model.configuration, recorded_signals=recorded_signals
        )

    return SimulationContext(
        model_uuid=model.uuid,
        diagram=diagram,
        results_options=results_options,
        recorded_signals=recorded_signals,
        simulator_options=simulator_options,
        start_time=model.configuration.start_time,
        stop_time=model.configuration.stop_time,
        model_parameters=model_parameters,
        init_script=init_script,
    )


def eval_parameter(value: str, _globals: dict, _locals: dict):
    # Block parameters can be left empty, but JSON may contain null or "None"
    if not value or value == "None":
        return None

    # FIXME: These are passed correctly when called from `eval_parameters`, but
    # not when called directly from `load_model`, so when pulling a model from the
    # dashboard anything with `np` or `jnp` will fail.  Is there a better way to
    # ensure these are always available in both cases?
    _locals = {
        **_locals,
        "np": np,
        "jnp": jnp,
        "math": math,
        "false": False,
        "true": True,
    }

    # Resolve parameters used in the expression
    params_in_global, _ = resolve_parameters(value, _globals)
    params_in_local, _ = resolve_parameters(value, _locals)

    parameters = params_in_global | params_in_local

    # Value is a python expression that depends on parameters.
    # We should wrap it in a Parameter object to track dependencies.
    if len(parameters) > 0:
        p = Parameter(
            value=value,
            is_python_expr=True,
            py_namespace={**_globals, **_locals},
        )
        for param in parameters:
            ParameterCache.add_dependent(param, p)
        return p

    p = eval(value, _globals, _locals)
    # Rules for user-input parameters:
    # - if explicitly given as an array with dtype, use that dtype
    # - if boolean, use as given
    # - otherwise, convert to a numpy array
    if not hasattr(p, "dtype") and not isinstance(p, bool):
        p = np.array(p)
        # if the array has integer dtype convert to float. Note that
        # this case will still not be reached if the parameter is explicitly
        # declared as an array with integer datatype.  However, this will
        # promote inputs like `"0"` or `"[1, 0]"` to floats.
        if issubclass(p.dtype.type, np.integer):
            p = p.astype(float)
    return p


def eval_parameters(
    default_parameters: list[model_json.ParameterDefinition] = None,
    instance_parameters: dict[str, model_json.Parameter] = None,
    call_site_namespace: dict[str, Any] = None,
    name_path: list[str] = None,
    ui_id_path: list[str] = None,
):
    # parameter handling.
    # at this point we have the following:
    # 1] diagrams[block_id].dynamic_parameters
    #   these are from the submodel-uuid-ver.json[parameter_definitions].
    #   these are the definitions and defaults for all parameters in the name space of the submodel
    # 2] instance_parameters
    #   these are the parameter values from the instance of the submodel. this must be a subset of 1]
    #   this dict is typically limited to those params which have been modified from their default value.
    #   the values of these should have been evaluated in the context of the parent of the instance.
    # 3] call_site_namespace
    #   these are the parameters in the name space of the parent.
    #   these are only used here because before getting here, only the instance parameters
    #   have been evaluated in the parent name space. if a parameters has not been modified,
    #   then we use the default value from 1] but we need to evaluate it in the parent name space.
    #
    # Note: reference submodels have a 'protected' parameter names space. parents can only pass
    # parameter values through the submodels defined parameters. i.e. there is no shared name space
    # between the reference submodel instance and its parent.

    if default_parameters is None:
        default_parameters = []
    if call_site_namespace is None:
        call_site_namespace = {}
    if instance_parameters is None:
        instance_parameters = {}

    # We don't technically need to pass np and jnp here, but it makes them explicitly
    # available in the local `eval` environment.
    _locals = {
        **call_site_namespace,
        "np": np,
        "jnp": jnp,
        "math": math,
        "false": False,
        "true": True,
    }

    def _eval(param_name, value: str):
        try:
            return eval_parameter(value, globals(), _locals)
        # We will probably want to be much more discriminating with respect to what gets
        # caught here. This uses exception chaining to record the offending block id
        # and parameter name.
        except Exception as exc:
            traceback.print_exc()
            raise BlockParameterError(
                name_path=name_path, ui_id_path=ui_id_path, parameter_name=param_name
            ) from exc

    # multi round eval is required in case we have parameters that reference other parameters.
    # FIXME: WC-112 this should not be allowed
    def _multi_round_eval(params, eval_fn):
        eval_results = {}
        max_eval_depth = 10
        for i in range(max_eval_depth):
            need_eval = False
            for pname, p in params.items():
                try:
                    eval_results[pname] = eval_fn(pname, p)
                except BlockParameterError as exc:
                    if i == max_eval_depth - 1:
                        raise exc
                    need_eval = True
            _locals.update(eval_results)
            if not need_eval:
                break
        return eval_results

    def _eval_param_def(pname, p):
        if p.default_value != "":
            return _eval(pname, p.default_value)
        return None

    def _eval_param(pname, p):
        if p.is_string:
            return p.value
        return _eval(pname, p.value)

    default_values = _multi_round_eval(
        {p.name: p for p in default_parameters},
        _eval_param_def,
    )
    instance_values = _multi_round_eval(
        instance_parameters,
        _eval_param,
    )

    default_values.update(instance_values)
    return default_values


def eval_init_script(
    init_script_file_name: str,
    namespace_params: dict = None,
    name_path: list[str] = None,
    ui_id_path: list[str] = None,
) -> dict:
    # FIXME: don't do this:
    # We don't technically need to pass np and jnp here, but it makes them explicitly
    # available in the local `exec` environment. a user could also do these imports.
    # "__main__": {} is necessary for script of the form
    #   imports ...
    #   a = 1
    #   def f(b):
    #       return a+b
    #   out_0 = f(2)
    # withotu getting "a not defined" error.

    namespace_params = namespace_params or {}

    _locals = {
        **namespace_params,
        "np": np,
        "jnp": jnp,
        "control": control,
        "__main__": {},
    }

    with open(init_script_file_name, "r") as file:
        _init_script_code = file.read()

    try:
        exec(_init_script_code, _locals, _locals)
    except Exception as e:
        raise ModelInitializationError(
            f"Failed to execute init_script {init_script_file_name}",
            name_path=name_path,
            ui_id_path=ui_id_path,
        ) from e

    # _locals will be pretty messy after exec() since it wil contain all the stuff
    # python puts in globals for any execution. doesn't matter, this globals env
    # is only retained for parameter evaluation.
    return _locals


def GroupBlock(
    block_spec: model_json.Node,
    global_discrete_interval: float,
    subdiagrams: dict[str, model_json.Diagram],
    state_machines: dict[str, model_json.LoadStateMachine],
    acausal_networks: dict[str, AcausalNetwork],
    parent_path: list[str],
    parent_ui_id_path: list[str],
    record_mode: str = "selected",
    namespace_params: dict[str, Any] = None,
    block_overrides=None,
    recorded_signals=None,
    start_time: float = 0.0,
) -> Diagram:
    return make_subdiagram(
        block_spec.name,
        subdiagrams.get_diagram(block_spec.uuid),
        subdiagrams,
        state_machines,
        acausal_networks,
        ui_id=block_spec.uuid,
        parent_path=parent_path,
        parent_ui_id_path=parent_ui_id_path,
        global_discrete_interval=global_discrete_interval,
        namespace_params=namespace_params or {},
        block_overrides=block_overrides,
        record_mode=record_mode,
        recorded_signals=recorded_signals,
        start_time=start_time,
    )


def register_reference_submodel(ref_id: str, model: model_json.Model):
    def _make_subdiagram_instance(
        instance_name: str,
        parameters,
        uuid: str = None,
        parent_path: list[str] = None,
        parent_ui_id_path: list[str] = None,
        record_mode: str = "selected",
        recorded_signals: dict[str, SystemCallback] = None,
        global_discrete_interval: float = 0.1,
        start_time: float = 0.0,
    ):
        if uuid is None:
            uuid = str(uuid4())
        diagram = dataclasses.replace(model.diagram, uuid=uuid)

        acausal_parent_ui_id_path = (
            ["root"] if parent_ui_id_path is None else ["root"] + parent_ui_id_path
        )
        acausal_networks = identify_acausal_networks(
            name=instance_name,
            diagram=diagram,
            subdiagrams=model.subdiagrams,
            parent_path=parent_path,
            parent_ui_id_path=acausal_parent_ui_id_path,
            ui_id=uuid,
            acausal_networks=None,
        )

        return make_subdiagram(
            instance_name,
            diagram,
            model.subdiagrams,
            model.state_machines,
            acausal_networks=acausal_networks,
            namespace_params=parameters,
            ui_id=uuid,
            parent_path=parent_path,
            parent_ui_id_path=parent_ui_id_path,
            global_discrete_interval=global_discrete_interval,
            record_mode=record_mode,
            recorded_signals=recorded_signals,
            start_time=start_time,
        )

    parameter_definitions = None
    if model.parameter_definitions:
        parameter_definitions = model.parameter_definitions
    elif model.parameters:
        # NOTE: simworker-go instantiates the submodel with parameters (dict of parameters)
        # see test_0080a.py for example.
        # So here we convert to parameter definitions (list of ParameterDefinition).
        logger.debug("Converting instantiated parameters to parameter definitions.")
        parameter_definitions = [
            model_json.ParameterDefinition(name=k, default_value=v.value)
            for k, v in model.parameters.items()
        ]

    parameters = eval_parameters(default_parameters=parameter_definitions)
    parameters = [
        Parameter(name=name, value=value) for name, value in parameters.items()
    ]

    ReferenceSubdiagram.register(
        _make_subdiagram_instance,
        parameter_definitions=parameters,
        ref_id=ref_id,
    )


# FIXME: simplify this function. it's too long.
# pylint: disable=too-complex
# flake8: noqa: C901
def make_subdiagram(
    name: str,
    diagram: model_json.Diagram,
    subdiagrams: dict[str, model_json.Diagram],
    state_machines: dict[str, model_json.LoadStateMachine],
    acausal_networks: dict[str, AcausalNetwork],
    parent_ui_id_path: list[str] = None,
    parent_path: list[str] = None,
    ui_id: str = None,
    namespace_params=None,
    block_overrides: dict[str, SystemBase] = None,
    record_mode: str = "selected",
    recorded_signals: dict[str, SystemCallback] = None,
    global_discrete_interval: float = 0.1,
    start_time: float = 0.0,
    model_parameters: dict[str, Parameter] = None,
) -> Diagram:
    if namespace_params is None:
        namespace_params = {}

    if parent_path is None:
        parent_path = []

    if parent_ui_id_path is None:
        parent_ui_id_path = []

    # The "node_spec" passed here is the "node" specification that doesn't
    # contain the actual blocks. the info about the block tat constains the
    # subdiagram, e.g. submodel instance.

    # TODO: correctly handle RefSubmodelConfiguration features
    # (atomic, discrete_step)

    builder = DiagramBuilder()

    # I/O ports are considered "blocks" in the UI, so they need to be tracked
    # specificially (lists of block names)
    exported_inputs: list[str] = []
    exported_outputs: list[str] = []

    # needed for dereferencing node ids in link specs. this map is local to a
    # canvas.
    block_uuid_to_name: dict[str, str] = {}

    # block name to created object
    blocks: dict[str, SystemBase | Diagram] = {}

    # check if this canvas has an acausal diagram.
    if acausal_networks is not None:
        # NOTE: when isolating AcausalDiagrams, we need to identify then
        # with name_path that starts with 'root' because otherwise the
        # name_path for a AcausalNetwork at the root level would have empty
        # string as id.
        acausal_network_key = ".".join(["root"] + parent_ui_id_path)
        acausal_network = acausal_networks.get(acausal_network_key, None)
    else:
        acausal_network = None
    # if this canavas has an acausal diagram, produce the equivalent acausal_system
    if acausal_network is not None:
        acausal_system = build_acausal_system(
            acausal_network,
            namespace_params,
            name=".".join(["root"] + parent_path + ["acausal_system"]),
            parent_path=parent_path,
            parent_ui_id_path=parent_ui_id_path,
        )
        blocks[acausal_system.name] = acausal_system
        builder.add(acausal_system)
        acausal_links = acausal_network.acausal_links
        causal_inlinks = acausal_network.causal_inlinks
        causal_outlinks = acausal_network.causal_outlinks
    else:
        acausal_links = {}
        causal_inlinks = {}
        causal_outlinks = {}

    # process causal blocks
    for block_spec in diagram.nodes:
        block: SystemBase | Diagram = None

        # block names are used as locally (in this canvas) unique identifiers
        block_name = block_spec.name
        block_ui_id = block_spec.uuid
        block_uuid_to_name[block_ui_id] = block_name

        # these are used for rich errors before the block is created
        block_name_path = parent_path + [block_name]
        block_ui_id_path = parent_ui_id_path + [block_ui_id]

        if block_spec.type == "core.Inport":
            exported_inputs.append(block_name)
        elif block_spec.type == "core.Outport":
            exported_outputs.append(block_name)

        is_phealf_block = False

        try:
            # FIXME: refactor below contents of try into a function

            if block_overrides and block_name in block_overrides:
                # FIXME this was probably broken inside subdiagrams and looks extremely hacky anyway
                block_name_path_str = ".".join(block_name_path)
                block = block_overrides[block_name_path_str]

            # branches for recursing into subdiagrams
            elif block_spec.type == "core.ReferenceSubmodel":
                logger.debug(
                    "Creating reference submodel %s (ref id: %s) "
                    "with instance_parameters: %s, call_site_namespace: %s",
                    block_name,
                    block_spec.submodel_reference_uuid,
                    block_spec.parameters,
                    namespace_params,
                )

                # Note: only expressions are supported here, no string literals

                parameters = eval_parameters(
                    instance_parameters=block_spec.parameters,
                    call_site_namespace=namespace_params,
                    name_path=block_name_path,
                    ui_id_path=block_ui_id_path,
                )

                block = ReferenceSubdiagram.create_diagram(
                    block_spec.submodel_reference_uuid,
                    instance_parameters=parameters,
                    global_discrete_interval=global_discrete_interval,
                    record_mode=record_mode,
                    instance_name=block_name,
                    recorded_signals=recorded_signals,
                    parent_path=parent_path + [block_name],
                    parent_ui_id_path=parent_ui_id_path + [block_ui_id],
                    uuid=block_spec.uuid,
                    start_time=start_time,
                )
            elif block_spec.type in ("core.Group", "core.Submodel"):
                block = GroupBlock(
                    block_spec,
                    global_discrete_interval,
                    subdiagrams,
                    state_machines,
                    acausal_networks,
                    parent_path=parent_path + [block_name],
                    parent_ui_id_path=parent_ui_id_path + [block_ui_id],
                    namespace_params=namespace_params,
                    record_mode=record_mode,
                    recorded_signals=recorded_signals,
                    start_time=start_time,
                )
            # branch for normal 'blocks'. not subdiagrams.
            else:
                common_kwargs = {
                    "name": block_name,
                    "ui_id": block_ui_id,
                }
                parameters = eval_parameters(
                    instance_parameters=block_spec.parameters,
                    call_site_namespace=namespace_params,
                    name_path=block_name_path,
                    ui_id_path=block_ui_id_path,
                )
                # branches for 'special' blocks
                if block_spec.type == "core.StateMachine":
                    block = block_interface.get_block_fcn(block_spec.type)(
                        block_spec=block_spec,
                        discrete_interval=global_discrete_interval,
                        state_machine_diagram=state_machines[
                            block_spec.state_machine_diagram_id
                        ],
                        **common_kwargs,
                        **parameters,
                    )
                elif block_spec.type == "core.ModelicaFMU":
                    block = block_interface.get_block_fcn(block_spec.type)(
                        block_spec=block_spec,
                        discrete_interval=global_discrete_interval,
                        start_time=start_time,
                        **common_kwargs,
                        **parameters,
                    )
                elif block_spec.type.startswith("acausal."):
                    # FIXME: the above condition is terrible
                    # NOTE: acausal blocks were already isolated and processed, nothing to do here.
                    # NOTE: we need to flag that the 'block' being processed is not a 'LeafSystem'
                    # to avoid further processing (only applicable to LeafSystems) being applied to it.
                    is_phealf_block = True
                else:
                    block = block_interface.get_block_fcn(block_spec.type)(
                        block_spec=block_spec,
                        discrete_interval=global_discrete_interval,
                        **common_kwargs,
                        **parameters,
                    )

                if block_spec.inputs and not is_phealf_block:
                    input_port_names = [port.name for port in block_spec.inputs]
                    for port_name, port in zip(input_port_names, block.input_ports):
                        port.name = port_name

                if block_spec.outputs and not is_phealf_block:
                    output_port_names = [port.name for port in block_spec.outputs]
                    for port_name, port in zip(output_port_names, block.output_ports):
                        port.name = port_name

            # NOTE: Here we assume that the port order is the same in the frontend and wildcat
            # Log anything with record=True
            if block_spec.outputs and recorded_signals is not None:
                for i, port in enumerate(block_spec.outputs):
                    if port.record or record_mode == "all":
                        if is_phealf_block:
                            if block_ui_id in acausal_system.outports_maps.keys():
                                if (
                                    i
                                    in acausal_system.outports_maps[block_ui_id].keys()
                                ):
                                    # when the block is part of a acausal_system, we map its 'recorded'
                                    # ports to the corresponding acausal_system port.
                                    port_id = acausal_system.outports_maps[block_ui_id][
                                        i
                                    ]
                                    port_path = parent_path + [block_name, port.name]
                                    port_path = ".".join(port_path)
                                    logger.debug("Recording %s", port_path)
                                    recorded_signals[port_path] = (
                                        acausal_system.output_ports[port_id]
                                    )
                            else:
                                # this port is an acausal port, do nothing.
                                continue
                        else:
                            port_path = parent_path + [block_name, port.name]
                            port_path = ".".join(port_path)
                            logger.debug("Recording %s", port_path)

                            if i < 0 or i >= len(block.output_ports):
                                # This unlikely error can happen when for instance there's
                                # an invalid JSON with more i/o ports than what wildcat
                                # defines (eg. old StateSpace blocks)
                                raise LegacyBlockConfigurationError(
                                    message=f"Output port index {i} out of range "
                                    f"(0-{len(block.output_ports)-1})",
                                    name_path=block_name_path,
                                    ui_id_path=block_ui_id_path,
                                    port_index=i,
                                    port_name=port.name,
                                    port_direction="out",
                                )

                            recorded_signals[port_path] = block.output_ports[i]

            if not is_phealf_block:
                builder.add(block)
                blocks[block_name] = block

        except (StaticError, BlockInitializationError) as exc:
            raise exc
        except Exception as exc:
            traceback.print_exc()
            if isinstance(exc, CollimatorError):
                # Avoid repetition in the error message by only keeping this info in
                # the top-level error
                exc.name_path = None
                exc.system_id = None
            path = ".".join(block_name_path)
            raise BlockInitializationError(
                message=f"Failed to create block {path} of type {block_spec.type}",
                system=block,
                name_path=block_name_path,
                ui_id_path=block_ui_id_path,
            ) from exc

    # Export the input port of any Inport
    for input_port_id_key in exported_inputs:
        inport_name = blocks[input_port_id_key].name
        builder.export_input(blocks[input_port_id_key].input_ports[0], name=inport_name)

    # Export the output port of any Outport
    for output_port_id_key in exported_outputs:
        outport_name = blocks[output_port_id_key].name
        builder.export_output(
            blocks[output_port_id_key].output_ports[0], name=outport_name
        )

    for link in diagram.links:
        if (
            (link.src is None)
            or (link.dst is None)
            or (link.src.node not in block_uuid_to_name)
            or (link.dst.node not in block_uuid_to_name)
            or link.uuid in acausal_links.keys()  # handled in build_acausal_system()
        ):
            continue

        src_block_name = block_uuid_to_name[link.src.node]
        dst_block_name = block_uuid_to_name[link.dst.node]
        src_port_index = int(link.src.port)
        dst_port_index = int(link.dst.port)

        if acausal_network is not None:
            if link.uuid in causal_inlinks.keys():
                dst_block_name = acausal_system.name
                dst_port_index = acausal_system.inports_maps[link.dst.node][
                    dst_port_index
                ]
            elif link.uuid in causal_outlinks.keys():
                src_block_name = acausal_system.name
                src_port_index = acausal_system.outports_maps[link.src.node][
                    src_port_index
                ]
        else:
            # These unlikely errors can happen when for instance there's
            # an invalid JSON with more i/o ports than what wildcat
            # defines (eg. old StateSpace blocks)
            # NOTE: these errors cannot happen for phealf related links.
            if src_port_index < 0 or src_port_index >= len(
                blocks[src_block_name].output_ports
            ):
                raise LegacyBlockConfigurationError(
                    f"Invalid src port index {src_port_index} for block {blocks[src_block_name].name}",
                    name_path=block_name_path,
                    ui_id_path=block_ui_id_path,
                )
            elif dst_port_index < 0 or dst_port_index >= len(
                blocks[dst_block_name].input_ports
            ):
                raise LegacyBlockConfigurationError(
                    f"Invalid dst port index {dst_port_index} for block {blocks[dst_block_name].name}",
                    name_path=block_name_path,
                    ui_id_path=block_ui_id_path,
                )

        builder.connect(
            blocks[src_block_name].output_ports[src_port_index],
            blocks[dst_block_name].input_ports[dst_port_index],
        )

    return builder.build(name=name, ui_id=ui_id, parameters=model_parameters)


def identify_acausal_networks(
    name: str,
    diagram: model_json.Diagram,
    subdiagrams: dict[str, model_json.Diagram],
    parent_ui_id_path: list[str] = None,
    parent_path: list[str] = None,
    ui_id: str = None,
    acausal_networks: dict[str, AcausalNetwork] = None,
) -> dict[str, AcausalNetwork]:
    """
    Acausal diagrams are a subset of the diagram on a canvas. The subset is made up of
    all the acausal component blocks on the canvas, and all the link between two acausal
    ports of those blocks.
    Acausal diagrams cannot be processed the same way as causal diagrams, e.g. create one
    LeafSystem at a time, connect the inputs/outputs, and then 'build' the diagram. Acausal
    diagrams must have all components and network equations collection into one equation
    set, and from thei equation set a single LeafSystem (acausal_system) is produce.
    Therefore, before the model traversal for creating LeafSystems and building diagrams,
    we have this model traversal which collects the acausal blocks and links that make
    up the acausal diagram in a given canvas. this data is retained, and then when the
    same canvas is reached in the second model traversal, first we will process the acausal
    block/inks to create the Acausal LeafSystem (acausal_system), and integrate that into the
    rest of the diagram on that canvas replacing the acausal blocks and links that were
    there in the UI representation.
    In this traversal, there is no attempt to evaluate parameters, nor create class
    instances for blocks. The goal is only to isolate the subset of the diagram that is
    the acausal diagram. As such, we can treat submodels the same as groups.
    """

    if parent_path is None:
        parent_path = []

    if parent_ui_id_path is None:
        parent_ui_id_path = []

    # The "node_spec" passed here is the "node" specification that doesn't
    # contain the actual blocks. the info about the block that constains the
    # subdiagram, e.g. submodel instance.

    # acausal networks only supported within a single canvas for now
    if acausal_networks is None:
        acausal_networks = {}
    # this data is retained for use in the next model traversal.
    blocks = {}  # dict{acausal_block_uuid:block_spec}
    acausal_links = {}  # links between two acausal ports
    causal_inlinks = {}  # links between causal_block and actuator_causal_port
    causal_outlinks = {}  # links between sensor_causal_port and causal_block

    # this data is only used here locally, and tossed on departure.
    acausal_system_input_ui_ids = {}  # dict{block_ui_id:list(port_idx)}
    acausal_system_output_ui_ids = {}  # dict{block_ui_id:list(port_idx)}
    block_uuid_to_name = {}

    for block_spec in diagram.nodes:
        # block names are used as locally (in this canvas) unique identifiers
        block_name = block_spec.name
        block_ui_id = block_spec.uuid
        block_uuid_to_name[block_ui_id] = block_name

        # these are used for rich errors before the block is created
        block_name_path = parent_path + [block_name]
        block_ui_id_path = parent_ui_id_path + [block_ui_id]

        # branche for recursing into subdiagrams
        if block_spec.type == "core.ReferenceSubmodel":
            # acausal diagrams under submodels are processed in RegisterSubmodel.
            continue
        elif block_spec.type in (
            "core.Group",
            "core.Submodel",
        ):
            acausal_networks = identify_acausal_networks(
                name=block_name,
                diagram=subdiagrams.get_diagram(block_spec.uuid),
                subdiagrams=subdiagrams,
                parent_path=block_name_path,
                parent_ui_id_path=block_ui_id_path,
                ui_id=block_ui_id,
                acausal_networks=acausal_networks,
            )

        # branch for identifying acausal component blocks
        elif block_spec.type.startswith("acausal."):
            # FIXME: the above condition is terrible
            blocks[block_ui_id] = block_spec
            # print(f"acausal block inputs: {block_spec.inputs}")
            if block_spec.inputs:
                for idx, ioport in enumerate(block_spec.inputs):
                    # FIXME, we need a more explicit way to identify explicit ports of acausal blocks
                    if ioport.variant.variant_kind is None:
                        if block_ui_id not in acausal_system_input_ui_ids.keys():
                            acausal_system_input_ui_ids[block_ui_id] = []
                        port_ids = acausal_system_input_ui_ids[block_ui_id]
                        port_ids.append(idx)
                        acausal_system_input_ui_ids[block_ui_id] = port_ids

            # print(f"acausal block outputs: {block_spec.outputs}")
            if block_spec.outputs:
                for idx, ioport in enumerate(block_spec.outputs):
                    # FIXME, we need a more explicit way to identify explicit ports of acausal blocks
                    if ioport.variant.variant_kind is None:
                        if block_ui_id not in acausal_system_output_ui_ids.keys():
                            acausal_system_output_ui_ids[block_ui_id] = []
                        port_ids = acausal_system_output_ui_ids[block_ui_id]
                        port_ids.append(idx)
                        acausal_system_output_ui_ids[block_ui_id] = port_ids

    acausal_system_block_ui_ids = set(blocks.keys())
    for link in diagram.links:
        if (
            (link.src is None)
            or (link.dst is None)
            or (link.src.node not in block_uuid_to_name)
            or (link.dst.node not in block_uuid_to_name)
        ):
            continue

        src_block_name = block_uuid_to_name[link.src.node]
        dst_block_name = block_uuid_to_name[link.dst.node]
        src_port_index = int(link.src.port)
        dst_port_index = int(link.dst.port)

        node_ui_id_set = set([link.src.node, link.dst.node])
        # when the dst of a link is a block in the acausal_system network, this is a acausal_system
        # input, and vice versa for output. Ah, but not quite, read comments below.
        is_acausal_system_input = False
        is_acausal_system_output = False

        # this logic identifies when the dst port of this link has previously been identified
        # as a causal inport of an acausal source block. Since we only want to find causal links,
        # it is imperative that the port_side='inputs', since if port_side='outputs' means by
        # definition that this port cannot be a causal inport.
        if link.dst.node in acausal_system_input_ui_ids.keys():
            port_side_ok = True
            if link.dst.port_side:
                if link.dst.port_side != "inputs":
                    port_side_ok = False
            if (
                port_side_ok
                and dst_port_index in acausal_system_input_ui_ids[link.dst.node]
            ):
                is_acausal_system_input = True
        # similar to the above comment, but for sensor blocks, and hence causal outports, and
        # port_side (if present) must be 'outputs'.
        elif link.src.node in acausal_system_output_ui_ids.keys():
            port_side_ok = True
            if link.src.port_side:
                if link.src.port_side != "outputs":
                    port_side_ok = False
            if (
                port_side_ok
                and src_port_index in acausal_system_output_ui_ids[link.src.node]
            ):
                is_acausal_system_output = True

        if is_acausal_system_output and is_acausal_system_input:
            raise NotImplementedError(
                f"Direct causal connections between acausal sensors and actuators is not supported."
                f"Offending blocks: {src_block_name}, {dst_block_name}"
            )

        if (
            node_ui_id_set.issubset(acausal_system_block_ui_ids)
            and not is_acausal_system_input
            and not is_acausal_system_output
        ):
            # when both src and dst blocks are blocks in the physical network, and neither
            # has been identified as an explicit connection to the physical network, this
            # means this link is between two acausal ports.
            acausal_links[link.uuid] = link
        elif is_acausal_system_input and link.src.node not in blocks.keys():
            # dst node is a acausal_system actuator block, and the other is not an acausal block.
            # this case can only be a causal input to the acausal_system.
            causal_inlinks[link.uuid] = link
        elif is_acausal_system_output and link.dst.node not in blocks.keys():
            # src node is a acausal_system sensor block, and the other is not an acausal block.
            # this case can only be a causal output from the acausal_system.
            causal_outlinks[link.uuid] = link

    if blocks:
        acausal_network = AcausalNetwork(
            blocks=blocks,
            acausal_links=acausal_links,
            causal_inlinks=causal_inlinks,
            causal_outlinks=causal_outlinks,
        )
        parent_ui_id_path_str = ".".join(parent_ui_id_path)
        acausal_networks[parent_ui_id_path_str] = acausal_network

    return acausal_networks


def build_acausal_system(
    acausal_network: AcausalNetwork,
    namespace_params,
    name: str,
    parent_path: list[str],
    parent_ui_id_path: list[str],
) -> LeafSystem:
    """
    this builds an AcausalDiagram object using the data for blocks and links that was collected
    for the acausal_system during the model traversal in identify_acausal_networks().
    """
    blocks = acausal_network.blocks
    links = acausal_network.acausal_links

    # create acausal component objects from block_specs
    uuid_to_comp = {}
    comp_to_uuid = {}
    comp_to_inports_map = {}
    comp_to_outports_map = {}
    ev = EqnEnv()
    for block_spec in blocks.values():
        block_name_path = parent_path + [block_spec.name]
        block_ui_id_path = parent_ui_id_path + [block_spec.uuid]
        common_kwargs = {"name": block_spec.name}
        parameters = eval_parameters(
            instance_parameters=block_spec.parameters,
            call_site_namespace=namespace_params,
            name_path=block_name_path,
            ui_id_path=block_ui_id_path,
        )
        comp = block_interface.get_block_fcn(block_spec.type)(
            ev,
            block_spec=block_spec,
            **common_kwargs,
            **parameters,
        )
        uuid_to_comp[block_spec.uuid] = comp
        comp_to_uuid[comp] = block_spec.uuid

        # make a maps between json port index and component port name
        inports_map = {}
        for i, port in enumerate(block_spec.inputs):
            if port.variant.variant_kind is None:
                # this means it is a causal port
                # inports_map[i] = port.name
                inports_map[port.name] = i
        if inports_map:
            comp_to_inports_map[comp] = inports_map

        outports_map = {}
        for i, port in enumerate(block_spec.outputs):
            if port.variant.variant_kind is None:
                # this means it is a causal port
                # outports_map[i] = port.name
                outports_map[port.name] = i
        if outports_map:
            comp_to_outports_map[comp] = outports_map

    # print(f"\n\n{comp_to_inports_map=}\n\n")
    # print(f"\n\n{comp_to_outports_map=}\n\n")

    # create acausal diagram object
    ad = AcausalDiagram(name=name, comp_list=uuid_to_comp.values())

    # define connections in the acausal network
    for link_uuid, link in links.items():
        cmp_a = uuid_to_comp[link.src.node]
        cmp_b = uuid_to_comp[link.dst.node]

        src_port_index = int(link.src.port)
        dst_port_index = int(link.dst.port)

        # FIXME: this is a garnly nasty hack on top of the one below.
        # the UI has peculiar way of recording how links are connected when
        # they are between acausal ports (in order to deal with the any to any freedom).
        # normally, 'src' means the round exit port, and 'dst' means the triangle
        # entry port. but for acausal connections, this doesn't necessarily hold.
        # there is an optional qualifier in end point (e.g. 'src') called 'port_side'.
        # port_side overrides the src->round and dst->triangle rule, so that it becomes
        # port_side='outputs'->round, port_side='inputs'->triangle.
        # this means that the 'sign' of the hacky +/- index used below is controlled
        # by the port_side field when it is present.
        cmp_a_port_index_sign = 1
        if link.src.port_side == "inputs":
            cmp_a_port_index_sign = -1
        cmp_b_port_index_sign = -1
        if link.dst.port_side == "outputs":
            cmp_b_port_index_sign = 1

        # FIXME: this is a wicked nasty hack
        # acausal components ports should really be referred to by name, but json
        # links refer to ports by src/dst+index. acausal components do not have src/dst,
        # so to differentiate here we make dst index negative, that way components can
        # use the index only to know which port is connected.
        # as such, we have to index starting at one because we cant differentiate between
        # 0 and -0.
        cmp_a_port_idx = (src_port_index + 1) * cmp_a_port_index_sign
        cmp_b_port_idx = (dst_port_index + 1) * cmp_b_port_index_sign
        # print(f"{cmp_a=} {cmp_a_port_idx=} {cmp_a.port_idx_to_name=}")
        # print(f"{cmp_b=} {cmp_b_port_idx=} {cmp_b.port_idx_to_name=}")
        port_a = cmp_a.port_idx_to_name[cmp_a_port_idx]
        port_b = cmp_b.port_idx_to_name[cmp_b_port_idx]
        ad.connect(cmp_a, port_a, cmp_b, port_b)

    # create an acausal compiler object, and generate the acausal_system
    compiler = AcausalCompiler(ev, ad, verbose=False)
    acausal_system = compiler.generate_acausal_system(name=name)

    # This creats a map of the form {cmp_ui_id: {cmp_port_id: acausal_system_port_id}} which is
    # required for:
    #   1] reconnecting any out going links from the original diagram components'
    #   causal out ports, to the acausal_system out ports.
    #   2] asserting that a acausal_system out port should be recorded by checking if the
    #   corresponding component casual out port was requested to be recorded.
    # NOTE: the reason why we create the map by iterating over acausal_system.outsym_to_portid
    # as opposed to acausal_network.causal_outlinks is because a sensor component causal
    # outport is not necessarily connected to any thing, but we still need this map for
    # item 2 above.
    outports_maps: dict[str, dict[int, int]] = {}
    if acausal_system.outsym_to_portid is not None:
        for sym, acausal_system_port_id in acausal_system.outsym_to_portid.items():
            cmp = ad.sym_to_cmp[sym]
            cmp_ui_id = comp_to_uuid[cmp]
            ports_map = comp_to_outports_map[cmp]
            # FIXME: this is very hacky. should have the Sym object retain the
            # base string that is also the component port name.
            port_name = sym.name.replace(cmp.name + "_", "")
            cmp_port_id = ports_map[port_name]
            cmp_port_map = outports_maps.get(cmp_ui_id, {})
            cmp_port_map[cmp_port_id] = acausal_system_port_id
            outports_maps[cmp_ui_id] = cmp_port_map

    # This repeats the above, but for in ports. Obvioulsy, the 'recorded' part
    # doesn't apply.
    # NOTE: here we could iterate over acausal_network.causal_inlinks, but that
    # would require that comp_to_inports_map be made the other way around. So to
    # keep things simple, we just produce the map the same way for both in and out.
    inports_maps: dict[str, dict[int, int]] = {}
    if acausal_system.insym_to_portid is not None:
        for sym, acausal_system_port_id in acausal_system.insym_to_portid.items():
            cmp = ad.sym_to_cmp[sym]
            cmp_ui_id = comp_to_uuid[cmp]
            ports_map = comp_to_inports_map[cmp]
            # FIXME: this is very hacky. should have the Sym object retain the
            # base string that is also the component port name.
            port_name = sym.name.replace(cmp.name + "_", "")
            cmp_port_id = ports_map[port_name]
            cmp_port_map = inports_maps.get(cmp_ui_id, {})
            cmp_port_map[cmp_port_id] = acausal_system_port_id
            inports_maps[cmp_ui_id] = cmp_port_map

    # save these maps for later lookups when writing signal_types.json
    # FIXME: saving in the object is pretty hacky but way simpler than passing around
    acausal_system.acausal_network = acausal_network
    acausal_system.outports_maps = outports_maps
    acausal_system.inports_maps = inports_maps

    return acausal_system


def simulation_settings(
    config: model_json.Configuration, recorded_signals: dict[str, SystemCallback] = None
):
    sim_output_mode_lookup = {
        "auto": ResultsMode.auto,
        "discrete_steps_only": ResultsMode.discrete_steps_only,
        "fixed_interval": ResultsMode.fixed_interval,
    }
    sim_output_mode = sim_output_mode_lookup.get(
        config.sim_output_mode, ResultsMode.auto
    )

    method = config.solver.method
    if method in ["auto", "non-stiff"]:
        pass
    elif method in ["RK45"]:
        method = "non-stiff"
    elif method in ["stiff", "BDF", "Kvaerno5"]:
        method = "stiff"
    else:
        raise ValueError(f"Unsupported solver method: {config.solver.method}")

    numerical_backend = config.numerical_backend or "auto"
    if numerical_backend not in ["auto", "numpy", "jax"]:
        raise ValueError(f"Unsupported numerical backend: {config.numerical_backend}")
    if numerical_backend == "auto":
        numerical_backend = backend.active_backend

    results_options = ResultsOptions(
        mode=sim_output_mode,
        max_results_interval=config.max_results_interval,
        fixed_results_interval=config.fixed_results_interval,
    )

    simulator_options = SimulatorOptions(
        math_backend=numerical_backend,
        max_major_steps=config.max_major_steps,
        max_major_step_length=config.sample_time,
        min_minor_step_size=config.solver.min_step,
        max_minor_step_size=config.solver.max_step,
        atol=config.solver.absolute_tolerance,
        rtol=config.solver.relative_tolerance,
        ode_solver_method=method,
        return_context=False,
        recorded_signals=recorded_signals,
        save_time_series=recorded_signals is not None and len(recorded_signals) > 0,
    )

    logger.info("Simulation settings: %s", simulator_options)
    logger.info("Results settings: %s", results_options)

    return results_options, simulator_options
