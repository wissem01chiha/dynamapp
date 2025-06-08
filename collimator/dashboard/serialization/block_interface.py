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

import importlib
from importlib.util import find_spec as try_import
import inspect
from types import MappingProxyType

import numpy as np

from collimator import library
from collimator.dashboard.serialization import model_json
import collimator.library.state_machine as sm_lib
from collimator.logging import logger
from collimator.experimental import electrical, rotational, translational, thermal

# The ClassNameBlock functions are thin layers between a serialized JSON
# description of the block and its actual implementation in Wildcat. In
# most cases, there should not be a need for such a wrapper. But in some
# cases, the JSON description is a bit different from the implemented block
# for a variety of reasons:
# - a flag selects an implementation or another (eg. PythonScript),
# - the json description contains strings in place of special objects (eg. ROS2
#   uses type arguments),


def _process_user_define_ports(block_spec: model_json.Node):
    inputs_ = [d.name for d in block_spec.inputs]
    outputs_ = [d.name for d in block_spec.outputs]
    return inputs_, outputs_


def PythonScriptBlock(
    block_spec: model_json.Node,
    dt: float = None,
    discrete_interval: float = None,  # dt vs. discrete_interval is confusing
    init_script: str = "",
    user_statements: str = "",
    finalize_script: str = "",  # presently ignored
    accelerate_with_jax: bool = False,
    **kwargs,
):
    inputs_, outputs_ = _process_user_define_ports(block_spec)

    common_kwargs = {
        "name": kwargs.pop("name", None),
        "ui_id": kwargs.pop("ui_id", None),
    }

    block_cls = (
        library.CustomJaxBlock if accelerate_with_jax else library.CustomPythonBlock
    )

    return _wrap(block_cls)(
        inputs=inputs_,
        outputs=outputs_,
        dt=discrete_interval if dt is None else dt,
        init_script=init_script,
        user_statements=user_statements,
        finalize_script=finalize_script,
        accelerate_with_jax=accelerate_with_jax,
        time_mode=block_spec.time_mode,
        parameters=kwargs,
        **common_kwargs,
    )


def IntegratorBlock(initial_states, **kwargs):
    return _wrap(library.Integrator)(
        initial_state=initial_states,
        **kwargs,
    )


def CosineWaveBlock(**kwargs):
    kwargs["phase"] = np.pi / 2 + kwargs["phase"]
    return _wrap(library.Sine)(
        **kwargs,
    )


def DataSourceBlock(**kwargs):
    # avoid user-visible warning (this parameter is for cloud integration)
    _ = kwargs.pop("data_integration_id", None)
    return _wrap(library.DataSource)(**kwargs)


def DemuxBlock(block_spec, **kwargs):
    n_out = len(block_spec.outputs)
    return _wrap(library.Demultiplexer)(
        n_out,
        **kwargs,
    )


def StackBlock(block_spec, axis, **kwargs):
    n_in = len(block_spec.inputs)
    return _wrap(library.Stack)(
        n_in,
        axis,
        **kwargs,
    )


def TransferFunctionBlock(numerator_coefficients, denominator_coefficients, **kwargs):
    return _wrap(library.TransferFunction)(
        numerator_coefficients,
        denominator_coefficients,
        **kwargs,
    )


def StateSpaceBlock(A, B, C, D, initial_states=None, **kwargs):
    return _wrap(library.LTISystem)(
        A,
        B,
        C,
        D,
        initialize_states=initial_states,
        **kwargs,
    )


def ModelicaFMUBlock(
    file_name: str,
    discrete_interval: float,
    block_spec: model_json.Node,
    name: str = None,
    fmu_guid: str = None,
    start_time: float = 0.0,
    **kwargs,
):
    input_names = [d.name for d in block_spec.inputs]
    output_names = [d.name for d in block_spec.outputs]

    return _wrap(library.ModelicaFMU)(
        file_name=file_name,
        dt=discrete_interval,
        name=name,
        input_names=input_names,
        output_names=output_names,
        start_time=start_time,
        parameters=kwargs,
    )


def IntegratorDiscreteBlock(discrete_interval, initial_states, **kwargs):
    return _wrap(library.IntegratorDiscrete)(
        dt=discrete_interval,
        initial_state=initial_states,
        **kwargs,
    )


def PIDDiscreteBlock(
    block_spec,
    discrete_interval,
    Kp,
    Ki,
    Kd,
    filter_type="none",
    filter_coefficient=1.0,
    **kwargs,
):
    dt = kwargs.pop("dt", None)
    _ = kwargs.pop(
        "tuning_time", None
    )  # not a param of the block, so remove if present.
    return _wrap(library.PIDDiscrete)(
        kp=Kp,
        ki=Ki,
        kd=Kd,
        dt=dt if dt is not None else discrete_interval,
        filter_type=filter_type,
        filter_coefficient=filter_coefficient,
        **kwargs,
    )


def PIDBlock(Kp, Ki, Kd, N=100, **kwargs):
    _ = kwargs.pop(
        "tuning_time", None
    )  # not a param of the block, so remove if present.
    return _wrap(library.PID)(
        kp=Kp,
        ki=Ki,
        kd=Kd,
        n=N,
        **kwargs,
    )


def DiscreteInitializerBlock(discrete_interval, initial_state, **kwargs):
    dt = kwargs.pop("dt", None)
    return _wrap(library.DiscreteInitializer)(
        initial_state=initial_state,
        dt=dt if dt is not None else discrete_interval,
        **kwargs,
    )


def TransferFunctionDiscreteBlock(
    discrete_interval, numerator_coefficients, denominator_coefficients, **kwargs
):
    dt = kwargs.pop("dt", None)
    return _wrap(library.TransferFunctionDiscrete)(
        dt=dt if dt is not None else discrete_interval,
        num=numerator_coefficients,
        den=denominator_coefficients,
        **kwargs,
    )


def RandomNumberBlock(discrete_interval, distribution, **kwargs):
    # Custom wrapper for some deserialization and type conversions
    dt = kwargs.pop("dt", None)
    seed = kwargs.pop("seed", None)
    shape = kwargs.pop("shape", ())
    dtype = kwargs.pop("dtype", None)

    if seed is not None:
        seed = int(seed)
    if shape is not None:
        shape = tuple(map(int, shape))
    if dtype is not None:
        dtype = np.dtype(dtype)

    return _wrap(library.RandomNumber)(
        dt=dt if dt is not None else discrete_interval,
        distribution=distribution,
        seed=seed,
        shape=shape,
        dtype=dtype,
        **kwargs,
    )


def _wrap_ros2_block(cls: type, io_ports: model_json.IOPort, **kwargs):
    # The ROS2 blocks are very experimental and require a complex setup before use.
    # This glue code is super hacky and intended for quick demos. We can absolutely
    # release this and example notebooks (using the dashboard API) but we need to
    # make it clear this is an unstable feature.

    topic = kwargs.pop("topic", None)
    msg_type = kwargs.pop("msg_type", None)
    message_class: type = None

    rclpy = try_import("rclpy")
    geometry_msgs = try_import("geometry_msgs")
    std_msgs = try_import("std_msgs")
    if rclpy is None or geometry_msgs is None or std_msgs is None:
        raise ImportError("ROS2 blocks require rclpy, geometry_msgs and std_msgs")

    # FIXME/TODO: primitive types are not actually supported...
    # if msg_type == "std_msgs/Float64":
    #     message_class = std_msgs.msg.Float64
    # elif msg_type == "std_msgs/Int32":
    #     message_class = std_msgs.msg.Int32
    # elif msg_type == "std_msgs/String":
    #     message_class = std_msgs.msg.String

    ns = ".".join(msg_type.split("/")[:-1])
    classname = msg_type.split("/")[-1]
    try:
        module = importlib.import_module(ns)
        message_class = getattr(module, classname)
    except ImportError as e:
        raise ImportError(f"Could not import message class {msg_type}") from e
    except AttributeError as e:
        raise AttributeError(f"Could not find class {msg_type} in module {ns}") from e

    # FIXME: abusing port parameters to carry ros2 message field/type info
    # A proper fix would be not to set these awkward definitions on this block
    # but rather on a bus creator block. Note: the implementation requires
    # use of ordered dicts where the order matches that of the I/O ports.
    fields: dict[str, str] = {}
    for io_port in io_ports:
        field = io_port.parameters.get("field_name", None)
        dtype = io_port.parameters.get("dtype", None)
        if field is None or dtype is None:
            raise ValueError("Field name and data type must be provided for all inputs")
        fields[field.value] = dtype.value

    return _wrap_discrete(cls)(
        topic=topic, msg_type=message_class, fields=fields, **kwargs
    )


def Ros2PublisherBlock(block_spec: model_json.Node, **kwargs):
    return _wrap_ros2_block(library.Ros2Publisher, block_spec.inputs, **kwargs)


def Ros2SubscriberBlock(block_spec: model_json.Node, **kwargs):
    return _wrap_ros2_block(library.Ros2Subscriber, block_spec.outputs, **kwargs)


def WhiteNoiseBlock(discrete_interval, **kwargs):
    # Custom wrapper for some deserialization and type conversions
    corr_time = kwargs.pop("correlation_time", None)
    noise_power = kwargs.pop("noise_power", 1.0)
    num_samples = int(kwargs.pop("num_samples", 10))
    seed = kwargs.pop("seed", None)
    shape = kwargs.pop("shape", ())
    dtype = kwargs.pop("dtype", None)

    if dtype is not None:
        dtype = np.dtype(dtype)

    return _wrap(library.WhiteNoise)(
        correlation_time=(corr_time if corr_time is not None else discrete_interval),
        noise_power=noise_power,
        num_samples=num_samples,
        seed=seed,
        shape=shape,
        dtype=dtype,
        **kwargs,
    )


def KalmanFilterBlock(discrete_interval, **kwargs):
    # Custom wrapper for some deserialization and type conversions

    dt = kwargs.pop("dt", None)  # needed for all
    if dt is None:
        dt = discrete_interval

    matrices_provided = kwargs.pop("matrices_provided", True)
    use_ihkf = kwargs.pop("use_ihkf", False)

    # For continuous-time plant
    plant_submodel_uuid = kwargs.pop("plant_submodel_uuid", None)
    discretization_method = kwargs.pop("discretization_method", "zoh")
    x_eq = kwargs.pop("x_eq", None)
    u_eq = kwargs.pop("u_eq", None)

    # For input matrices
    A = kwargs.pop("A")
    B = kwargs.pop("B")
    C = kwargs.pop("C", None)
    D = kwargs.pop("D", None)

    # For both continuous-time plant and input matrices
    G = kwargs.pop("G", None)
    Q = kwargs.pop("Q", None)
    R = kwargs.pop("R", None)
    x_hat_0 = kwargs.pop("x_hat_0", None)

    # only needed if use_ihkf is False
    P_hat_0 = kwargs.pop("P_hat_0", None)

    # Convert list inputs to numpy arrays
    if x_eq is not None:
        x_eq = np.array(x_eq)
    if u_eq is not None:
        u_eq = np.array(u_eq)
    if A is not None:
        A = np.array(A)
    if B is not None:
        B = np.array(B)
    if C is not None:
        C = np.array(C)
    if D is not None:
        D = np.array(D)
    if G is not None:
        G = np.array(G)
    if Q is not None:
        Q = np.array(Q)
    if R is not None:
        R = np.array(R)
    if x_hat_0 is not None:
        x_hat_0 = np.array(x_hat_0)
    if P_hat_0 is not None:
        P_hat_0 = np.array(P_hat_0)

    if matrices_provided:
        if use_ihkf:
            return _wrap(library.InfiniteHorizonKalmanFilter)(
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                G=G,
                Q=Q,
                R=R,
                x_hat_0=x_hat_0,
                **kwargs,
            )
        else:
            return _wrap(library.KalmanFilter)(
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                G=G,
                Q=Q,
                R=R,
                x_hat_0=x_hat_0,
                P_hat_0=P_hat_0,
                **kwargs,
            )
    else:
        # NOTE: there is no way to specify custom parameters or re-run the
        # submodel init script at this point.
        plant = library.ReferenceSubdiagram.create_diagram(
            plant_submodel_uuid,
            instance_parameters={},
            instance_name="ct_plant_for_kf",
        )
        if use_ihkf:
            kf = _wrap(
                library.InfiniteHorizonKalmanFilter.global_filter_for_continuous_plant
            )(
                plant,
                x_eq,
                u_eq,
                dt,
                G=G,
                Q=Q,
                R=R,
                x_hat_0=x_hat_0,
                **kwargs,
            )
        else:
            kf = _wrap(library.KalmanFilter.global_filter_for_continuous_plant)(
                plant,
                x_eq,
                u_eq,
                dt,
                G=G,
                Q=Q,
                R=R,
                x_hat_0=x_hat_0,
                P_hat_0=P_hat_0,
                discretization_method=discretization_method,
                **kwargs,
            )

        return kf


def UnscentedKalmanFilterBlock(discrete_interval, **kwargs):
    # Custom wrapper for some deserialization and type conversions

    dt = kwargs.pop("dt", None)  # needed for all
    if dt is None:
        dt = discrete_interval

    # For continuous-time plant
    plant_submodel_uuid = kwargs.pop("plant_submodel_uuid", None)
    discretization_method = kwargs.pop("discretization_method", "zoh")

    # wheter noise statistics are provided discreteized
    discretized_noise = kwargs.pop("discretized_noise", False)

    # noise matrices
    G = kwargs.pop("G", None)
    Q = kwargs.pop("Q", None)
    R = kwargs.pop("R", None)
    x_hat_0 = kwargs.pop("x_hat_0", None)
    P_hat_0 = kwargs.pop("P_hat_0", None)

    plant = library.ReferenceSubdiagram.create_diagram(
        plant_submodel_uuid,
        instance_parameters={},
        instance_name="ct_plant_for_ukf",
    )

    ukf = _wrap(library.UnscentedKalmanFilter.for_continuous_plant)(
        plant=plant,
        dt=dt,
        G_func=lambda t: G,
        Q_func=lambda t: Q,
        R_func=lambda t: R,
        x_hat_0=x_hat_0,
        P_hat_0=P_hat_0,
        discretization_method=discretization_method,
        discretized_noise=discretized_noise,
    )

    return ukf


def ExtendedKalmanFilterBlock(discrete_interval, **kwargs):
    # Custom wrapper for some deserialization and type conversions

    dt = kwargs.pop("dt", None)  # needed for all
    if dt is None:
        dt = discrete_interval

    # For continuous-time plant
    plant_submodel_uuid = kwargs.pop("plant_submodel_uuid", None)
    discretization_method = kwargs.pop("discretization_method", "zoh")

    # wheter noise statistics are provided discreteized
    discretized_noise = kwargs.pop("discretized_noise", False)

    # noise matrices
    G = kwargs.pop("G", None)
    Q = kwargs.pop("Q", None)
    R = kwargs.pop("R", None)
    x_hat_0 = kwargs.pop("x_hat_0", None)
    P_hat_0 = kwargs.pop("P_hat_0", None)

    plant = library.ReferenceSubdiagram.create_diagram(
        plant_submodel_uuid,
        instance_parameters={},
        instance_name="ct_plant_for_ukf",
    )

    ekf = _wrap(library.ExtendedKalmanFilter.for_continuous_plant)(
        plant=plant,
        dt=dt,
        G_func=lambda t: G,
        Q_func=lambda t: Q,
        R_func=lambda t: R,
        x_hat_0=x_hat_0,
        P_hat_0=P_hat_0,
        discretization_method=discretization_method,
        discretized_noise=discretized_noise,
    )

    return ekf


def IOPortBlock(**kwargs):
    if "description" in kwargs:
        kwargs.pop("description")
    if "port_id" in kwargs:
        kwargs.pop("port_id")
    if "default_value" in kwargs:
        kwargs.pop("default_value")
    return _wrap(library.IOPort)(
        **kwargs,
    )


def PyTwinBlock(discrete_interval, **kwargs):
    # Custom wrapper for some deserialization and type conversions
    dt = kwargs.pop("dt", None)
    pytwin_file = kwargs.pop("pytwin_file")
    pytwin_config = kwargs.pop("pytwin_config", None)
    parameters = kwargs.pop("parameters", None)
    inputs = kwargs.pop("inputs", None)

    if parameters is not None:
        parameters = parameters.item()

    if inputs is not None:
        inputs = inputs.item()

    return _wrap(library.PyTwin)(
        pytwin_file=pytwin_file,
        dt=dt if dt is not None else discrete_interval,
        pytwin_config=pytwin_config,
        parameters=parameters,
        inputs=inputs,
        **kwargs,
    )


def _create_partial_transitions(data: model_json.StateMachineTransition):
    return sm_lib.StateMachineTransition(guard=data.guard, actions=data.actions)


# function for 'loaded json dataclasses' -> 'used in block dataclasses'
def _create_state_machine_data(load_sm: model_json.StateMachine):
    # process the state machine diagram
    # first enumerate the states with integers since we need a type that
    # can be stored in a wildcat state (i.e. not string)
    states_lookup = {node.uuid: idx for idx, node in enumerate(load_sm.nodes)}
    transitions_lookup = {trns.uuid: trns for trns in load_sm.links}

    # create the simplified state machine rep.
    states = {}
    intial_state = None
    inital_actions = load_sm.entry_point.actions
    for node in load_sm.nodes:
        idx = states_lookup[node.uuid]
        node_uuid = node.uuid
        transitions = []
        for exit_trsn_uuid in node.exit_priority_list:
            load_trns = transitions_lookup[exit_trsn_uuid]
            transition = _create_partial_transitions(load_trns)
            transition.dst = states_lookup[load_trns.destNodeId]
            transitions.append(transition)

        state = sm_lib.StateMachineState(name=node.name, transitions=transitions)
        states[idx] = state

        if node_uuid == load_sm.entry_point.dest_id:
            if intial_state is not None:
                raise ValueError("sm cannot have more that one entry point")
            intial_state = idx

    return sm_lib.StateMachineData(
        states=states,
        intial_state=intial_state,
        inital_actions=inital_actions,
    )


def StateMachineBlock(
    block_spec: model_json.Node,
    dt: str = None,
    discrete_interval: float = None,
    state_machine_diagram: model_json.StateMachine = None,
    **kwargs,
):
    sm_data = _create_state_machine_data(state_machine_diagram)
    inputs_, outputs_ = _process_user_define_ports(block_spec)

    return _wrap(library.StateMachine)(
        sm_data=sm_data,
        inputs=inputs_,
        outputs=outputs_,
        dt=float(dt) if dt is not None else discrete_interval,
        time_mode=block_spec.time_mode,
        **kwargs,
    )


def FilterDiscreteBlock(block_spec, discrete_interval, b_coefficients, **kwargs):
    dt = kwargs.pop("dt", None)
    filter_type = kwargs.pop("filter_type", None)
    _ = kwargs.pop("a_coefficients", None)  # b_coefficients only used for IIR filter
    if filter_type == "IIR":
        raise NotImplementedError("IIR filter not implement")
    return _wrap(library.FilterDiscrete)(
        dt=dt if dt is not None else discrete_interval,
        b_coefficients=b_coefficients,
        **kwargs,
    )


def PyTorchBlock(block_spec, **kwargs):
    num_inputs = kwargs.pop("num_inputs", None)
    num_outputs = kwargs.pop("num_outputs", None)

    if num_inputs is None:
        num_inputs = len(block_spec.inputs)
    else:
        num_inputs = int(num_inputs)

    if num_outputs is None:
        num_outputs = len(block_spec.outputs)
    else:
        num_outputs = int(num_outputs)

    return _wrap(library.PyTorch)(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        **kwargs,
    )


def TensorFlowBlock(block_spec, **kwargs):
    num_inputs = kwargs.pop("num_inputs", None)
    num_outputs = kwargs.pop("num_outputs", None)

    if num_inputs is None:
        num_inputs = len(block_spec.inputs)
    else:
        num_inputs = int(num_inputs)

    if num_outputs is None:
        num_outputs = len(block_spec.outputs)
    else:
        num_outputs = int(num_outputs)

    block = _wrap(library.TensorFlow)(**kwargs)

    file_name = kwargs["file_name"]
    add_batch_dim_to_inputs = _convert_bools(kwargs)["add_batch_dim_to_inputs"]

    if add_batch_dim_to_inputs:
        if block.num_inputs - 1 == num_inputs and block.num_outputs == num_outputs:
            return block
        else:
            raise ValueError(
                f"Number of inputs and outputs in block are {(num_inputs, num_outputs)} "
                f"but expected {(block.num_inputs-1, block.num_outputs)} after reading "
                f"the signature of the SavedModel {file_name}. Note that the parameter "
                f"`add_batch_dim_to_inputs` is set to `true`."
            )
    else:
        if block.num_inputs == num_inputs and block.num_outputs == num_outputs:
            return block
        else:
            raise ValueError(
                f"Number of inputs and outputs in block are {(num_inputs, num_outputs)} "
                f"but expected {(block.num_inputs, block.num_outputs)} after reading "
                f"the signature of the SavedModel {file_name}. Note that the parameter "
                f"`add_batch_dim_to_inputs` is set to `false`."
            )


def MuJoCoBlock(
    block_spec: model_json.Node,
    file_name: str,
    dt: float = None,
    discrete_interval: float = None,
    use_mjx: bool = True,
    **kwargs,
):
    if file_name is None:
        raise ValueError("MuJoCo block requires an XML file path")

    known_outports = [
        "qpos",
        "qvel",
        "act",
        "sensor_data",
        "video",
    ]
    custom_output_scripts = {}
    for i, output in enumerate(block_spec.outputs):
        if output.kind == "dynamic" or output.name not in known_outports:
            script = output.parameters.get("script", None)
            if not script or not script.is_string:
                raise ValueError(
                    f"Output port {output.name} ({i}) requires a valid script "
                    "that defines the numerical callback function `output_fn`."
                )
            custom_output_scripts[output.name] = script.value

    if use_mjx is False:
        # Non-MJX MuJoCo block is always discrete (acts like an FMU)
        return _wrap(library.MuJoCo)(
            file_name=file_name,
            dt=discrete_interval if dt is None else dt,
            custom_output_scripts=custom_output_scripts,
            **kwargs,
        )

    return _wrap(library.MJX)(
        file_name=file_name,
        dt=dt,
        custom_output_scripts=custom_output_scripts,
        **kwargs,
    )


def VideoSinkBlock(
    block_spec: model_json.Node,
    dt: float = None,
    discrete_interval: float = None,
    file_name: str = None,
    **kwargs,
):
    dt = discrete_interval if dt is None else dt
    file_name = "artifacts/" + (block_spec.name if file_name is None else file_name)
    return _wrap(library.VideoSink)(dt=dt, file_name=file_name, **kwargs)


def _convert_bools(params: dict):
    bools = {
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }
    return {k: bools.get(v, v) if isinstance(v, str) else v for k, v in params.items()}


def _filter_init_kwargs(cls: type, kwargs: dict):
    # Filters the parameters actually accepted by __init__ of the given block
    # class. This prevents failing simulations in case of a slightly invalid
    # JSON model.

    fn = cls.__init__ if isinstance(cls, type) else cls
    cls_init_param_defs = inspect.signature(fn).parameters

    allowed_kwargs = set(["ui_id", "name"])
    cls_init_param_names = set(cls_init_param_defs.keys())
    allowed_param_names = cls_init_param_names | allowed_kwargs

    unsupported_kwargs = []
    new_kwargs = {}
    for arg_name in kwargs.keys():
        if arg_name not in allowed_param_names:
            unsupported_kwargs.append(arg_name)
        else:
            new_kwargs[arg_name] = kwargs[arg_name]

    if len(unsupported_kwargs) > 0:
        logger.warning(
            "Found unsupported parameters for %s: %s, "
            "they will be not be passed to the block constructor.",
            cls.__name__,
            unsupported_kwargs,
        )

    return new_kwargs


def _wrap(block_cls):
    def _wrapped(*args, block_spec=None, discrete_interval=None, **kwargs):
        kwargs = _filter_init_kwargs(block_cls, kwargs)
        return block_cls(*args, **_convert_bools(kwargs))

    return _wrapped


def _wrap_reducer(block_cls):
    def _wrapped(*args, block_spec=None, discrete_interval=None, **kwargs):
        n_in = len(block_spec.inputs)
        kwargs = _filter_init_kwargs(block_cls, kwargs)
        return block_cls(n_in, **_convert_bools(kwargs))

    return _wrapped


def _wrap_discrete(block_cls):
    def _wrapped(*args, block_spec=None, discrete_interval=None, **kwargs):
        kwargs = _filter_init_kwargs(block_cls, kwargs)
        return block_cls(discrete_interval, **_convert_bools(kwargs))

    return _wrapped


# Mapping of block types to their corresponding block creation functions
# MappingProxyType makes this dictionary read-only
_fcn_map = MappingProxyType(
    {
        "core.Abs": _wrap(library.Abs),
        "core.Adder": _wrap_reducer(library.Adder),
        "core.Arithmetic": _wrap_reducer(library.Arithmetic),
        "core.BatteryCell": _wrap(library.BatteryCell),
        # bus creator
        # bus selector
        "core.Chirp": _wrap(library.Chirp),
        "core.Clock": _wrap(library.Clock),
        "core.DiscreteClock": _wrap_discrete(library.DiscreteClock),
        "core.Comparator": _wrap(library.Comparator),
        # conditional
        "core.Constant": _wrap(library.Constant),
        "core.CoordinateRotation": _wrap(library.CoordinateRotation),
        "core.CoordinateRotationConversion": _wrap(
            library.CoordinateRotationConversion
        ),
        "core.CosineWave": CosineWaveBlock,
        "core.CrossProduct": _wrap(library.CrossProduct),
        "core.DataSource": DataSourceBlock,
        "core.DeadZone": _wrap(library.DeadZone),
        # "core.Delay": _wrap(library.None),
        "core.Demux": DemuxBlock,
        "core.Derivative": _wrap(library.Derivative),
        "core.DerivativeDiscrete": _wrap_discrete(library.DerivativeDiscrete),
        "core.DiscreteInitializer": DiscreteInitializerBlock,
        "core.DotProduct": _wrap(library.DotProduct),
        # drive cycle
        "core.EdgeDetection": _wrap_discrete(library.EdgeDetection),
        "core.Exponent": _wrap(library.Exponent),
        "core.FilterDiscrete": FilterDiscreteBlock,
        "core.Gain": _wrap(library.Gain),
        # group implemented in model_interface.py
        "core.IfThenElse": _wrap(library.IfThenElse),
        # image segmentation
        # image source
        "core.Inport": IOPortBlock,
        "core.Integrator": IntegratorBlock,
        "core.IntegratorDiscrete": IntegratorDiscreteBlock,
        # iterator [and its loop control blocks]
        # linearized sysyem
        # State estimators from Kalman filter family
        "core.KalmanFilter": KalmanFilterBlock,
        "core.InfiniteHorizonKalmanFilter": _wrap(library.InfiniteHorizonKalmanFilter),
        "core.UnscentedKalmanFilter": UnscentedKalmanFilterBlock,
        "core.ExtendedKalmanFilter": ExtendedKalmanFilterBlock,
        "core.Log": _wrap(library.Logarithm),
        "core.LogicalOperator": _wrap(library.LogicalOperator),
        "core.LogicalReduce": _wrap(library.LogicalReduce),
        "core.LookupTable1d": _wrap(library.LookupTable1d),
        "core.LookupTable2d": _wrap(library.LookupTable2d),
        # loop blah, iterator blocks
        "core.MatrixConcatenation": _wrap_reducer(library.MatrixConcatenation),
        "core.MatrixInversion": _wrap(library.MatrixInversion),
        "core.MatrixMultiplication": _wrap_reducer(library.MatrixMultiplication),
        "core.MatrixTransposition": _wrap(library.MatrixTransposition),
        "core.MinMax": _wrap_reducer(library.MinMax),
        "core.ModelicaFMU": ModelicaFMUBlock,
        "core.Mux": _wrap_reducer(library.Multiplexer),
        "core.MuJoCo": MuJoCoBlock,
        # object detection
        "core.Offset": _wrap(library.Offset),
        "core.Outport": IOPortBlock,
        "core.PID": PIDBlock,
        "core.PID_Discrete": PIDDiscreteBlock,
        "core.Power": _wrap(library.Power),
        "core.PyTorch": PyTorchBlock,
        "core.TensorFlow": TensorFlowBlock,
        "core.MLP": _wrap(library.MLP),
        "core.Product": _wrap_reducer(library.Product),
        "core.ProductOfElements": _wrap(library.ProductOfElements),
        "core.Pulse": _wrap(library.Pulse),
        "core.PythonScript": PythonScriptBlock,
        "core.PyTwin": PyTwinBlock,
        "core.Quantizer": _wrap(library.Quantizer),
        "core.Ramp": _wrap(library.Ramp),
        "core.RandomNumber": RandomNumberBlock,
        "core.RateLimiter": _wrap_discrete(library.RateLimiter),
        "core.Reciprocal": _wrap(library.Reciprocal),
        # "core.ReferenceSubmodel": implemented in model_interface.py
        "core.Relay": _wrap(library.Relay),
        # replicator
        "core.RigidBody": _wrap(library.RigidBody),
        "core.Ros2Publisher": Ros2PublisherBlock,
        "core.Ros2Subscriber": Ros2SubscriberBlock,
        "core.Saturate": _wrap(library.Saturate),
        "core.Sawtooth": _wrap(library.Sawtooth),
        "core.ScalarBroadcast": _wrap(library.ScalarBroadcast),
        "core.SignalDatatypeConversion": _wrap(library.SignalDatatypeConversion),
        "core.SineWave": _wrap(library.Sine),
        "core.SINDy": _wrap(library.Sindy),
        "core.Slice": _wrap(library.Slice),
        "core.SquareRoot": _wrap(library.SquareRoot),
        "core.Stack": StackBlock,
        "core.StateMachine": StateMachineBlock,
        "core.StateSpace": StateSpaceBlock,
        "core.Step": _wrap(library.Step),
        "core.Stop": _wrap(library.Stop),
        "core.SumOfElements": _wrap(library.SumOfElements),
        "core.TransferFunction": TransferFunctionBlock,
        "core.TransferFunctionDiscrete": TransferFunctionDiscreteBlock,
        "core.Trigonometric": _wrap(library.Trigonometric),
        "core.UnitDelay": _wrap_discrete(library.UnitDelay),
        "core.VideoSink": VideoSinkBlock,
        "core.VideoSource": _wrap(library.VideoSource),
        "core.WhiteNoise": WhiteNoiseBlock,
        "core.ZeroOrderHold": _wrap_discrete(library.ZeroOrderHold),
        # Acausal electrical blocks
        "acausal.electrical.Battery": _wrap(electrical.Battery),
        "acausal.electrical.Capacitor": _wrap(electrical.Capacitor),
        "acausal.electrical.Ground": _wrap(electrical.Ground),
        "acausal.electrical.Resistor": _wrap(electrical.Resistor),
        "acausal.electrical.Inductor": _wrap(electrical.Inductor),
        "acausal.electrical.CurrentSensor": _wrap(electrical.CurrentSensor),
        "acausal.electrical.VoltageSensor": _wrap(electrical.VoltageSensor),
        "acausal.electrical.VoltageSource": _wrap(electrical.VoltageSource),
        "acausal.electrical.CurrentSource": _wrap(electrical.CurrentSource),
        "acausal.electrical.IdealMotor": _wrap(electrical.IdealMotor),
        "acausal.electrical.BLDC": _wrap(electrical.BLDC),
        # Acausal rotational blocks
        "acausal.rotational.TorqueSource": _wrap(rotational.TorqueSource),
        "acausal.rotational.SpeedSource": _wrap(rotational.SpeedSource),
        "acausal.rotational.Damper": _wrap(rotational.Damper),
        "acausal.rotational.FixedAngle": _wrap(rotational.FixedAngle),
        "acausal.rotational.Friction": _wrap(rotational.Friction),
        "acausal.rotational.IdealGear": _wrap(rotational.IdealGear),
        "acausal.rotational.IdealPlanetary": _wrap(rotational.IdealPlanetary),
        "acausal.rotational.IdealWheel": _wrap(rotational.IdealWheel),
        "acausal.rotational.Inertia": _wrap(rotational.Inertia),
        "acausal.rotational.Spring": _wrap(rotational.Spring),
        "acausal.rotational.TorqueSensor": _wrap(rotational.TorqueSensor),
        "acausal.rotational.MotionSensor": _wrap(rotational.MotionSensor),
        # Acausal translational blocks
        "acausal.translational.ForceSource": _wrap(translational.ForceSource),
        "acausal.translational.SpeedSource": _wrap(translational.SpeedSource),
        "acausal.translational.Damper": _wrap(translational.Damper),
        "acausal.translational.FixedPosition": _wrap(translational.FixedPosition),
        "acausal.translational.Friction": _wrap(translational.Friction),
        "acausal.translational.Mass": _wrap(translational.Mass),
        "acausal.translational.Spring": _wrap(translational.Spring),
        "acausal.translational.ForceSensor": _wrap(translational.ForceSensor),
        "acausal.translational.MotionSensor": _wrap(translational.MotionSensor),
        # Acausal thermal blocks
        "acausal.thermal.HeatCapacitor": _wrap(thermal.HeatCapacitor),
        "acausal.thermal.HeatflowSensor": _wrap(thermal.HeatflowSensor),
        "acausal.thermal.HeatflowSource": _wrap(thermal.HeatflowSource),
        "acausal.thermal.TemperatureSensor": _wrap(thermal.TemperatureSensor),
        "acausal.thermal.TemperatureSource": _wrap(thermal.TemperatureSource),
        "acausal.thermal.ThermalInsulator": _wrap(thermal.ThermalInsulator),
        "acausal.thermal.ThermalRadiation": _wrap(thermal.ThermalRadiation),
        # Quanser blocks
        "quanser.QubeServoModel": _wrap(library.QubeServoModel),
        "quanser.QuanserHAL": _wrap_discrete(library.QuanserHAL),
    }
)


def get_block_fcn(node_type: str):
    return _fcn_map[node_type]
