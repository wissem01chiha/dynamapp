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

"""MuJoCo (MJX) LeafSystem implementation. Experimental."""

from abc import abstractmethod
from contextlib import redirect_stdout
import inspect
from io import StringIO
import traceback
import typing as T
import copy

import jax
import jax.numpy as jnp
import numpy as np

from collimator.backend import io_callback, dispatcher
from collimator.backend.typing import Array
from collimator.framework import LeafSystem
from collimator.framework.error import (
    CollimatorError,
    PythonScriptError,
)
from collimator.framework.parameter import Parameter
from collimator.framework.state import LeafState
from collimator.framework.system_base import parameters
from collimator.lazy_loader import LazyLoader
from collimator.logging import logdata, logger


if T.TYPE_CHECKING:
    import mujoco
    from mujoco import mjx, Renderer
    from mujoco.mjx._src import math as mjx_math
    import mujoco.mjx._src.types as mjx_types
    from mujoco.mjx import Data as mjxData

else:
    mujoco = LazyLoader("mujoco", globals(), "mujoco")
    mjx = LazyLoader("mjx", globals(), "mujoco.mjx")
    mjx_math = LazyLoader("mjx_math", globals(), "mujoco.mjx._src.math")
    mjx_types = LazyLoader("types", globals(), "mujoco.mjx._src.types")
    mjxData = LazyLoader("Data", globals(), "mujoco.mjx.Data")


# Disable invalid pylint errors (mujoco type hints are not great)
# pylint: disable=no-member

# FIXME: custom_output_scripts serialization will not work since these are
# not block parameters but port parameters.


def position_derivatives(jnt_typs, qpos, qvel):
    """Get derivatives of the generalised position coordinates"""
    qpos_dot, qi, vi = [], 0, 0
    for jnt_typ in jnt_typs:
        if jnt_typ == mjx_types.JointType.FREE:
            trans_der = qvel[vi : vi + 3]
            quat = qpos[qi + 3 : qi + 7]
            omega = qvel[vi + 3 : vi + 6]
            quat_der = 0.5 * mjx_math.quat_mul(quat, jnp.insert(omega, 0, 0.0))
            qpos_dot.append(jnp.concatenate([trans_der, quat_der]))
            qi, vi = qi + 7, vi + 6
        elif jnt_typ == mjx_types.JointType.BALL:
            quat = qpos[qi : qi + 4]
            omega = qvel[vi : vi + 3]
            quat_der = 0.5 * mjx_math.quat_mul(quat, jnp.insert(omega, 0, 0.0))
            qpos_dot.append(quat_der)
            qi, vi = qi + 4, vi + 3
        elif jnt_typ in (mjx_types.JointType.HINGE, mjx_types.JointType.SLIDE):
            trans_der = qvel[vi]
            qpos_dot.append(trans_der[None])
            qi, vi = qi + 1, vi + 1
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")

    return jnp.concatenate(qpos_dot) if qpos_dot else jnp.empty((0,))


class MuJoCoBase(LeafSystem):
    """Base class for the MJX and standard MuJoCo blocks.

    Subclasses must only implement a few abstract methods for compute."""

    @parameters(
        static=[
            "use_mjx",
            "file_name",
            "dt",
            "key_frame_0",
            "qpos_0",
            "qvel_0",
            "act_0",
            "enable_sensor_data",
            "enable_video_output",
            "video_size",
            "enable_mocap_pos",
            "vHIL",
            "vHIL_dt",
        ]
    )
    def __init__(
        self,
        use_mjx: bool,
        file_name: str,
        dt: float = None,
        key_frame_0: int | str = None,
        qpos_0: Array = None,
        qvel_0: Array = None,
        act_0: Array = None,
        enable_sensor_data=False,
        enable_video_output=False,
        video_size: tuple[int, int] = None,
        enable_mocap_pos=False,
        custom_output_scripts: dict[str, str] = None,
        vHIL=False,
        vHIL_dt=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not file_name:
            raise CollimatorError("XML file name must be specified", system=self)

        self.use_mjx = use_mjx
        if use_mjx and dispatcher.active_backend != "jax":
            raise CollimatorError(
                "The MJX block requires the JAX backend to be active.",
                system=self,
            )

        self._model = mujoco.MjModel.from_xml_path(file_name)
        self._data = mujoco.MjData(self._model)
        self._renderer: T.Optional["Renderer"] = None

        if dt is None and not use_mjx:
            logger.warning(
                "The 'dt' parameter is not set. The model will use the default timestep "
                "specified in the XML file or fallback to 0.01.",
                **logdata(block=self),
            )

        dt = dt or self._model.opt.timestep or 0.01
        self._model.opt.timestep = dt

        self.nq, self.nv, self.na, self.nu = (
            self._model.nq,
            self._model.nv,
            self._model.na,
            self._model.nu,
        )

        self.nstate = self.nq + self.nv + self.na

        if key_frame_0 is not None:
            key = (
                self._model.keyframe(str(key_frame_0)).id
                if str(key_frame_0) == key_frame_0
                else int(key_frame_0)
            )
            mujoco.mj_resetDataKeyframe(self._model, self._data, key=key)
            self.qpos_0 = self._data.qpos
            self.qvel_0 = self._data.qvel
            self.act_0 = self._data.act
        else:
            if self._model.nkey > 0:
                logger.warning(
                    "The model has keyframes, but 'key_frame_0' is not set. "
                    "The model will be initialized with all zeros. You might want to "
                    f"set the keyframe to 0 or {repr(self._model.keyframe(0).name)}.",
                    **logdata(block=self),
                )

            if qpos_0 is None:
                self.qpos_0 = jnp.zeros(self.nq)

            if qvel_0 is None:
                self.qvel_0 = jnp.zeros(self.nv)

            if act_0 is None:
                self.act_0 = jnp.zeros(self.na)

        self.qpos_start, self.qpos_end = 0, self.nq
        self.qvel_start, self.qvel_end = self.nq, self.nq + self.nv
        self.act_start, self.act_end = self.nq + self.nv, self.nq + self.nv + self.na

        self.sensor_names = [
            self._model.sensor(i).name for i in range(self._model.nsensor)
        ]
        self.sensor_dim = self._model.sensor_dim
        self.pure_callback_sensordata_result_type: jax.ShapeDtypeStruct = None
        self._video_default: np.ndarray = None
        self.enable_mocap_pos = enable_mocap_pos
        # self._custom_cache_indices: list[int] = []

        # Output some useful debugging information, copy/pasteable by users.
        self.ctrl_0 = self._data.ctrl
        as_param = Parameter(value=self.ctrl_0)
        ctrl_expr, _ = as_param.value_as_api_param()
        logger.info(
            "Initialized MuJoCo from model file '%s' with dt=%s, %s generalized positions, "
            "%s generalized velocities, and %s actuators. The default control "
            "input vector (nu=%s) is:\n%s",
            file_name,
            dt,
            self.nq,
            self.nv,
            self.na,
            self.nu,
            ctrl_expr,
            **logdata(block=self),
        )

        self.declare_input_port("control")

        if enable_mocap_pos:
            self.declare_input_port("mocap_pos")

    @abstractmethod
    def _qpos(self, state): ...

    @abstractmethod
    def _qvel(self, state): ...

    @abstractmethod
    def _act(self, state): ...

    @abstractmethod
    def _output_video(self, time, state, *inputs, **parameters): ...

    def render(self, data):
        if self._renderer is None:
            raise RuntimeError("Renderer not initialized.")
        self._renderer.update_scene(data)
        return self._renderer.render()

    @abstractmethod
    def normalize_qpos_quat(self, qpos):
        """
        Normalize the quaternion components of the generalized position coordinates.
        """

    def _declare_video_output_port(self, video_size: tuple[int, int] = None):
        # We don't need to specify the video FPS here, as the output cb
        # will be called by the connected VideoSink.
        if video_size is not None:
            height, width = int(video_size[0]), int(video_size[1])
            self._renderer = mujoco.Renderer(self._model, width=width, height=height)
        else:
            self._renderer = mujoco.Renderer(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._video_default = self.render(self._data)

        # If mocap_pos is enabled, mjx will not see that, because mjx does not support
        # mocap_pos. So we only use mocap_pos for video rendering and custom outputs.
        video_needs_inputs = self.enable_mocap_pos and self.use_mjx

        self.declare_output_port(
            self._output_video,
            default_value=self._video_default,
            requires_inputs=video_needs_inputs,
            name="video",
        )

    def _pure_callback_update_viewer(self, ctrl):
        if self.viewer.is_running:
            self.viewer.sync()
            self.data_vhil.ctrl[:] = ctrl
            mujoco.mj_step(self.model_vhil, self.data_vhil)
        return jnp.array(0.0)

    def _update_viewer(self, time, state, *inputs, **parameters):
        ctrl = inputs[0]
        return jax.pure_callback(
            self._pure_callback_update_viewer,
            self.pure_callback_update_result_type,
            ctrl,
        )

    def _declare_vhil_fake_output_port(self, vHIL_dt: float):
        # Do we need a separate set of model/data?
        self.model_vhil = copy.deepcopy(self._model)
        self.data_vhil = mujoco.MjData(self.model_vhil)

        # set dt of the mujoco model to vHIL_dt
        self.model_vhil.opt.timestep = vHIL_dt

        # update data to initial values
        self.data_vhil.qpos[:] = self.qpos_0
        self.data_vhil.qvel[:] = self.qvel_0
        self.data_vhil.act[:] = self.act_0

        self.viewer = mujoco.viewer.launch_passive(self.model_vhil, self.data_vhil)
        self.viewer.sync()

        fake_port_value = jnp.array(0.0)

        # declare a fake output port to get a periodic update callback
        # FIXME maybe this should be io_callback since it's quite explicitly I/O
        self.declare_output_port(
            self._update_viewer,
            default_value=fake_port_value,
            requires_inputs=True,
            period=vHIL_dt,
            offset=0.0,
        )

        self.pure_callback_update_result_type = jax.ShapeDtypeStruct(
            fake_port_value.shape, fake_port_value.dtype
        )

    def _mj_forward(self, time, qpos, qvel, act, ctrl=None):
        data = self._data
        data.time = time
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.act[:] = act
        if ctrl is not None:
            data.ctrl[:] = ctrl
        mujoco.mj_forward(self._model, data)
        return data

    def _pure_callback_sensordata(self, time, qpos, qvel, act, ctrl):
        data = self._mj_forward(time, qpos, qvel, act, ctrl)
        return data.sensordata

    def _output_sensor_data(self, time, state, *inputs, **parameters):
        qpos = self._qpos(state)
        qvel = self._qvel(state)
        act = self._act(state)
        ctrl = inputs[0] if inputs else None

        qpos = self.normalize_qpos_quat(qpos)

        return jax.pure_callback(
            self._pure_callback_sensordata,
            self.pure_callback_sensordata_result_type,
            time,
            qpos,
            qvel,
            act,
            ctrl,
        )

    def _declare_sensor_data_port(self, dt: float | None = None):
        sensordata_0 = self._sensordata()
        self.pure_callback_sensordata_result_type = jax.ShapeDtypeStruct(
            sensordata_0.shape, sensordata_0.dtype
        )

        logger.info(
            "MuJoCo block will output sensor data from the following sensors: %s",
            self.sensor_names,
            **logdata(block=self),
        )

        self.declare_output_port(
            self._output_sensor_data,
            default_value=sensordata_0,
            offset=dt,
            period=dt,
            name="sensor_data",
            # FIXME prereqs?
            # requires_inputs=True,
            # prerequisites_of_calc=[self._step_cache_index],
        )

    def _add_custom_output(
        self,
        name: str,
        script: str | T.Callable,
        dt: float | None,
        requires_inputs: bool,
    ):
        """
        Add a custom output port to the block, defined by a python script.

        The script should define a function that takes the arguments `control`, `model`,
        and `data`, and returns the output value. The function must be named `fn`.

        Args:
            name (str): The name of the output port.
            script (str): The python script defining the output function.
            requires_inputs (bool): Whether the output function requires inputs.
        """

        def execute(fn, **kwargs):
            """Wrap custom script execution"""
            stdout_buffer = StringIO()
            exc = None
            retval = None

            with redirect_stdout(stdout_buffer):
                try:
                    retval = fn(**kwargs)
                except BaseException as e:  # pylint: disable=broad-except
                    exc = e

            stdout = stdout_buffer.getvalue()
            if stdout:
                if stdout.endswith("\n"):
                    stdout = stdout[:-1]
                logger.info(stdout, **logdata(block=self))

            if exc:
                bt = exc.__traceback__
                bt_str = "".join(traceback.format_exception(exc.__class__, exc, bt)[2:])
                raise PythonScriptError(
                    f"Error executing custom output script for port '{name}':\n{bt_str}",
                    system=self,
                    port_direction="out",
                    port_name=name,
                ) from exc

            if retval is None:
                return None
            return np.asarray(retval)

        # Minimal env with model & data
        env = {
            "__main__": {},
            "model": self._model,
            "data": self._data,
        }

        if not isinstance(script, T.Callable):
            compiled = compile(script, f"<{self.name}.{name}>", "exec")
            execute(lambda: exec(compiled, env, env))

            if "fn" in env:
                fn = env["fn"]
            else:
                raise PythonScriptError(
                    f"Script for custom output port '{name}' does not define "
                    "a valid lambda or 'fn' callback function.",
                    system=self,
                    port_direction="out",
                    port_name=name,
                )
        else:
            fn = script

        sig = inspect.signature(fn)
        has_data = "data" in sig.parameters
        has_model = "model" in sig.parameters
        has_mjx_data = "mjx_data" in sig.parameters
        has_mjx_model = "mjx_model" in sig.parameters
        has_control = "control" in sig.parameters
        has_mocap_pos = "mocap_pos" in sig.parameters

        def fn_kwargs():
            kwargs = {}
            if has_model:
                kwargs["model"] = self._model
            if has_data:
                kwargs["data"] = self._data
            if has_mjx_model:
                kwargs["mjx_model"] = self.model
            if has_mjx_data:
                kwargs["mjx_data"] = self._data

            # FIXME: we likely don't need to pass control or mocap_pos to the custom
            # scripts. Control should not be necessary by design, since passing it
            # makes the block potentially feedthrough, and it is something calculated
            # outside of mujoco. As for mocap data, it is only used for rendering, and
            # can be another input to the external controller block.
            if has_control:
                kwargs["control"] = self._data.ctrl
            if has_mocap_pos and self.enable_mocap_pos:
                # FIXME? unclear why the shape changes from (1,n) to (n,)
                kwargs["mocap_pos"] = np.array(self._data.mocap_pos).flatten()
            return kwargs

        default_value = execute(fn, **fn_kwargs())

        def cb(time, state, *inputs, **parameters):
            def _cb(inputs):
                kwargs = fn_kwargs()
                if has_control:
                    kwargs["control"] = inputs[0]
                if has_mocap_pos and self.enable_mocap_pos:
                    kwargs["mocap_pos"] = inputs[1]
                return execute(fn, **kwargs)

            return io_callback(_cb, default_value, inputs)

        self.declare_output_port(
            cb,
            name=name,
            default_value=default_value,
            requires_inputs=requires_inputs,
            period=dt,
            offset=0,
        )

    def _declare_custom_output_ports(
        self,
        custom_output_scripts: dict[str, str],
        dt: float,
        requires_inputs: bool = True,
    ):
        # EXTREMELY EXPERIMENTAL:
        # A list of custom python scripts to execute, that define each a new
        # "dynamic" output port. In the UI/JSON, it is a property of the port, not
        # of the block.
        # Python context: control, model, data
        for name, script in (custom_output_scripts or {}).items():
            try:
                self._add_custom_output(name, script, dt, requires_inputs)
            except Exception as e:
                if isinstance(e, CollimatorError):
                    raise
                raise PythonScriptError(
                    f"Error setting up custom output script for port '{name}': {e}",
                    system=self,
                    port_direction="out",
                    port_name=name,
                ) from e


class MJX(MuJoCoBase):
    """
    A system that wraps a MuJoCo model and provides a continuous-time ODE LeafSystem.
    Currently only supports a single body system.

    Input ports:
        (0) The control input vector `control`.

    Output ports:
        (0) The generalized position coordinates `qpos`.
        (1) The generalized velocity coordinates `qvel`.
        (2) The actuator coordinates `act`.
        (3) The sensor data `sensor_data` (if enabled).
        (4) The video output `video` as RGB frames of shape (H,W,3) (if enabled).
        (5) A fake output port, present only if `vHIL=True` and outputs Array(0.0).
        (6+) Custom output ports, defined with user-specified python scripts.

    Parameters:
        file_name (str):
            The path to the MuJoCo XML model file.

        dt (float, optional):
            If None, collimator's internal solver will be used and this block can
            be considered as a continuous block. If set, the model will be run in
            a discrete mode with the specified timestep, using MJX's solver, more like
            Co-Simulation. In that case, it might be favorable to set `use_mjx=False`.

        key_frame_0 (int|str, optional):
            The keyframe to initialize the model from.

        qpos_0 (Array, optional):
            The initial generalized position coordinates.

        qvel_0 (Array, optional):
            The initial generalized velocity coordinates.

        act_0 (Array, optional):
            The initial actuator coordinates.

        enable_sensor_data (bool, optional):
            Whether to output the sensor data to an optional port named 'sensor_data'.

        enable_video_output (bool, optional):
            Whether to output the rendered video frames to an optional port named 'video'.

        video_size (tuple[int, int], optional):
            The size of the video output frames as a (H,W) tuple.

        enable_mocap_pos (bool, optional):
            Whether to enable the mocap_pos input port for motion capture tracking.

        vHIL (bool, optional):
            Whether to run in virtual hardware-in-the-loop mode.

        vHIL_dt (float, optional):
            The timestep for the virtual hardware-in-the-loop mode.

    Notes:
    (i) `_model` and `_data` refer to MuJoCo's `mjModel` and `mjData` objects
    respectively. `model` and `data` are the corresponding MJX objects.
    (ii) While `sensordata` output is supported as a pure callback to MuJoCo since MJX
    has not yet implemented this aspect. This can be expensive.
    """

    @parameters(
        static=[
            "file_name",
            "dt",
            "key_frame_0",
            "qpos_0",
            "qvel_0",
            "act_0",
            "enable_sensor_data",
            "enable_video_output",
            "video_size",
            "enable_mocap_pos",
            "vHIL",
            "vHIL_dt",
        ]
    )
    def __init__(
        self,
        file_name: str,
        dt: float = None,
        key_frame_0: int | str = None,
        qpos_0: Array = None,
        qvel_0: Array = None,
        act_0: Array = None,
        enable_sensor_data=False,
        enable_video_output=False,
        video_size: tuple[int, int] = None,
        enable_mocap_pos=False,
        custom_output_scripts: dict[str, str] = None,
        vHIL=False,
        vHIL_dt=0.01,
        **kwargs,
    ):
        super().__init__(
            use_mjx=True,
            file_name=file_name,
            dt=dt,
            key_frame_0=key_frame_0,
            qpos_0=qpos_0,
            qvel_0=qvel_0,
            act_0=act_0,
            enable_sensor_data=enable_sensor_data,
            enable_video_output=enable_video_output,
            video_size=video_size,
            enable_mocap_pos=enable_mocap_pos,
            custom_output_scripts=custom_output_scripts,
            vHIL=vHIL,
            vHIL_dt=vHIL_dt,
            **kwargs,
        )

        try:
            self.model = mjx.put_model(self._model)
            self.data = mjx.put_data(self._model, self._data)
        except NotImplementedError as e:
            logger.error(
                "This robot model uses features not implemented in MJX. "
                "Please try the MuJoCo block instead (toggle use_mjx to false), "
                "or modify the MJCF file.",
                **logdata(block=self, exception=f"{type(e).__name__}: {str(e)}"),
            )
            raise e

        if dt is None or dt == 0:
            self.dt = None
            logger.info(
                "MuJoCo MJX block is running in continuous mode and will use "
                "Collimator's solver.",
                **logdata(block=self),
            )

            state_0 = jnp.concatenate([self.qpos_0, self.qvel_0, self.act_0])
            self.declare_continuous_state(ode=self._ode, default_value=state_0)

        else:
            self.dt = dt
            known_solver_names = {
                mujoco.mjtSolver.mjSOL_CG: "Conjugate Gradient",
                mujoco.mjtSolver.mjSOL_NEWTON: "Newton",
            }
            logger.info(
                "MuJoCo MJX block is running in discrete mode with dt=%s, "
                "this will use MJX's solver '%s' and not Collimator's solver.",
                dt,
                known_solver_names.get(
                    self.model.opt.solver,
                    str(self.model.opt.solver),
                ),
                **logdata(block=self),
            )

            callback_index = self.declare_cache(
                self._step_cache_cb,
                default_value=self.data,
                period=dt,
                offset=0.0,
                requires_inputs=True,
            )
            self.mjx_data_cache_index = self.callbacks[callback_index].cache_index

        self.declare_output_port(
            self._output_qpos,
            default_value=self.qpos_0,
            requires_inputs=False,
            name="qpos",
        )

        self.declare_output_port(
            self._output_qvel,
            default_value=self.qvel_0,
            requires_inputs=False,
            name="qvel",
        )

        self.declare_output_port(
            self._output_act,
            default_value=self.act_0,
            requires_inputs=False,
            name="act",
        )

        if enable_sensor_data:
            logger.warning(
                "Sensor data output with MJX might be very slow. Consider switching "
                "to the non-MJX MuJoCo block for better performance.",
                **logdata(block=self),
            )
            self._declare_sensor_data_port(self.dt)

        if enable_video_output:
            logger.warning(
                "Video output with MJX might be very slow. Consider switching "
                "to the non-MJX MuJoCo block for better performance.",
                **logdata(block=self),
            )
            self._declare_video_output_port(video_size)

        if vHIL:
            self._declare_vhil_fake_output_port(vHIL_dt)

        self._declare_custom_output_ports(custom_output_scripts, self.dt)

    def _cached_data(self, state: LeafState) -> mjxData:
        return state.cache[self.mjx_data_cache_index]

    def _qpos(self, state: LeafState):
        if self.dt is not None:
            return self._cached_data(state).qpos

        return state.continuous_state[self.qpos_start : self.qpos_end]

    def _qvel(self, state: LeafState):
        if self.dt is not None:
            return self._cached_data(state).qvel

        return state.continuous_state[self.qvel_start : self.qvel_end]

    def _act(self, state: LeafState):
        if self.dt is not None:
            return self._cached_data(state).act

        return state.continuous_state[self.act_start : self.act_end]

    def _ode(self, time, state, *inputs, **parameters):
        # Implementation of the ODE when running model in continuous mode, with
        # collimator's internal solver.

        qpos = self._qpos(state)
        qvel = self._qvel(state)
        act = self._act(state)

        ctrl = inputs[0]

        # FIXME: Should we normalize the quaternions here to avoid numerical drift?
        # qpos = self.normalize_qpos_quat(qpos)

        model, data = self.model, self.data
        data = data.replace(time=time, qpos=qpos, qvel=qvel, act=act, ctrl=ctrl)

        data = mjx.forward(model, data)

        qvel_dot = data.qacc
        qpos_dot = position_derivatives(model.jnt_type, qpos, qvel)
        act_dot = data.act_dot

        state_dot = jnp.concatenate([qpos_dot, qvel_dot, act_dot])

        return state_dot

    def _step_cache_cb(self, time, state: LeafState, *inputs, **parameters):
        # Implementation of the ODE when running model in discrete mode, with
        # MJX's solver. This is like the non-MJX variant or an FMU.

        # TODO: try a version wrapped with io_callback to compare
        # compilation times. Splitting the compute graph between collimator
        # and mjx could bring improvements, but quite obviously at the cost
        # of any usefulness of mjx over mujoco (autodiff, vmap, ...).

        ctrl = inputs[0]

        data = self._cached_data(state)
        data = data.replace(time=time, ctrl=ctrl)
        data = mjx.step(self.model, data)

        return data

    def _output_qpos(self, time, state, *inputs, **parameters):
        qpos = self._qpos(state)
        qpos_normalized_quats = self.normalize_qpos_quat(qpos)
        return qpos_normalized_quats

    def _output_qvel(self, time, state, *inputs, **parameters):
        return self._qvel(state)

    def _output_act(self, time, state, *inputs, **parameters):
        return self._act(state)

    def _mj_forward(self, time, qpos, qvel, act, ctrl=None):
        data = self._data
        data.time = time
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.act[:] = act
        if ctrl is not None:
            data.ctrl[:] = ctrl
        mujoco.mj_forward(self._model, data)
        return data

    def _pure_callback_sensordata(self, time, qpos, qvel, act, ctrl):
        data = self._mj_forward(time, qpos, qvel, act, ctrl)
        return data.sensordata

    def _output_sensor_data(self, time, state, *inputs, **parameters):
        qpos = self._qpos(state)
        qvel = self._qvel(state)
        act = self._act(state)
        ctrl = inputs[0] if inputs else None

        qpos = self.normalize_qpos_quat(qpos)

        return jax.pure_callback(
            self._pure_callback_sensordata,
            self.pure_callback_sensordata_result_type,
            time,
            qpos,
            qvel,
            act,
            ctrl,
        )

    def normalize_qpos_quat(self, qpos):
        """
        Normalize the quaternion components of the generalized position coordinates.
        """
        qpos_normalized, qi = [], 0

        for jnt_typ in self.model.jnt_type:
            if jnt_typ == mjx_types.JointType.FREE:
                trans = qpos[qi : qi + 3]
                quat = qpos[qi + 3 : qi + 7]
                norm_quat = mjx_math.normalize(quat)
                qpos_normalized.append(jnp.concatenate([trans, norm_quat]))
                qi = qi + 7
            elif jnt_typ == mjx_types.JointType.BALL:
                quat = qpos[qi : qi + 4]
                norm_quat = mjx_math.normalize(quat)
                qpos_normalized.append(norm_quat)
                qi = qi + 4
            elif jnt_typ in (mjx_types.JointType.HINGE, mjx_types.JointType.SLIDE):
                trans = qpos[qi]
                qpos_normalized.append(trans[None])
                qi = qi + 1
            else:
                raise RuntimeError(f"unrecognized joint type: {jnt_typ}")

        return jnp.concatenate(qpos_normalized) if qpos_normalized else jnp.empty((0,))

    def _update_viewer(self, time, state, *inputs, **parameters):
        ctrl = inputs[0]
        return jax.pure_callback(
            self._pure_callback_update_viewer,
            self.pure_callback_update_result_type,
            ctrl,
        )

    def _pure_callback_update_viewer(self, ctrl):
        if self.viewer.is_running:
            self.viewer.sync()
            self.data_vhil.ctrl[:] = ctrl
            mujoco.mj_step(self.model_vhil, self.data_vhil)
        return jnp.array(0.0)

    def _output_video_discrete(self, time, state, *inputs, **parameters):
        def _discrete_cb(state):
            mjx_data = self._cached_data(state)
            data = mjx.get_data(self._model, mjx_data)
            if self.enable_mocap_pos:
                data.mocap_pos[:] = inputs[1]
            return self.render(data)

        return io_callback(_discrete_cb, self._video_default, time, state)

    def _output_video(self, time, state, *inputs, **parameters):
        if self.dt is not None:
            return self._output_video_discrete(time, state, *inputs, **parameters)

        def _continuous_cb(time, qpos, qvel, act, inputs):
            data = self._mj_forward(time, qpos, qvel, act)
            if self.enable_mocap_pos:
                data.mocap_pos[:] = inputs[1]
            return self.render(data)

        qpos = self._qpos(state)
        qvel = self._qvel(state)
        act = self._act(state)

        qpos = self.normalize_qpos_quat(qpos)

        return io_callback(
            _continuous_cb, self._video_default, time, qpos, qvel, act, inputs
        )


class MuJoCo(MuJoCoBase):
    """MuJoCo implementation without MJX.

    Refer to MJX for the main docs.

    Unlike the MJX variant of the block, this version uses the solver provided
    by mujoco itself and the physics are fully handled by mujoco. This behaves
    like a Co-Simulation environment.

    This variant may be used to speed up compilation times or in situations where
    full JAX is not available or practical.
    """

    def __init__(
        self,
        file_name: str,
        dt: float = 0.01,
        key_frame_0: int | str = None,
        qpos_0: Array = None,
        qvel_0: Array = None,
        act_0: Array = None,
        enable_sensor_data=False,
        enable_video_output=False,
        video_size: tuple[int, int] = None,
        enable_mocap_pos=False,
        custom_output_scripts: dict[str, str] = None,
        vHIL=False,
        vHIL_dt=0.01,
        **kwargs,
    ):
        super().__init__(
            use_mjx=False,
            file_name=file_name,
            dt=dt,
            key_frame_0=key_frame_0,
            qpos_0=qpos_0,
            qvel_0=qvel_0,
            act_0=act_0,
            enable_sensor_data=enable_sensor_data,
            enable_video_output=enable_video_output,
            video_size=video_size,
            enable_mocap_pos=enable_mocap_pos,
            custom_output_scripts=custom_output_scripts,
            vHIL=vHIL,
            vHIL_dt=vHIL_dt,
            **kwargs,
        )

        # This output cb implements the call to _step and is the reference callback
        # that all other outputs will depend on.
        def _qpos_cb(time, state, *inputs, **parameters):
            def cb(inputs):
                self._data.ctrl = inputs[0]
                if enable_mocap_pos:
                    self._data.mocap_pos[:] = inputs[1]
                mujoco.mj_step(self._model, self._data)
                qpos_normalized_quats = self.normalize_qpos_quat(self._qpos())
                return qpos_normalized_quats

            return io_callback(cb, self.qpos_0, inputs)

        self._step_cache_index = self.declare_output_port(
            _qpos_cb,
            default_value=self.qpos_0,
            requires_inputs=True,
            offset=dt,
            period=dt,
            name="qpos",
        )

        def _qvel_cb(time, state, *inputs, **parameters):
            return io_callback(self._qvel, self.qvel_0)

        self.declare_output_port(
            _qvel_cb,
            default_value=self.qvel_0,
            requires_inputs=True,
            offset=dt,
            period=dt,
            name="qvel",
            prerequisites_of_calc=[self._step_cache_index],
        )

        def _act_cb(time, state, *inputs, **parameters):
            return io_callback(self._act, self.act_0)

        self.declare_output_port(
            _act_cb,
            default_value=self.act_0,
            requires_inputs=True,
            offset=dt,
            period=dt,
            name="act",
            prerequisites_of_calc=[self._step_cache_index],
        )

        if enable_sensor_data:
            self._declare_sensor_data_port(dt)
        if enable_video_output:
            self._declare_video_output_port(video_size)
        if vHIL:
            self._declare_vhil_fake_output_port(vHIL_dt)
        self._declare_custom_output_ports(
            custom_output_scripts, dt, requires_inputs=False
        )

    # def post_simulation_finalize(self) -> None:
    #     # FIXME this should not be here but I had a "too many files opened" error
    #     self._model = None
    #     self._data = None
    #     return super().post_simulation_finalize()

    def _qpos(self, state=None):
        return self._data.qpos

    def _qvel(self, state=None):
        return self._data.qvel

    def _act(self, state=None):
        return self._data.act

    def _sensordata(self):
        return self._data.sensordata

    def normalize_qpos_quat(self, qpos):
        mujoco.mj_normalizeQuat(self._model, qpos)
        return qpos

    def render(self, data=None):
        if data is None:
            data = self._data
        return super().render(data)

    def _output_video(self, time, state, *inputs, **parameters):
        def cb(time):
            return self.render(self._data)

        return io_callback(
            cb,
            self._video_default,
            time,
        )
