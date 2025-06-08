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

from collections import namedtuple
from contextlib import redirect_stderr, redirect_stdout
import functools
from io import StringIO
import logging
import types
from typing import TYPE_CHECKING, Any, List, Mapping

import jax
import jax.numpy as jnp

from ..framework import LeafSystem, LeafState, parameters
from ..logging import logdata, logger
from ..framework.error import (
    BlockInitializationError,
    ErrorCollector,
    PythonScriptError,
    PythonScriptTimeNotSupportedError,
)
from ..backend import io_callback, jit, numpy_api as cnp


if TYPE_CHECKING:
    from ..backend.typing import Array, DTypeLike
    from ..framework.context import ContextBase


__all__ = [
    "CustomJaxBlock",
    "CustomPythonBlock",
]


def _caused_by_nameerror(e):
    if e is None:
        return None
    if isinstance(e, NameError):
        return e
    return _caused_by_nameerror(e.__cause__)


def _default_exec(
    code: str | types.CodeType,
    env: dict[str, Any],
    logger: logging.Logger,
    inputs: dict[str, jax.Array] = None,
    return_vars: list[str] = None,
    return_dtypes: list[DTypeLike] = None,
    system: LeafSystem = None,
    code_name: str = "step_code",
):
    """
    `env` is a mutable state this is required because the python script block
    keeps the state across simulation steps.
    """

    stdout_buffer = StringIO()
    strerr_buffer = StringIO()
    exception = None

    if inputs is not None:
        env.update(inputs)

    with redirect_stderr(strerr_buffer):
        with redirect_stdout(stdout_buffer):
            try:
                exec(code, env, env)
            except BaseException as e:
                exception = e

    stdout = stdout_buffer.getvalue()
    if stdout:
        stdout = stdout[:-1] if stdout[-1] == "\n" else stdout
        logger.info(stdout, **logdata(block=system))
    stderr = strerr_buffer.getvalue()
    if stderr:
        stderr = stderr[:-1] if stderr[-1] == "\n" else stderr
        logger.warning(stderr, **logdata(block=system))

    if exception is not None:
        name_error = _caused_by_nameerror(exception)
        if name_error and name_error.name == "time":
            raise PythonScriptTimeNotSupportedError(system=system) from exception
        raise PythonScriptError(system=system) from exception

    if return_vars is None:
        return

    for var in return_vars:
        if var not in env:
            raise PythonScriptError(
                f"Variable '{var}' not defined in {code_name}.", system=system
            )

    return [
        cnp.asarray(env[var], dtype=dtype)
        for var, dtype in zip(return_vars, return_dtypes)
    ]


def _filter_non_traceable(globals_dict):
    """
    since we have to use locals as globals, our locals gets really polluted with all
    kinds of stuff. trying to retain this whole thing results in jax errors. so this
    functions job is to split the global env in two: one for "dynamic" arrays that jax
    can trace and another for "static" data that cannot be traced and will be stored
    as a block attribute (e.g. functions, classes, modules, etc).
    """
    dynamic_globals = {}
    static_globals = {}
    for key, value in globals_dict.items():
        try:
            # Test pytree conversion but don't actually convert.  If the global was
            # declared like `x = [0, 1]` we don't want to convert this to
            # `x = [array(0), array(1)]`.  If the global does need to be converted
            # because its value will be used to initialize an output port, this
            # will be done during output initialization.
            jax.tree_util.tree_map(jnp.asarray, value)

            # Store the original value if the value had a pytree structure.
            dynamic_globals[key] = value
        except TypeError:
            # Feel free to remove below debug log if too noisy
            if not isinstance(value, types.ModuleType) and key not in ["__builtins__"]:
                logger.debug(
                    'Filtering non-traceable global "%s" (%s).', key, type(value)
                )
            # The value is not traceable, so store it as a static block attribute.
            static_globals[key] = value

    return dynamic_globals, static_globals


class CustomJaxBlock(LeafSystem):
    """JAX implementation of the PythonScript block.

    A few important notes and changes/limitations to this JAX implementation:
    - For this block all code must be written using the JAX-supported subset of Python:
        * Numerical operations should use `jax.numpy = jnp` instead of `numpy = np`
        * Standard control flow is not supported (if/else, for, while, etc.). Instead
            use `lax.cond`, `lax.fori_loop`, `lax.while_loop`, etc.
            https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#structured-control-flow-primitives
            Where possible, NumPy-style operations like `jnp.where` or `jnp.select` should
            be preferred to lax control flow primitives.
        * Functions must be pure and arrays treated as immutable.
            https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates
        Provided these assumptions hold, the code can be JIT compiled, differentiated,
        run on GPU, etc.
    - Variable scoping: the `init_code` and `step_code` are executed in the same scope,
        so variables declared in the `init_code` will be available in the `step_code`
        and can be modified in that scope. Internally, everything declared in
        `init_code` is treated as a single state-like cache entry.
        However, variables declared in the `step_code` will NOT persist between
        evaluations. Users should think of `step_code` as a normal Python function
        where locally declared variables will disappear on leaving the scope.
    - Persistent variables (outputs and anything declared in `init_code`) must have
        static shapes and dtypes. This means that you cannot declare `x = 0.0` in
        `init_code` and then later assign `x = jnp.zeros(4)` in `step_code`.

    These changes mean that many older PythonScript blocks may not be backwards compatible.

    Input ports:
        Variable number of input ports, one for each input variable declared in `inputs`.
        The order of the input ports is the same as the order of the input variables.

    Output ports:
        Variable number of output ports, one for each output variable declared in `outputs`.
        The order of the output ports is the same as the order of the output variables.

    Parameters:
        dt (float): The discrete time step of the block, or None if the block is
            in agnostic time mode.
        init_script (str): A string containing Python code that will be executed
            once when the block is initialized. This code can be used to declare
            persistent variables that will be available in the `step_code`.
        user_statements (str): A string containing Python code that will be executed
            once per time step (or per output port evaluation, in agnostic mode).
            This code can use the persistent variables declared in `init_script` and
            the block inputs.
        finalize_script (str): A string containing Python code that will be executed
            once when the block is finalized. This code can use the persistent
            variables declared in `init_script` and the block inputs. (Currently not
            yet supported).
        accelerate_with_jax (bool): If True, the block will be JIT compiled. If False,
            the block will be executed in pure Python.  This parameter exists for
            compatibility with UI options; when creating pure Python blocks from code
            (e.g. for testing), explicitly create the CustomPythonBlock class.
        time_mode (str): One of "discrete" or "agnostic". If "discrete", the block
            step code will be evaluated at peridodic intervals specified by "dt".
            If "agnostic", the block step code will be evaluated once per output
            port evaluation, and the block will not have a discrete time step.
        inputs (List[str]): A list of input variable names. The order of the input
            ports is the same as the order of the input variables.
        outputs (Mapping[str, Tuple[DTypeLike, ShapeLike]]): A dictionary mapping
            output variable names to a tuple of dtype and shape. The order of the
            output ports is the same as the order of the output variables.
        parameters (Mapping[str, Array]): A dictionary mapping parameter names to
            values. Parameters are treated as immutable and cannot be modified in
            the step code. Parameters can be arrays or scalars, but must have static
            shapes and dtypes in order to support JIT compilation.
    """

    @parameters(
        static=[
            "dt",
            "init_script",
            "user_statements",
            "finalize_script",
            "accelerate_with_jax",
            "time_mode",
        ]
    )
    def __init__(
        self,
        dt: float = None,
        init_script: str = "",
        user_statements: str = "",
        finalize_script: str = "",  # presently ignored for JAX block
        accelerate_with_jax: bool = True,
        time_mode: str = "discrete",  # [discrete, agnostic]
        inputs: List[str] = None,  # [name]
        outputs: List[str] = None,
        parameters: Mapping[str, Array] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        parameters = parameters or {}

        if time_mode not in ["discrete", "agnostic"]:
            raise BlockInitializationError(
                f"Invalid time mode '{time_mode}' for PythonScript block", system=self
            )

        if time_mode == "discrete" and dt is None:
            raise BlockInitializationError(
                "When in discrete time mode, dt is required for block", system=self
            )

        self.time_mode = time_mode

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []

        self.dt = dt

        # Note: 'optimize' level could be lowered in debug mode
        self.init_code = compile(
            init_script, filename="<init>", mode="exec", optimize=2
        )
        self.step_code = compile(
            user_statements, filename="<step>", mode="exec", optimize=2
        )

        if finalize_script != "" and not isinstance(self, CustomPythonBlock):
            raise BlockInitializationError(
                f"PythonScript block '{self.name_path_str}' has finalize_script "
                "but this is not supported at the moment.",
                system=self,
                parameter_name="finalize_script",
            )

        # Declare parameters
        for param_name, value in parameters.items():
            if isinstance(value, list):
                value = cnp.asarray(value)
            as_array = isinstance(value, cnp.ndarray) or cnp.isscalar(value)
            self.declare_dynamic_parameter(param_name, value, as_array=as_array)

        # Run the init_script
        persistent_env = self.exec_init()

        # Declare an input port for each of the input variables
        self.input_names = inputs
        for name in inputs:
            self.declare_input_port(name)

        # Declare a cache component for each of the output variables
        self._create_cache_type(outputs)

        if time_mode == "discrete":
            self._configure_discrete(dt, outputs, persistent_env)
        else:
            self._configure_agnostic(outputs, persistent_env)

    def initialize(
        self,
        dt: float = None,
        init_script: str = "",
        user_statements: str = "",
        finalize_script: str = "",  # presently ignored for JAX block
        accelerate_with_jax: bool = True,
        time_mode: str = "discrete",  # [discrete, agnostic]
        **parameters,
    ):
        pass

    def _initialize_outputs(self, outputs, persistent_env):
        default_outputs = {name: None for name in outputs}

        for name in outputs:
            # If the initial value is set explicitly in the init script,
            # override the default value.  We don't need to do this for
            # agnostic configuration since the outputs will be calculated
            # every evaluation anyway.
            if name in persistent_env:
                value = cnp.asarray(persistent_env[name])
                default_outputs[name] = value

                # Also update the persistent environment so that the data types
                # are consistent with the state.
                persistent_env[name] = value

            # Otherwise throw an error, since we don't know what the initial values
            # should be, or even what shape/dtype they should have.
            else:
                msg = (
                    f"Output variable '{name}' not explicitly initialized in "
                    "init_script for PythonScript block in 'Discrete' time mode. "
                    "Either initialize the variable as an array with the correct "
                    "shape and dtype, or make the block time mode 'Agnostic'."
                )
                raise PythonScriptError(message=msg, system=self)

        return self.CacheType(
            persistent_env=persistent_env,
            **default_outputs,
        )

    def _configure_discrete(self, dt, outputs, persistent_env):
        default_values = self._initialize_outputs(outputs, persistent_env)

        # The step function acts as a periodic update that will update all components
        # of the discrete state.
        self.step_callback_index = self.declare_cache(
            self.exec_step,
            period=dt,
            offset=dt,
            requires_inputs=True,
            default_value=default_values,
        )

        cache = self.callbacks[self.step_callback_index]

        # Get the index into the state cache (different in general from the index
        # into the callback list, since not all callbacks are cached).
        self.step_cache_index = cache.cache_index

        def _make_callback(o_port_name):
            def _output(time, state, *inputs, **parameters):
                return getattr(state.cache[self.step_cache_index], o_port_name)

            return _output

        # Declare output ports for each state variable
        for o_port_name in outputs:
            self.declare_output_port(
                _make_callback(o_port_name),
                name=o_port_name,
                prerequisites_of_calc=[cache.ticket],
                requires_inputs=False,
                period=dt,
                offset=0.0,
            )

    def _configure_agnostic(self, outputs, persistent_env):
        # Create a callback to evaluate the step code and extract the
        # output. Note that this is inefficient since the step code will
        # be evaluated once _for each output port_, but it's the only way
        # to do this unless (until) we implement some variety of block
        # or function pre-ordering.
        def _make_callback(o_port_name):
            def _output(time, state, *inputs, **parameters):
                xd = self.exec_step(time, state, *inputs, **parameters)
                return getattr(xd, o_port_name)

            return _output

        # Declare output ports for each state variable
        for o_port_name in outputs:
            self.declare_output_port(
                _make_callback(o_port_name),
                name=o_port_name,
                requires_inputs=True,
            )

        # This callback doesn't need to do anything since it's never
        # actually called - the cache here just stores the initial environment
        # and the output ports are evaluated directly.  This should be changed
        # to avoid re-evaluation with multiple output ports once we can do full
        # function ordering.
        def _cache_callback(time, state, *inputs, **parameters):
            return state.cache[self.step_cache_index]

        # Since this is the return type for `exec_step` we have to declare all
        # the output ports as entries in the namedtuple, even though those values
        # won't actually be cached in "agnostic" time mode.  This is just so that
        # both "discrete" and "agnostic" modes can share the same code.
        default_values = self.CacheType(
            persistent_env=persistent_env,
            **{o_port_name: None for o_port_name in outputs},
        )
        self.step_callback_index = self.declare_cache(
            _cache_callback,
            default_value=default_values,
            requires_inputs=False,
            prerequisites_of_calc=[inport.ticket for inport in self.input_ports],
        )

        cache = self.callbacks[self.step_callback_index]
        self.step_cache_index = cache.cache_index

    def _create_cache_type(self, outputs):
        # Store the output ports as a name for type inference and casting
        self.output_names = outputs

        # Also store the dictionary of local environment variables as a cache entry
        # This is the only persistent state of the system (besides outputs) - anything
        # declared in the "step" function will be forgotten at the end of the step

        self.CacheType = namedtuple("CacheType", self.output_names + ["persistent_env"])

    @property
    def local_env_base(self):
        # Define a starting point for the local code execution environment.
        # we have to inclide __main__ so that the code behaves like a module.
        # this allows for code like this:
        #   imports ...
        #   a = 1
        #   def f(b):
        #       return a+b
        #   out_0 = f(2)
        #
        # without getting a 'a not defined' error.
        return {
            "__main__": {},
        }

    def exec_init(self) -> dict[str, Array]:
        # Before executing the step code, we have to build up the local environment.
        # This includes specified modules, python block user defined parameters.

        default_parameters = {
            name: param.get() for name, param in self.dynamic_parameters.items()
        }

        local_env = {
            **self.local_env_base,
            **default_parameters,
        }

        # similar to above where we included __main__ so the code behaves as a module,
        # here we have to pass the local_env with __main__ as 1] globals, since that
        # is what allow the code to be executed as a module. 2] local since that is where
        # the new bindings will be written, that we need to retain since the code in step_code
        # may depend on these bindings.
        exec(self.init_code, local_env, local_env)

        # persistent_env contains bindings for parameters and for values from init_script
        persistent_env, static_env = _filter_non_traceable(local_env)

        # Since this is called during block initialization and not any JIT-compiled code,
        # we can safely store any untraceable variables as block attributes.  For example,
        # this may contain custom functions, classes, etc.
        self.static_env = static_env

        return persistent_env

    def exec_step(self, time: float, state: LeafState, *inputs, **parameters):
        # Before executing the step code, we have to build up the local environment.
        # This includes the persistent variables (anything declared in `init_code`),
        # time, block inputs, user-defined parameters, and specified modules.

        # Retrieve the variables declared in `init_code` from the discrete state
        full_env = state.cache[self.step_cache_index]
        persistent_env = full_env.persistent_env

        # Inputs are in order of port declaration, so they match `self.input_names`
        input_env = dict(zip(self.input_names, inputs))

        # Create a dictionary of all the information that the step function will need
        base_copy = self.local_env_base.copy()
        local_env = {
            **self.static_env,
            **base_copy,
            **persistent_env,
            **input_env,
        }

        # Execute the step code in the local environment
        exec(self.step_code, local_env, local_env)

        # Updated state variables are stored in the local environment
        xd = {name: local_env[name] for name in self.output_names}

        # Store the persistent variables in the corresponding discrete state
        xd["persistent_env"] = {key: local_env[key] for key in persistent_env}

        # Make sure the results have a consistent data type
        for name in self.output_names:
            xd[name] = cnp.asarray(local_env[name])

            # Also make sure the value stored in the persistent environment
            # has the same data type
            if name in persistent_env:
                xd["persistent_env"][name] = xd[name]

        return self.CacheType(**xd)

    def check_types(
        self,
        context: ContextBase,
        error_collector: ErrorCollector = None,
    ):
        """Test-compile the init and step code to check for errors."""
        try:
            jit(self.wrap_callback(self.exec_step))(context)
        except BaseException as exc:
            with ErrorCollector.context(error_collector):
                name_error = _caused_by_nameerror(exc)
                if name_error and name_error.name == "time":
                    raise PythonScriptTimeNotSupportedError(system=self) from exc
                raise PythonScriptError(system=self) from exc


class CustomPythonBlock(CustomJaxBlock):
    """Container for arbitrary user-defined Python code.

    Implemented to support legacy PythonScript blocks.

    Not traceable (no JIT compilation or autodiff). The internal implementation
    and behavior of this block differs vastly from the JAX-compatible block as
    this block stores state directly within the Python instance. Objects
    and modules can be kept as discrete state.

    Note that in "agnostic" mode, the step code will be evaluated _once per
    output port evaluation_. Because locally defined environment variables
    (in the init script) are preserved between evaluations, any mutation of
    these variables will be preserved. This can lead to unexpected behavior
    and should be avoided. Stateful behavior should be implemented using
    discrete state variables instead.
    """

    __exec_fn = _default_exec

    def __init__(
        self,
        dt: float = None,
        init_script: str = "",
        user_statements: str = "",
        finalize_script: str = "",  # presently ignored
        inputs: List[str] = None,  # [name]
        outputs: List[str] = None,
        accelerate_with_jax: bool = False,
        time_mode: str = "discrete",
        parameters: Mapping[str, Array] = None,
        **kwargs,
    ):
        self._static_data_initialized = False
        self._parameters = parameters or {}
        self._persistent_env = {}

        # Will populate return type information during static initialization
        self.result_shape_dtypes = None
        self.return_dtypes = None

        super().__init__(
            dt=dt,
            init_script=init_script,
            user_statements=user_statements,
            finalize_script=finalize_script,
            inputs=inputs,
            outputs=outputs,
            accelerate_with_jax=accelerate_with_jax,
            time_mode=time_mode,
            parameters=self._parameters,
            **kwargs,
        )

        if time_mode == "agnostic" and cnp.active_backend == "jax":
            logger.warning(
                "System %s is in agnostic time mode but is not traced with JAX. Be "
                "advised that the step code will be evaluated once per output port "
                "evaluation. Any mutation of the local environment should be strictly "
                "avoided as it will likely lead to unexpected behavior.",
                self.name_path_str,
            )

    def initialize(self, **kwargs):
        pass

    @property
    def has_feedthrough_side_effects(self) -> bool:
        # See explanation in `SystemBase.has_ode_side_effects`.
        return self.time_mode == "agnostic"

    @staticmethod
    def set_exec_fn(exec_fn: callable):
        CustomPythonBlock.__exec_fn = exec_fn

    @property
    def local_env_base(self):
        # Define a starting point for the local code execution environment.
        return {
            "__main__": {},
            "true": True,
            "false": False,
        }

    def exec_init(self) -> None:
        default_parameters = {
            name: param.get() for name, param in self.dynamic_parameters.items()
        }

        local_env = {
            **self.local_env_base,
            **self._parameters,
            **default_parameters,
        }

        exec_fn = functools.partial(
            CustomPythonBlock.__exec_fn,
            code=self.init_code,
            env=local_env,
            logger=logger,
            system=self,
            code_name="init_script",
        )

        try:
            io_callback(exec_fn, None)
        except KeyboardInterrupt as e:
            logger.error(
                "Python block '%s' init script execution was interrupted.", self.name
            )
            raise PythonScriptError(
                message="Python block init script execution was interrupted.",
                system=self,
            ) from e
        except PythonScriptError as e:
            logger.error("%s: exec_init failed.", self.name)
            raise e
        except BaseException as e:
            logger.error("%s: exec_init failed.", self.name)
            raise PythonScriptError(system=self) from e
        self._persistent_env = local_env

        return None

    def exec_step(self, time, state, *inputs, **parameters):
        if not self._static_data_initialized:
            raise PythonScriptError(
                "Trying to execute step code before static data has been initialized",
                system=self,
            )
        logger.debug(
            "Executing step for %s with state=%s, inputs=%s",
            self.name,
            state,
            inputs,
        )

        # Inputs are in order of port declaration, so they match `self.input_names`
        input_env = dict(zip(self.input_names, inputs))

        base_copy = self.local_env_base.copy()
        local_env = {
            **base_copy,
            **self._persistent_env,
        }

        exec_fn = functools.partial(
            CustomPythonBlock.__exec_fn,
            code=self.step_code,
            env=local_env,
            logger=logger,
            return_vars=self.output_names,
            return_dtypes=self.return_dtypes,
            system=self,
            code_name="step_code",
        )

        try:
            return_vars = io_callback(
                exec_fn, self.result_shape_dtypes, inputs=input_env
            )
        except KeyboardInterrupt:
            logger.error(
                "Python block '%s' init script execution was interrupted.", self.name
            )
            raise
        except NameError as e:
            err_msg = (
                f"Python block '{self.name}' step script execution failed with a NameError on"
                + f" missing variable '{e.name}'."
                + " All names used in this script should be declared in the init script."
                + f" The execution environment contains the following names: {', '.join(list(local_env.keys()))}"
            )
            logger.error(err_msg)
            logger.error("NameError: %s", e)
            raise PythonScriptError(system=self) from e
        except PythonScriptError as e:
            logger.error("%s: exec_step failed.", self.name)
            raise e
        except BaseException as e:
            logger.error("%s: exec_step failed.", self.name)
            raise PythonScriptError(system=self) from e

        # Keep local env for next step but only if defined in init_script
        # NOTE: If this restriction turns out to be counterproductive, we can
        # remove it and remove the NameError handling above as well. The thinking
        # here is that this could help avoiding stuff like `if time == 0: x = 0`
        # See https://collimator.atlassian.net/browse/WC-98
        self._persistent_env = {
            key: local_env[key] for key in self._persistent_env if key in local_env
        }

        # Updated state variables are stored in the local environment
        xd = {name: return_vars[i] for i, name in enumerate(self.output_names)}

        return self.CacheType(persistent_env=None, **xd)

    def _initialize_outputs(self, outputs, _persistent_env):
        # Override the base implemenetation since `persistent_env` will be None
        # in this case. Instead, pass the class attribute where the environment
        # is actually maintained.
        default_outputs = {name: None for name in outputs}
        default_values = self.CacheType(
            persistent_env=self._persistent_env,
            **default_outputs,
        )
        default_values = super()._initialize_outputs(outputs, self._persistent_env)
        default_outputs = default_values._asdict()
        self._persistent_env = default_outputs.pop("persistent_env")

        # Determine return data types
        self._initialize_result_shape_dtypes(
            [default_outputs[output] for output in outputs]
        )

        return self.CacheType(
            persistent_env=None,
            **default_outputs,
        )

    def _initialize_result_shape_dtypes(self, outputs):
        self.result_shape_dtypes = []
        self.return_dtypes = []
        for value in outputs:
            self.result_shape_dtypes.append(
                jax.ShapeDtypeStruct(value.shape, value.dtype)
            )
            self.return_dtypes.append(value.dtype)

    def initialize_static_data(self, context):
        # If in agnostic mode, call the step function once to determine the
        # data types and then store those in result_shape_dtype and return_dtypes.
        context = LeafSystem.initialize_static_data(self, context)

        if self.result_shape_dtypes is not None:
            # These data types are already known (block is in discrete mode)
            self._static_data_initialized = True
            return context

        inputs = self.collect_inputs(context)
        input_env = dict(zip(self.input_names, inputs))

        base_copy = self.local_env_base.copy()
        local_env = {
            **base_copy,
            **self._persistent_env,
        }

        # Will not do any type conversion
        return_dtypes = [None for _ in self.output_names]

        exec_fn = functools.partial(
            CustomPythonBlock.__exec_fn,
            self.step_code,
            local_env,
            logger,
            return_vars=self.output_names,
            return_dtypes=return_dtypes,
            system=self,
            code_name="step_code",
        )

        return_vars = exec_fn(inputs=input_env)

        self._initialize_result_shape_dtypes(return_vars)

        self._static_data_initialized = True

        return context

    def check_types(
        self,
        context: ContextBase,
        error_collector=None,
    ):
        pass
