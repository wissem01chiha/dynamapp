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

"""Implementations of "primitive" blocks as LeafSystems.

Through out **kwargs is meant to pass these common args to super.__init__():
    system_id
    name
    ui_id
"""

from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING, NamedTuple
from functools import partial
from collections import namedtuple
from enum import IntEnum
import jax.numpy as jnp
import numpy as np
from ..logging import logger
from ..framework.error import BlockParameterError, ErrorCollector
from ..framework.event import LeafEventCollection, ZeroCrossingEvent
from ..framework.system_base import UpstreamEvalError
from ..framework import (
    LeafSystem,
    ShapeMismatchError,
    DtypeMismatchError,
    DependencyTicket,
    parameters,
)
from ..backend import cond, numpy_api as cnp
from ..lazy_loader import LazyLoader
from .generic import SourceBlock, FeedthroughBlock, ReduceBlock
from .linear_system import derivative_filter

if TYPE_CHECKING:
    import equinox as eqx
    from ..framework.port import OutputPort
    from ..backend.typing import Array
else:
    eqx = LazyLoader("eqx", globals(), "equinox")

__all__ = [
    "Abs",
    "Adder",
    "Arithmetic",
    "Chirp",
    "Clock",
    "Comparator",
    "Constant",
    "CrossProduct",
    "DeadZone",
    "Demultiplexer",
    "DerivativeDiscrete",
    "DiscreteInitializer",
    "DiscreteClock",
    "DotProduct",
    "EdgeDetection",
    "Exponent",
    "FilterDiscrete",
    "Gain",
    "IfThenElse",
    "Integrator",
    "IntegratorDiscrete",
    "IOPort",
    "Logarithm",
    "LogicalOperator",
    "LogicalReduce",
    "LookupTable1d",
    "LookupTable2d",
    "MatrixConcatenation",
    "MatrixInversion",
    "MatrixMultiplication",
    "MatrixTransposition",
    "MinMax",
    "Multiplexer",
    "Offset",
    "PIDDiscrete",
    "Power",
    "Product",
    "ProductOfElements",
    "Pulse",
    "Quantizer",
    "Ramp",
    "RateLimiter",
    "Reciprocal",
    "Saturate",
    "Sawtooth",
    "ScalarBroadcast",
    "Sine",
    "Slice",
    "SquareRoot",
    "Stack",
    "Step",
    "Stop",
    "SumOfElements",
    "Trigonometric",
    "UnitDelay",
    "ZeroOrderHold",
]


def check_state_type(
    sys: LeafSystem,
    inp_data: Array,
    state_data: Array,
    error_collector: ErrorCollector = None,
) -> None:
    """Check that the state type of a block matches the type of an input port."""
    inp_data = cnp.asarray(inp_data)
    state_data = cnp.asarray(state_data)

    with ErrorCollector.context(error_collector):
        if inp_data.shape != state_data.shape:
            logger.debug(
                "System %s shape mismatch, %s != %s",
                sys.system_id,
                inp_data.shape,
                state_data.shape,
            )
            raise ShapeMismatchError(
                system=sys,
                expected_shape=state_data.shape,
                actual_shape=inp_data.shape,
            )
        if inp_data.dtype != state_data.dtype:
            logger.debug(
                "System %s dtype mismatch, %s != %s",
                sys.system_id,
                inp_data.dtype,
                state_data.dtype,
            )
            raise DtypeMismatchError(
                system=sys,
                expected_dtype=state_data.dtype,
                actual_dtype=inp_data.dtype,
            )


def is_discontinuity(port: OutputPort) -> bool:
    """Does this signal represent a discontinuous input to an ODE?"""
    signal_is_continuous = port.tracker.depends_on(
        [DependencyTicket.time, DependencyTicket.xc]
    )
    port_is_ode_rhs = port.tracker.is_prerequisite_of([DependencyTicket.xcdot])
    return signal_is_continuous and port_is_ode_rhs


class Abs(FeedthroughBlock):
    """Output the absolute value of the input signal.

    Input ports:
        None

    Output ports:
        (0) The absolute value of the input signal.

    Events:
        An event is triggered when the output changes from positive to negative
        or vice versa.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(cnp.abs, *args, **kwargs)

    def _zero_crossing(self, _time, _state, u):
        return u

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output is
        # fed to an ODE.
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(self._zero_crossing, direction="crosses_zero")

        return super().initialize_static_data(context)


class Adder(ReduceBlock):
    """Computes the sum/difference of the input.

    The add/subtract operation can be switched by setting the `operators` parameter.
    For example, a 3-input block specified as `Adder(3, operators="+-+")` would add
    the first and third inputs and subtract the second input.

    Input ports:
        (0..n_in-1) The input signals to add/subtract.

    Output ports:
        (0) The sum/difference of the input signals.
    """

    @parameters(static=["operators"])
    def __init__(self, n_in, *args, operators=None, **kwargs):
        super().__init__(n_in, None, *args, **kwargs)

    def initialize(self, operators):
        if operators is not None and any(char not in {"+", "-"} for char in operators):
            raise BlockParameterError(
                message=f"Adder block {self.name} has invalid operators {operators}. Can only contain '+' and '-'",
                system=self,
                parameter_name="operators",
            )

        if operators is None:
            _func = sum
        else:
            signs = [1 if op == "+" else -1 for op in operators]

            def _func(inputs):
                signed_inputs = [s * u for (s, u) in zip(signs, inputs)]
                return sum(signed_inputs)

        self.replace_op(_func)


class Arithmetic(ReduceBlock):
    """Performs addition, subtraction, multiplication, and division on the input.

    The arithmetic operation is determined by setting the `operators` parameter.
    For example, a 4-input block specified as `Arithmetic(4, operators="+-*/")` would:
        - Add the first input,
        - Subtract the second input,
        - Multiply the third input,
        - Divide by the fourth input.

    Input ports:
        (0..n_in-1) The input signals for the specified arithmetic operations.

    Output ports:
        (0) The result of the specified arithmetic operations on the input signals.

    """

    @parameters(static=["operators"])
    def __init__(self, n_in, *args, operators=None, **kwargs):
        super().__init__(n_in, None, *args, **kwargs)

    def initialize(self, operators):
        if operators is not None and any(
            char not in {"+", "-", "*", "/"} for char in operators
        ):
            raise BlockParameterError(
                message=f"Arithmetic block {self.name} has invalid operators {operators}. Can only contain '+', '-', '*', '/'.",
                system=self,
                parameter_name="operators",
            )

        ops = {
            "+": cnp.add,
            "-": cnp.subtract,
            "/": cnp.divide,
            "*": cnp.multiply,
        }

        def evaluate_expression(operands, operators):
            operands = operands[:]
            operators = operators[:]

            # Handle multiplication and division
            while "*" in operators or "/" in operators:
                for op in ("*", "/"):
                    if op in operators:
                        index = operators.index(op)
                        result = ops[op](operands[index], operands[index + 1])
                        operands = operands[:index] + [result] + operands[index + 2 :]
                        operators = operators[:index] + operators[index + 1 :]

            # Handle addition and subtraction
            while "+" in operators or "-" in operators:
                for op in ("-", "+"):
                    if op in operators:
                        index = operators.index(op)
                        result = ops[op](operands[index], operands[index + 1])
                        operands = operands[:index] + [result] + operands[index + 2 :]
                        operators = operators[:index] + operators[index + 1 :]

            return operands[0]

        def _func(inputs):
            inputs = list(inputs)
            if operators[0] == "/":
                inputs[0] = 1.0 / inputs[0]
            if operators[0] == "-":
                inputs[0] = -inputs[0]
            ops = operators[1:]
            return evaluate_expression(inputs, ops)

        self.replace_op(_func)


class Chirp(SourceBlock):
    """Produces a signal like the linear method of

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html

    Parameters:
        f0 (float): Frequency (Hz) at time t=phi.
        f1 (float): Frequency (Hz) at time t=stop_time.
        stop_time (float): Time to end the signal (seconds).
        phi (float): Phase offset (radians).

    Input ports:
        None

    Output ports:
        (0) The chirp signal.
    """

    @parameters(dynamic=["f0", "f1", "stop_time", "phi"])
    def __init__(self, f0, f1, stop_time, phi=0.0, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, f0, f1, stop_time, phi):
        # FIXME: There's an extra factor of 2 that doesn't seem like it's in the SciPy version.
        def _func(time, stop_time, f0, f1, phi):
            f = f0 + (f1 - f0) * time / (2 * stop_time)
            return cnp.cos(f * time + phi)

        self.replace_op(_func)


class Clock(SourceBlock):
    """Source block returning simulation time.

    Input ports:
        None

    Output ports:
        (0) The simulation time.

    Parameters:
        dtype:
            The data type of the output signal.  The default is "None", which will
            default to the current default floating point precision
    """

    def __init__(self, dtype=None, **kwargs):
        super().__init__(lambda t: cnp.array(t, dtype=dtype), **kwargs)


class Comparator(LeafSystem):
    """Compare two signals using typical relational operators.

    When using == and != operators, the block uses tolerances to determine if the
    expression is true or false.

    Parameters:
        operator: one of ("==", "!=", ">=", ">", ">=", "<")
        atol: the absolute tolerance value used with "==" or "!="
        rtol: the relative tolerance value used with "==" or "!="

    Input Ports:
        (0) The left side operand
        (1) The right side operand

    Output Ports:
        (0) The result of the comparison (boolean signal)

    Events:
        An event is triggered when the output changes from true to false or vice versa.
    """

    @parameters(static=["operator", "atol", "rtol"])
    def __init__(self, atol=1e-5, rtol=1e-8, operator=None, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port()
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, atol, rtol, operator):
        func_lookup = {
            ">": cnp.greater,
            ">=": cnp.greater_equal,
            "<": cnp.less,
            "<=": cnp.less_equal,
            "==": self._equal,
            "!=": self._ne,
        }

        if operator not in func_lookup:
            message = (
                f"Comparator block '{self.name}' has invalid selection "
                + f"'{operator}' for parameter 'operator'. Valid options: "
                + ",".join([k for k in func_lookup.keys()])
            )
            raise BlockParameterError(
                message=message, system=self, parameter_name="operator"
            )

        self.rtol = rtol
        self.atol = atol

        compare = func_lookup[operator]

        def _compute_output(_time, _state, *inputs, **_params):
            return compare(*inputs)

        self.configure_output_port(
            self._output_port_idx,
            _compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )
        self.evt_direction = self._process_operator(operator)

    def _equal(self, x, y):
        if cnp.issubdtype(x.dtype, cnp.floating):
            return cnp.isclose(x, y, self.rtol, self.atol)
        return x == y

    def _ne(self, x, y):
        if cnp.issubdtype(x.dtype, cnp.floating):
            return cnp.logical_not(cnp.isclose(x, y, self.rtol, self.atol))
        return x != y

    def _zero_crossing(self, _time, _state, *inputs, **_params):
        return inputs[0] - inputs[1]

    def _process_operator(self, operator):
        if operator in ["<", "<="]:
            return "positive_then_non_positive"
        if operator in [">", ">="]:
            return "negative_then_non_negative"
        return "crosses_zero"

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output is
        # fed to an ODE.
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(
                self._zero_crossing, direction=self.evt_direction
            )

        return super().initialize_static_data(context)


class Constant(LeafSystem):
    """A source block that emits a constant value.

    Parameters:
        value: The constant value of the block.

    Input ports:
        None

    Output ports:
        (0) The constant value.
    """

    @parameters(dynamic=["value"])
    def __init__(self, value, *args, **kwargs):
        super().__init__(**kwargs)
        self._output_port_idx = self.declare_output_port(name="out_0")

    def initialize(self, value):
        def _func(time, state, *inputs, **parameters):
            return parameters["value"]

        self.configure_output_port(
            self._output_port_idx,
            _func,
            prerequisites_of_calc=[DependencyTicket.nothing],
            requires_inputs=False,
        )


class CrossProduct(ReduceBlock):
    """Compute the cross product between the inputs.

    See NumPy docs for details:
    https://numpy.org/doc/stable/reference/generated/numpy.cross.html

    Input ports:
        (0) The first input vector.
        (1) The second input vector.

    Output ports:
        (0) The cross product of the inputs.
    """

    def __init__(self, *args, **kwargs):
        def _cross(inputs):
            return cnp.cross(*inputs)

        super().__init__(2, _cross, *args, **kwargs)


class DeadZone(FeedthroughBlock):
    """Generates zero output within a specified range.

    Applies the following function:
    ```
             [ input,       input < -half_range
    output = | 0,           -half_range <= input <= half_range
             [ input        input > half_range
    ```

    Parameters:
        half_range: The range of the dead zone.  Must be > 0.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal modified by the dead zone.

    Events:
        An event is triggered when the signal enters or exits the dead zone
        in either direction.
    """

    @parameters(dynamic=["half_range"])
    def __init__(self, half_range=1.0, **kwargs):
        super().__init__(self._dead_zone, **kwargs)
        if half_range <= 0:
            raise BlockParameterError(
                message=f"DeadZone block {self.name} has invalid half_range {half_range}. Must be > 0.",
                system=self,
                parameter_name="half_range",
            )

    def initialize(self, half_range):
        pass

    def _dead_zone(self, x, **params):
        return cnp.where(abs(x) < params["half_range"], x * 0, x)

    def _lower_limit_event_value(self, _time, _state, *inputs, **params):
        (u,) = inputs
        return u + params["half_range"]

    def _upper_limit_event_value(self, _time, _state, *inputs, **params):
        (u,) = inputs
        return u - params["half_range"]

    def initialize_static_data(self, context):
        # Add zero-crossing events so ODE solvers can't try to integrate
        # through a discontinuity.
        if not self.has_zero_crossing_events and (self.output_ports[0]):
            self.declare_zero_crossing(
                self._lower_limit_event_value, direction="crosses_zero"
            )
            self.declare_zero_crossing(
                self._upper_limit_event_value, direction="crosses_zero"
            )

        return super().initialize_static_data(context)


class Demultiplexer(LeafSystem):
    """Split a vector signal into its components.

    Input ports:
        (0) The vector signal to split.

    Output ports:
        (0..n_out-1) The components of the input signal.
    """

    def __init__(self, n_out, **kwargs):
        super().__init__(**kwargs)

        self.declare_input_port()

        # Need a helper function so that the lambda captures the correct value of i
        # and doesn't use something that ends up fixed in scope.
        def _declare_output(i):
            def _compute_output(_time, _state, *inputs, **_params):
                (input_vec,) = inputs
                return input_vec[i]

            self.declare_output_port(
                _compute_output,
                prerequisites_of_calc=[self.input_ports[0].ticket],
            )

        for i in cnp.arange(n_out):
            _declare_output(i)


class DerivativeDiscrete(LeafSystem):
    """Discrete approximation to the derivative of the input signal w.r.t. time.'

    By default the block uses a simple backward difference approximation:
    ```
    y[k] = (u[k] - u[k-1]) / dt
    ```
    However, the block can also be configured to use a recursive filter for a
    better approximation. In this case the filter coefficients are determined
    by the `filter_type` and `filter_coefficient` parameters. The filter is
    a pair of two-element arrays `a` and `b` and the filter equation is:
    ```
    a0*y[k] + a1*y[k-1] = b0*u[k] + b1*u[k-1]
    ```

    Denoting the `filter_coefficient` parameter by `N`, the following filters are
    available:
    - "none": The default, a simple finite difference approximation.
    - "forward": A filtered forward Euler discretization. The filter is:
        `a = [1, (N*dt - 1)]` and `b = [N, -N]`.
    - "backward": A filtered backward Euler discretization. The filter is:
        `a = [(1 + N*dt), -1]` and `b = [N, -N]`.
    - "bilinear": A filtered bilinear transform discretization. The filter is:
        `a = [(2 + N*dt), (-2 + N*dt)]` and `b = [2*N, -2*N]`.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The approximate derivative of the input signal.

    Parameters:
        dt:
            The time step of the discrete approximation.
        filter_type:
            One of "none", "forward", "backward", or "bilinear". This determines the
            type of filter used to approximate the derivative. The default is "none",
            corresponding to a simple backward difference approximation.
        filter_coefficient:
            The coefficient in the filter (`N` in the equations above). This is only
            used if `filter_type` is not "none". The default is 1.0.
    """

    @parameters(static=["filter_type", "filter_coefficient"])
    def __init__(self, dt, filter_type="none", filter_coefficient=1.0, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self.deriv_output = self.declare_output_port(
            period=dt,
            offset=0.0,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def initialize(self, filter_type="none", filter_coefficient=1.0):
        # Determine the coefficients of the filter, if applicable
        # The filter is a pair of two-element array and the filter
        # equation is:
        # a0*y[k] + a1*y[k-1] = b0*u[k] + b1*u[k-1]
        self.filter = derivative_filter(
            N=filter_coefficient, dt=self.dt, filter_type=filter_type
        )

        self.declare_discrete_state(default_value=None, as_array=False)

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # At t=0 we have no prior information, so the output will
        # be held from its initial value (zero). At t=dt, we have
        # a previous sample, so there is enough information to estimate
        # the derivative.
        self.configure_output_port(
            self.deriv_output,
            self._output,
            period=self.dt,
            offset=self.dt,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def _output(self, _time, state, *inputs, **_params):
        # Compute the filtered derivative estimate
        (u,) = inputs
        b, a = self.filter
        y_prev = state.cache[self.deriv_output]
        u_prev = state.discrete_state
        y = (b[0] * u + b[1] * u_prev - a[1] * y_prev) / a[0]
        return y

    def _update(self, time, state, u, **params):
        # Every dt seconds, update the state to the current values
        return u

    def initialize_static_data(self, context):
        """Infer the size and dtype of the internal states"""
        # If building as part of a subsystem, this may not be fully connected yet.
        # That's fine, as long as it is connected by root context creation time.
        # This probably isn't a good long-term solution:
        #   see https://collimator.atlassian.net/browse/WC-51
        try:
            u = self.eval_input(context)
            self._default_discrete_state = u
            local_context = context[self.system_id].with_discrete_state(u)
            self._default_cache[self.deriv_output] = 0 * u
            local_context = local_context.with_cached_value(self.deriv_output, 0 * u)
            context = context.with_subcontext(self.system_id, local_context)

        except UpstreamEvalError:
            logger.debug(
                "DerivativeDiscrete.initialize_static_data: UpstreamEvalError. "
                "Continuing without default value initialization."
            )
        return super().initialize_static_data(context)


class DiscreteClock(LeafSystem):
    """Source block that produces the time sampled at a fixed rate.

    The block maintains the most recently sampled time as a discrete state, provided
    to the output port during the following interval. Graphically, a discrete clock
    sampled at 100 Hz would have the following time series:

    ```
      x(t)                  ●━
        |                   ┆
    .03 |              ●━━━━○
        |              ┆
    .02 |         ●━━━━○
        |         ┆
    .01 |    ●━━━━○
        |    ┆
      0 ●━━━━○----+----+----+-- t
        0   .01  .02  .03  .04
    ```

    The recorded states are the closed circles, which should be interpreted at index
    `n` as the value seen by all other blocks on the interval `(t[n], t[n+1])`.

    Input ports:
        None

    Output ports:
        (0) The sampled time.

    Parameters:
        dt:
            The sampling period of the clock.
        start_time:
            The simulation time at which the clock starts. Defaults to 0.
    """

    def __init__(self, dt, dtype=None, start_time=0, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype or float
        start_time = cnp.array(start_time, dtype=self.dtype)

        self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
            requires_inputs=False,
            default_value=start_time,
            prerequisites_of_calc=[DependencyTicket.time],
        )

    def _output(self, time, _state, *_inputs, **_params):
        return cnp.array(time, dtype=self.dtype)


class DiscreteInitializer(LeafSystem):
    """Discrete Initializer.

    Outputs True for first discrete step, then outputs False there after.
    Or, outputs False for first discrete step, then outputs True there after.
    Practical for cases where it is necessary to have some signal fed initially
    by some initialization, but then after from else in the model.

    Input ports:
        None

    Output ports:
        (0) The dot product of the inputs.
    """

    @parameters(dynamic=["initial_state"])
    def __init__(self, dt, initial_state=True, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.declare_output_port(self._output)
        self._periodic_update_idx = self.declare_periodic_update()

    def initialize(self, initial_state):
        self.declare_discrete_state(default_value=initial_state, dtype=cnp.bool_)
        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=cnp.inf,
            offset=self.dt,
        )

    def _update(self, time, state, *_inputs, **_params):
        return cnp.logical_not(state.discrete_state)

    def _output(self, _time, state, *_inputs, **_params):
        return state.discrete_state


class DotProduct(ReduceBlock):
    """Compute the dot product between the inputs.

    This block dispatches to `jax.numpy.dot`, so the semantics, broadcasting rules,
    etc. are the same.  See the JAX docs for details:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.dot.html

    Input ports:
        (0) The first input vector.
        (1) The second input vector.

    Output ports:
        (0) The dot product of the inputs.
    """

    def __init__(self, **kwargs):
        super().__init__(2, self._compute_output, **kwargs)

    def _compute_output(self, inputs):
        return cnp.dot(inputs[0], inputs[1])


class EdgeDetection(LeafSystem):
    """Output is true only when the input signal changes in a specified way.

    The block updates at a discrete rate, checking the boolean- or binary-valued input
    signal for changes.  Available edge detection modes are:
        - "rising": Output is true when the input changes from False (0) to True (1).
        - "falling": Output is true when the input changes from True (1) to False (0).
        - "either": Output is true when the input changes in either direction

    Input ports:
        (0) The input signal. Must be boolean or binary-valued.

    Output ports:
        (0) The edge detection output signal. Boolean-valued.

    Parameters:
        dt:
            The sampling period of the block.
        edge_detection:
            One of "rising", "falling", or "either". Determines the type of edge
            detection performed by the block.
        initial_state:
            The initial value of the output signal.
    """

    class DiscreteStateType(NamedTuple):
        prev_input: Array
        output: bool

    @parameters(static=["edge_detection"], dynamic=["initial_state"])
    def __init__(self, dt, edge_detection, initial_state=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.declare_input_port()

        # Declare the periodic update
        self._periodic_update_idx = self.declare_periodic_update()

        # Declare the output port
        self._output_port_idx = self.declare_output_port(
            self._output,
            prerequisites_of_calc=[DependencyTicket.xd, self.input_ports[0].ticket],
            requires_inputs=False,
        )

    def initialize(self, edge_detection, initial_state):
        # Determine the type of edge detection
        _detection_funcs = {
            "rising": self._detect_rising,
            "falling": self._detect_falling,
            "either": self._detect_either,
        }
        if edge_detection not in _detection_funcs:
            raise ValueError(
                f"EdgeDetection block {self.name} has invalid selection "
                f"{edge_detection} for 'edge_detection'"
            )
        self._detect_edge = _detection_funcs[edge_detection]

        # The discrete state will contain the previous input value and the output
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(
                prev_input=initial_state, output=False
            ),
            as_array=False,
        )
        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # Declare the output port
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xd, self.input_ports[0].ticket],
            requires_inputs=False,
        )

    def _update(self, time, state, *inputs, **params):
        # Update the stored previous state
        # and the output as the result of the edge detection function
        (e,) = inputs
        return self.DiscreteStateType(
            prev_input=e,
            output=self._detect_edge(time, state, e, **params),
        )

    def _output(self, _time, state, *_inputs, **_params):
        return state.discrete_state.output

    def _detect_rising(self, _time, state, *inputs, **_params):
        (e,) = inputs
        e_prev = state.discrete_state.prev_input
        e_prev = cnp.array(e_prev)
        e = cnp.array(e)
        not_e_prev = cnp.logical_not(e_prev)
        return cnp.logical_and(not_e_prev, e)

    def _detect_falling(self, _time, state, *inputs, **_params):
        (e,) = inputs
        e_prev = state.discrete_state.prev_input
        e_prev = cnp.array(e_prev)
        e = cnp.array(e)
        not_e = cnp.logical_not(e)
        return cnp.logical_and(e_prev, not_e)

    def _detect_either(self, _time, state, *inputs, **_params):
        (e,) = inputs
        e_prev = state.discrete_state.prev_input
        e_prev = cnp.array(e_prev)
        e = cnp.array(e)
        not_e_prev = cnp.logical_not(e_prev)
        not_e = cnp.logical_not(e)
        rising = cnp.logical_and(not_e_prev, e)
        falling = cnp.logical_and(e_prev, not_e)
        return cnp.logical_or(rising, falling)


class Exponent(FeedthroughBlock):
    """Compute the exponential of the input signal.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The exponential of the input signal.

    Parameters:
        base:
            One of "exp" or "2". Determines the base of the exponential function.
    """

    @parameters(static=["base"])
    def __init__(self, base, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, base):
        func_lookup = {"exp": cnp.exp, "2": cnp.exp2}
        if base not in func_lookup:
            raise BlockParameterError(
                message=f"Exponent block {self.name} has invalid selection {base} for 'base'. Valid selections: "
                + ", ".join([k for k in func_lookup.keys()]),
                parameter_name="base",
            )
        self.replace_op(func_lookup[base])


class FilterDiscrete(LeafSystem):
    """Finite Impulse Response (FIR) filter.

    Similar to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    Note: does not implement the IIR filter.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The filtered signal.

    Parameters:
        b_coefficients:
            Array of filter coefficients.
    """

    @parameters(dynamic=["b_coefficients"])
    def __init__(
        self,
        dt,
        b_coefficients,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, b_coefficients):
        initial_state = cnp.zeros(len(b_coefficients) - 1)
        self.declare_discrete_state(default_value=initial_state)

        self.is_feedthrough = b_coefficients[0] != 0
        prerequisites_of_calc = []
        if self.is_feedthrough:
            prerequisites_of_calc.append(self.input_ports[0].ticket)

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=self.dt,
        )

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=self.dt,
            requires_inputs=self.is_feedthrough,
            prerequisites_of_calc=prerequisites_of_calc,
        )

    def _update(self, _time, state, u, **_parameters):
        xd = state.discrete_state
        return cnp.concatenate([cnp.atleast_1d(u), xd[:-1]])

    def _output(self, time, state, *inputs, **parameters):
        xd = state.discrete_state
        b_coefficients = parameters["b_coefficients"]

        y = cnp.sum(cnp.dot(b_coefficients[1:], xd))

        if self.is_feedthrough:
            (u,) = inputs
            y += u * b_coefficients[0]

        return y


class Gain(FeedthroughBlock):
    """Multiply the input signal by a constant value.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal multiplied by the gain: `y = gain * u`.

    Parameters:
        gain:
            The value to scale the input signal by.
    """

    @parameters(dynamic=["gain"])
    def __init__(self, gain, *args, **kwargs):
        super().__init__(lambda x, gain: gain * x, *args, **kwargs)

    def initialize(self, gain):
        pass


class IfThenElse(LeafSystem):
    """Applies a conditional expression to the input signals.

    Given inputs `pred`, `true_val`, and `false_val`, the block computes:
    ```
    y = true_val if pred else false_val
    ```

    The true and false values may be any arrays, but must have the same
    shape and dtype.

    Input ports:
        (0) The boolean predicate.
        (1) The true value.
        (2) The false value.

    Output ports:
        (0) The result of the conditional expression. Shape and dtype will match
            the true and false values.

    Events:
        An event is triggered when the output changes from true to false or vice versa.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.declare_input_port()  # pred
        self.declare_input_port()  # true_val
        self.declare_input_port()  # false_val

        def _compute_output(_time, _state, *inputs, **_params):
            return cnp.where(inputs[0], inputs[1], inputs[2])

        self.declare_output_port(
            _compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    def _edge_detection(self, _time, _state, *inputs, **_params):
        return cnp.where(inputs[0], 1.0, -1.0)

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output is
        # fed to an ODE.
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(self._edge_detection, direction="crosses_zero")

        return super().initialize_static_data(context)


class Integrator(LeafSystem):
    """Integrate the input signal in time.

    The Integrator block is the main primitive for building continuous-time
    models.  It is a first-order integrator, implementing the following linear
    time-invariant ordinary differential equation for input values `u` and output
    values `y`:
    ```
        ẋ = u
        y = x
    ```
    where `x` is the state of the integrator.  The integrator is initialized
    with the value of the `initial_state` parameter.

    Options:
        Reset: the integrator can be configured to reset its state on an input
            trigger.  The reset value can be either the initial state of the
            integrator or an external value provided by an input port.
        Limits: the integrator can be configured such that the output and state
            are constrained by upper and lower limits.
        Hold: the integrator can be configured to hold integration based on an
            input trigger.

    The Integrator block is also designed to detect "Zeno" behavior, where the
    reset events happen asymptotically closer together.  This is a pathological
    case that can cause numerical issues in the simulation and should typically be
    avoided by introducing some physically realistic hysteresis into the model.
    However, in the event that Zeno behavior is unavoidable, the integrator will
    enter a "Zeno" state where the output is held constant until the trigger
    changes value to False.  See the "bouncing ball" demo for a Zeno example.

    Input ports:
        (0) The input signal.  Must match the shape and dtype of the initial
            continuous state.
        (1) The reset trigger.  Optional, only if `enable_reset` is True.
        (2) The reset value.  Optional, only if `enable_external_reset` is True.
        (3) The hold trigger. Optional, only if 'enable_hold' is True.

    Output ports:
        (0) The continuous state of the integrator.

    Parameters:
        initial_state:
            The initial value of the integrator state.  Can be any array, or even
            a nested structure of arrays, but the data type should be floating-point.
        enable_reset:
            If True, the integrator will reset its state to the initial value
            when the reset trigger is True.  Adds an additional input port for
            the reset trigger.  This signal should be boolean- or binary-valued.
        enable_external_reset:
            If True, the integrator will reset its state to the value provided
            by the reset value input port when the reset trigger is True. Otherwise,
            the integrator will reset to the initial value.  Adds an additional
            input port for the reset value.  This signal should match the shape
            and dtype of the initial continuous state.
        enable_limits:
            If True, the integrator will constrain its state and output to within
            the upper and lower limits. Either limit may be disbale by setting its
            value to None.
        enable_hold:
            If True, the integrator will hold integration when the hold trigger is
            True.
        reset_on_enter_zeno:
            If True, the integrator will reset its state to the initial value
            when the integrator enters the Zeno state.  This option is ignored unless
            `enable_reset` is True.
        zeno_tolerance:
            The tolerance used to determine if the integrator is in the Zeno state.
            If the time between events is less than this tolerance, then the
            integrator is in the Zeno state.  This option is ignored unless
            `enable_reset` is True.


    Events:
        An event is triggered when the "reset" port changes.

        An event is triggered when the state hit one of the limits.

        An event is triggered when the "hold" port changes.

        Another guard is conditionally active when the integrator is in the Zeno
        state, and is triggered when the "reset" port changes from True to False.
        This event is used to exit the Zeno state and resume normal integration.
    """

    @parameters(
        static=[
            "enable_reset",
            "enable_external_reset",
            "enable_limits",
            "enable_hold",
            "reset_on_enter_zeno",
        ],
        dynamic=["initial_state", "zeno_tolerance", "lower_limit", "upper_limit"],
    )
    def __init__(
        self,
        initial_state,
        enable_reset=False,
        enable_limits=False,
        lower_limit=None,
        upper_limit=None,
        enable_hold=False,
        enable_external_reset=False,
        zeno_tolerance=1e-6,
        reset_on_enter_zeno=False,
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dtype = dtype
        self.enable_reset = enable_reset
        self.enable_external_reset = enable_external_reset
        self.enable_hold = enable_hold
        self.discrete_state_type = namedtuple(
            "IntegratorDiscreteState", ["zeno", "counter", "tprev"]
        )

        self.xdot_index = self.declare_input_port(name="in_0")

        self.x0 = cnp.array(initial_state, dtype=self.dtype)
        self.dtype = self.dtype if self.dtype is not None else self.x0.dtype
        self._continuous_state_idx = self.declare_continuous_state(
            default_value=self.x0,
            ode=self._ode,
            prerequisites_of_calc=[self.input_ports[self.xdot_index].ticket],
        )

        if enable_reset:
            # Boolean input for triggering reset
            self.reset_trigger_index = self.declare_input_port(name="reset_trigger")
            # prerequisites_of_calc.append(
            #     self.input_ports[self.reset_trigger_index].ticket
            # )

            # Declare a custom discrete state to track Zeno behavior
            self.declare_discrete_state(
                default_value=self.discrete_state_type(
                    zeno=False, counter=0, tprev=0.0
                ),
                as_array=False,
            )

            #
            # Declare reset event
            #
            # when reset is triggered, execute the reset map.
            self.declare_zero_crossing(
                guard=self._reset_guard,
                reset_map=self._reset,
                name="reset_on",
                direction="negative_then_non_negative",
            )
            # when reset is deasserted, do not change the state.
            self.declare_zero_crossing(
                guard=self._reset_guard,
                name="reset_off",
                direction="positive_then_non_positive",
            )

            self.declare_zero_crossing(
                guard=self._exit_zeno_guard,
                reset_map=self._exit_zeno,
                name="exit_zeno",
                direction="positive_then_non_positive",
            )

            # Optional: reset value defined by external signal
            if enable_external_reset:
                self.reset_value_index = self.declare_input_port(name="reset_value")
                # prerequisites_of_calc.append(
                #     self.input_ports[self.reset_value_index].ticket
                # )

        if enable_hold:
            # Boolean input for triggering hold assert/deassert
            self.hold_trigger_index = self.declare_input_port(name="hold_trigger")

            def _hold_guard(_time, _state, *inputs, **_params):
                trigger = inputs[self.hold_trigger_index]
                return cnp.where(trigger, 1.0, -1.0)

            self.declare_zero_crossing(
                guard=_hold_guard,
                name="hold",
                direction="crosses_zero",
            )

        self._output_port_idx = self.declare_output_port(name="out_0")

    def initialize(
        self,
        initial_state,
        enable_reset=False,
        enable_limits=False,
        lower_limit=None,
        upper_limit=None,
        enable_hold=False,
        enable_external_reset=False,
        zeno_tolerance=1e-6,
        reset_on_enter_zeno=False,
    ):
        if self.enable_reset != enable_reset:
            raise ValueError("enable_reset cannot be changed after initialization")
        if self.enable_external_reset != enable_external_reset:
            raise ValueError(
                "enable_external_reset cannot be changed after initialization"
            )
        if self.enable_hold != enable_hold:
            raise ValueError("enable_hold cannot be changed after initialization")

        # Default initial condition unless modified in context
        self.x0 = cnp.array(initial_state, dtype=self.dtype)
        self.dtype = self.dtype if self.dtype is not None else self.x0.dtype

        self.configure_continuous_state(
            self._continuous_state_idx,
            default_value=self.x0,
            ode=self._ode,
            prerequisites_of_calc=[self.input_ports[self.xdot_index].ticket],
        )

        self.zeno_tolerance = zeno_tolerance
        self.reset_on_enter_zeno = reset_on_enter_zeno

        self.enable_limits = enable_limits
        self.has_lower_limit = lower_limit is not None
        self.has_upper_limit = upper_limit is not None

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
        )

        if enable_limits:
            if lower_limit is not None:

                def _lower_limit_guard(_time, state, *_inputs, **params):
                    return state.continuous_state - params["lower_limit"]

                self.declare_zero_crossing(
                    guard=_lower_limit_guard,
                    name="lower_limit",
                    direction="positive_then_non_positive",
                )

            if upper_limit is not None:

                def _upper_limit_guard(_time, state, *_inputs, **params):
                    return state.continuous_state - params["upper_limit"]

                self.declare_zero_crossing(
                    guard=_upper_limit_guard,
                    name="upper_limit",
                    direction="negative_then_non_negative",
                )

    def _ode(self, _time, state, *inputs, **params):
        # Normally, just integrate the input signal
        xdot = inputs[self.xdot_index]

        # However, if the reset trigger is high or the integrator is in the Zeno state,
        # then the integrator should hold
        if self.enable_reset:
            trigger = inputs[self.reset_trigger_index]
            in_zeno_state = state.discrete_state.zeno
            xdot = cnp.where((trigger | in_zeno_state), cnp.zeros_like(xdot), xdot)

        # Additionally, if the limits are enabled, the derivative is set to zero if
        # either limit is presnetly violated.
        if self.enable_limits:
            xc = state.continuous_state

            if self.has_lower_limit:
                llim_violation = cnp.logical_and(
                    xdot < 0.0, xc <= params["lower_limit"]
                )
            else:
                llim_violation = False

            if self.has_upper_limit:
                ulim_violation = cnp.logical_and(
                    xdot > 0.0, xc >= params["upper_limit"]
                )
            else:
                ulim_violation = False

            xdot = cnp.where(
                (llim_violation | ulim_violation), cnp.zeros_like(xdot), xdot
            )

        if self.enable_hold:
            hold = inputs[self.hold_trigger_index]
            xdot = cnp.where(hold, cnp.zeros_like(xdot), xdot)

        return xdot

    def _output(self, _time, state, *_inputs, **params):
        xc = state.continuous_state
        if self.enable_limits:
            lower_limit = params["lower_limit"] if self.has_lower_limit else -np.inf
            upper_limit = params["upper_limit"] if self.has_upper_limit else np.inf
            return cnp.clip(xc, lower_limit, upper_limit)

        return xc

    def _reset_guard(self, _time, _state, *inputs, **_params):
        trigger = inputs[self.reset_trigger_index]
        return cnp.where(trigger, 1.0, -1.0)

    def _reset(self, time, state, *inputs, **_params):
        # If the distance between events is less than the tolerance, then enter the Zeno state.
        dt = time - state.discrete_state.tprev
        zeno = (dt - self.zeno_tolerance) <= 0
        tprev = time

        # Handle the reset event as usual
        if self.enable_external_reset:
            xc = inputs[self.reset_value_index]
        else:
            xc = self.x0

        # Don't reset if entering Zeno state
        new_continuous_state = cnp.where(
            zeno & (not self.reset_on_enter_zeno),
            state.continuous_state,
            xc,
        )
        state = state.with_continuous_state(new_continuous_state)

        # Count number of resets (for debugging)
        counter = state.discrete_state.counter + 1

        # Update the discrete state
        xd_plus = self.discrete_state_type(zeno=zeno, counter=counter, tprev=tprev)
        state = state.with_discrete_state(xd_plus)

        logger.debug("Resetting to %s", state)
        return state

    def _exit_zeno_guard(self, _time, _state, *inputs, **_params):
        # This will only be active when in the Zeno state.  It monitors the boolean trigger input
        # and will go from 1.0 (when trigger=True) to 0.0 (when trigger=False)
        trigger = inputs[self.reset_trigger_index]
        return cnp.array(trigger, dtype=self.dtype)

    def _exit_zeno(self, _time, state, *_inputs, **_params):
        xd = state.discrete_state._replace(zeno=False)
        return state.with_discrete_state(xd)

    def determine_active_guards(self, root_context):
        # TODO: Update this to use the new zero crossing event system
        # defined in LeafSystem.
        zero_crossing_events = self.zero_crossing_events.mark_all_active()

        if not self.enable_reset:
            return zero_crossing_events

        def _get_reset(events: LeafEventCollection):
            return events.events[0]

        context = root_context[self.system_id]
        in_zeno_state = context.discrete_state.zeno

        reset = cond(
            in_zeno_state,
            lambda e: e.mark_inactive(),
            lambda e: e.mark_active(),
            _get_reset(zero_crossing_events),
        )

        def _get_exit_zeno(events: LeafEventCollection):
            return events.events[1]

        exit_zeno: ZeroCrossingEvent = cond(
            in_zeno_state,
            lambda e: e.mark_active(),
            lambda e: e.mark_inactive(),
            _get_exit_zeno(zero_crossing_events),
        )

        zero_crossing_events = eqx.tree_at(_get_reset, zero_crossing_events, reset)
        zero_crossing_events = eqx.tree_at(
            _get_exit_zeno, zero_crossing_events, exit_zeno
        )

        return zero_crossing_events

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xc = context[self.system_id].continuous_state
        check_state_type(
            self,
            inp_data=u,
            state_data=xc,
            error_collector=error_collector,
        )


class IntegratorDiscrete(LeafSystem):
    """Discrete first-order integrator.

    This block is a discrete-time approximation to the behavior of the Integrator
    block.  It implements the following linear time-invariant difference equation
    for input values `u` and output values `y`:
    ```
        x[k+1] = x[k] + dt * u[k]
        y[k] = x[k]
    ```
    where `x` is the state of the integrator.  The integrator is initialized with
    the value of the `initial_state` parameter.

    Options:
        Reset: the integrator can be configured to reset its state on an input
            trigger.  The reset value can be either the initial state of the
            integrator or an external value provided by an input port.
        Limits: the integrator can be configured such that the output and state
            are constrained by upper and lower limits.
        Hold: the integrator can be configured to hold integration based on an
            input trigger.

    Unlike the continuous-time integrator, the discrete integrator does not detect
    Zeno behavior, since this is not a concern in discrete-time systems.

    Input ports:
        (0) The input signal.  Must match the shape and dtype of the initial
            state.
        (1) The reset trigger.  Optional, only if `enable_reset` is True.
        (2) The reset value.  Optional, only if `enable_external_reset` is True.
        (3) The hold trigger. Optional, only if 'enable_hold' is True.

    Output ports:
        (0) The current state of the integrator.

    Parameters:
        initial_state:
            The initial value of the integrator state.  Can be any array, or even
            a nested structure of arrays, but the data type should be floating-point.
        enable_reset:
            If True, the integrator will reset its state to the initial value
            when the reset trigger is True.  Adds an additional input port for
            the reset trigger.  This signal should be boolean- or binary-valued.
        enable_external_reset:
            If True, the integrator will reset its state to the value provided
            by the reset value input port when the reset trigger is True. Otherwise,
            the integrator will reset to the initial value.  Adds an additional
            input port for the reset value.  This signal should match the shape
            and dtype of the initial continuous state.
        enable_limits:
            If True, the integrator will constrain its state and output to within
            the upper and lower limits. Either limit may be disbale by setting its
            value to None.
        enable_hold:
            If True, the integrator will hold integration when the hold trigger is
            True.
    """

    @parameters(
        static=[
            "enable_reset",
            "enable_external_reset",
            "enable_limits",
            "enable_hold",
        ],
        dynamic=["initial_state", "lower_limit", "upper_limit"],
    )
    def __init__(
        self,
        dt,
        initial_state,
        enable_reset=False,
        enable_hold=False,
        enable_limits=False,
        lower_limit=None,
        upper_limit=None,
        enable_external_reset=False,
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.dtype = dtype

        self.enable_reset = enable_reset
        self.enable_external_reset = enable_external_reset

        self.xdot_index = self.declare_input_port(
            name="in_0"
        )  # One vector-valued input

        self._periodic_update_idx = self.declare_periodic_update()

        if enable_reset:
            self.reset_trigger_index = self.declare_input_port(
                name="reset_trigger"
            )  # Boolean input for triggering reset

            if enable_external_reset:
                self.reset_value_index = self.declare_input_port(
                    name="reset_value"
                )  # Optional reset value

        self.enable_hold = enable_hold
        if enable_hold:
            self.hold_trigger_index = self.declare_input_port(
                name="hold_trigger"
            )  # Boolean input for triggering hold

        self.state_output_index = self.declare_output_port(name="out_0")

    def initialize(
        self,
        initial_state,
        enable_reset=False,
        enable_hold=False,
        enable_limits=False,
        lower_limit=None,
        upper_limit=None,
        enable_external_reset=False,
    ):
        if self.enable_reset != enable_reset:
            raise ValueError("enable_reset cannot be changed after initialization")
        if self.enable_external_reset != enable_external_reset:
            raise ValueError(
                "enable_external_reset cannot be changed after initialization"
            )
        if self.enable_hold != enable_hold:
            raise ValueError("enable_hold cannot be changed after initialization")

        # Default initial condition unless modified in context
        self.x0 = cnp.array(initial_state, dtype=self.dtype)
        self.dtype = self.dtype if self.dtype is not None else self.x0.dtype

        self.declare_discrete_state(default_value=self.x0)
        self.configure_periodic_update(
            self._periodic_update_idx, self._update, period=self.dt, offset=0.0
        )

        # Since the reset is applied to the output port, having this
        # active makes the block feedthrough with respect to related
        # input ports.
        self.is_feedthrough = enable_reset

        self.enable_limits = enable_limits
        self.has_lower_limit = lower_limit is not None
        self.has_upper_limit = upper_limit is not None

        prereqs = [DependencyTicket.xd]
        if enable_reset:
            prereqs.append(self.input_ports[self.reset_trigger_index].ticket)
            if enable_external_reset:
                prereqs.append(self.input_ports[self.reset_value_index].ticket)

        self.configure_output_port(
            self.state_output_index,
            self._output,
            period=self.dt,
            offset=0.0,
            default_value=self.x0,
            prerequisites_of_calc=prereqs,
        )

    def _reset(self, *inputs):
        if self.enable_external_reset:
            return inputs[self.reset_value_index]
        return self.x0

    def _apply_reset_and_limits(self, x_new, *inputs, **params):
        # Reset and limits are applied to both the update and outputs
        # so that they respond to the discontinuities simultaneously.

        if self.enable_reset:
            # If the reset is high, then return the reset value
            trigger = inputs[self.reset_trigger_index]
            x_new = cnp.where(trigger, self._reset(*inputs), x_new)

        if self.enable_limits:
            lower_limit = params["lower_limit"] if self.has_lower_limit else -cnp.inf
            upper_limit = params["upper_limit"] if self.has_upper_limit else cnp.inf
            x_new = cnp.clip(x_new, lower_limit, upper_limit)

        return x_new

    def _apply_hold(self, x, x_new, *inputs, **_params):
        # Hold is only applied to the update, but not the output

        if self.enable_hold:
            # If the reset is high, then return the reset value
            trigger = inputs[self.hold_trigger_index]
            x_new = cnp.where(trigger, x, x_new)

        return x_new

    def _update(self, _time, state, *inputs, **params):
        x = state.discrete_state
        xdot = inputs[self.xdot_index]
        x_new = x + self.dt * xdot
        x_new = self._apply_hold(x, x_new, *inputs, **params)
        x_new = self._apply_reset_and_limits(x_new, *inputs, **params)
        return x_new.astype(x.dtype)

    def _output(self, _time, state, *inputs, **params):
        x = state.discrete_state
        # To ensure that the discontinuities happen simultaneously with
        # the input signal, also apply the reset and limits to the outputs.
        # this makes the block feedthrough.
        y = self._apply_reset_and_limits(x, *inputs, **params)
        return y

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xd = context[self.system_id].discrete_state
        check_state_type(
            self,
            inp_data=u,
            state_data=xd,
            error_collector=error_collector,
        )


class IOPort(FeedthroughBlock):
    """Simple class for organizing input/output ports for groups/submodels.

    Since these are treated as standalone blocks in the UI rather than specific
    input/output ports exported to the parent model, it is more straightforward
    to represent them that way here as well.

    This class represents a simple one-input, one-output feedthrough block where
    the feedthrough function is an identity.  The input (resp. output) port can then
    be exported to the parent model to create an Inport (resp. Outport).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: x, *args, **kwargs)


class Logarithm(FeedthroughBlock):
    """Compute the logarithm of the input signal.

    This block dispatches to `jax.numpy.log`, `jax.numpy.log2`, or `jax.numpy.log10`,
    so the semantics, broadcasting rules, etc. are the same.  See the JAX docs for
    details:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log.html
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log2.html
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log10.html

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The logarithm of the input signal.

    Parameters:
        base:
            One of "natural", "2", or "10". Determines the base of the logarithm.
            The default is "natural".
    """

    @parameters(static=["base"])
    def __init__(self, base="natural", **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, base="natural"):
        func_lookup = {
            "10": cnp.log10,
            "2": cnp.log2,
            "natural": cnp.log,
        }
        if base not in func_lookup:
            # cannot pass system=self because this error must be raised BEFORE calling super.__init__()
            # in the case of inheritting from FeedthroughBlock.
            # if we call super.__init__() first, we get missing key error for func_lookup[base].
            raise BlockParameterError(
                message=f"Logarithm block {self.name} has invalid selection {base} for 'base'. Valid selections: "
                + ", ".join([k for k in func_lookup.keys()]),
                parameter_name="base",
            )
        self.replace_op(func_lookup[base])


class LogicalOperator(LeafSystem):
    """Apply a boolean function elementwise to the input signals.

    This block implements the following boolean functions:
        - "or": same as np.logical_or
        - "and": same as np.logical_and
        - "not": same as np.logical_not
        - "nor": equivalent to np.logical_not(np.logical_or(in_0,in_1))
        - "nand": equivalent to np.logical_not(np.logical_and(in_0,in_1))
        - "xor": same as np.logical_xor

    Input ports:
        (0,1) The input signals.  If numeric, they are interpreted as boolean
            types (so 0 is False and any other value is True).

    Output ports:
        (0) The result of the logical operation, a boolean-valued signal.

    Parameters:
        function:
            The boolean function to apply. One of "or", "and", "not", "nor", "nand",
            or "xor".

    Events:
        An event is triggered when the output changes from True to False or vice versa.
    """

    @parameters(static=["function"])
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port()
        if not function == "not":
            self.declare_input_port()
        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

    def initialize(self, function):
        self.function = function
        func_lookup = {
            "or": self._or,
            "and": self._and,
            "not": self._not,
            "xor": self._xor,
            "nor": self._nor,
            "nand": self._nand,
        }
        if function not in func_lookup:
            raise BlockParameterError(
                message=f"LogicalOperator block {self.name} has invalid selection {function} for 'function'. Valid options: "
                + ", ".join([f for f in func_lookup.keys()]),
                system=self,
            )

        if function != "not" and len(self.input_ports) < 2:
            raise BlockParameterError(
                message=f"Can't change logical operator from 'not' to {function} for block {self.name}",
                system=self,
            )

        if function == "not" and len(self.input_ports) > 1:
            raise BlockParameterError(
                message=f"Can't change logical operator from {function} to 'not' for block {self.name}",
                system=self,
            )

        self._func = func_lookup[function]

        self.configure_output_port(
            self._output_port_idx,
            self._func,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

    def _edge_detection(self, time, state, *inputs, **params):
        outp = self._func(time, state, *inputs, **params)
        return cnp.where(outp, 1.0, -1.0)

    def _or(self, time, state, *inputs, **parameters):
        return cnp.logical_or(cnp.array(inputs[0]), jnp.array(inputs[1]))

    def _and(self, time, state, *inputs, **parameters):
        return cnp.logical_and(cnp.array(inputs[0]), jnp.array(inputs[1]))

    def _not(self, time, state, *inputs, **parameters):
        (x,) = inputs
        return cnp.logical_not(cnp.array(x))

    def _xor(self, time, state, *inputs, **parameters):
        return cnp.logical_xor(cnp.array(inputs[0]), cnp.array(inputs[1]))

    def _nor(self, time, state, *inputs, **parameters):
        return cnp.logical_not(
            cnp.logical_or(jnp.array(inputs[0]), cnp.array(inputs[1]))
        )

    def _nand(self, time, state, *inputs, **parameters):
        return jnp.logical_not(
            cnp.logical_and(cnp.array(inputs[0]), cnp.array(inputs[1]))
        )

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity.  For efficiency, only do this if the output
        # is fed to an ODE block
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(self._edge_detection, direction="crosses_zero")

        return super().initialize_static_data(context)


class LogicalReduce(FeedthroughBlock):
    """Apply a boolean reduce function to the elements of the input signal.

    This block implements the following boolean functions:
        - "any": Output is True if any input element is True.
        - "all": Output is True if all input elements are True.

    Input ports:
        (0) The input signal.  If numeric, they are interpreted as boolean
            types (so 0 is False and any other value is True).

    Output ports:
        (0) The result of the logical operation, a boolean-valued signal.

    Parameters:
        function:
            The boolean function to apply. One of "any", "all".
        axis:
            Axis or axes along which a logical OR/AND reduction is performed.

    Events:
        An event is triggered when the output changes from True to False or vice versa.
    """

    @parameters(static=["function", "axis"])
    def __init__(self, function, axis=None, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, function, axis=None):
        self.function = function
        self.axis = int(axis) if axis is not None else None
        func_lookup = {
            "any": self._any,
            "all": self._all,
        }
        if function not in func_lookup:
            raise BlockParameterError(
                message=f"LogicalReduce block {self.name} has invalid selection {function} for 'function'. Valid options: "
                + ", ".join([f for f in func_lookup.keys()])
            )

        self._func = func_lookup[function]
        self.replace_op(self._func)

    def _edge_detection(self, _time, _state, *inputs, **_params):
        outp = self._func(inputs)
        return cnp.where(outp, 1.0, -1.0)

    def _any(self, inputs):
        return cnp.any(cnp.array(inputs), axis=self.axis)

    def _all(self, inputs):
        return cnp.all(cnp.array(inputs), axis=self.axis)

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity.  For efficiency, only do this if the output
        # is fed to an ODE block
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(self._edge_detection, direction="crosses_zero")

        return super().initialize_static_data(context)


class LookupTable1d(FeedthroughBlock):
    """Interpolate the input signal into a static lookup table.

    If a function `y = f(x)` is sampled at a set of points `(x_i, y_i)`, then this
    block will interpolate the input signal `x` to compute the output signal `y`.
    The behavior is modeled after `scipy.interpolate.interp1d` but is implemented
    in JAX.  Available interpolation modes are:
        - "linear": Linear interpolation using `jax.interp`.
        - "nearest": Nearest-neighbor interpolation.
        - "flat": Flat interpolation.

    Input ports:
        (0) The input signal, which is used as the interpolation coordinate.

    Output ports:
        (0) The interpolated output signal.

    Parameters:
        input_array:
            The array of input values at which the output values are provided.
        output_array:
            The array of output values.
        interpolation:
            One of "linear", "nearest", or "flat". Determines the type of interpolation
            performed by the block.

    Notes:
        Currently restricted to 1D input and output data.  This may be expanded to
        support multi-dimensional output arrays in the future.
    """

    @parameters(static=["input_array", "output_array", "interpolation"])
    def __init__(self, input_array, output_array, interpolation, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, input_array, output_array, interpolation):
        self.input_array = cnp.array(input_array)
        self.output_array = cnp.array(output_array)
        if len(self.input_array.shape) != 1:
            raise ValueError(
                f"LookupTable1d block {self.name} input_array must be 1D, got shape "
                f"{self.input_array.shape}"
            )
        if len(self.output_array.shape) != 1:
            raise ValueError(
                f"LookupTable1d block {self.name} output_array must be 1D, got shape "
                f"{self.output_array.shape}"
            )
        self.max_i = len(self.input_array) - 1

        func_lookup = {
            "linear": self._lookup_linear,
            "nearest": self._lookup_nearest,
            "flat": self._lookup_flat,
        }
        if interpolation not in func_lookup:
            raise ValueError(
                f"LookupTable1d block {self.name} has invalid selection {interpolation} "
                "for 'interpolation'"
            )
        self.replace_op(func_lookup[interpolation])

    def _lookup_linear(self, x):
        return cnp.interp(x, self.input_array, self.output_array)

    def _lookup_nearest(self, x):
        i = cnp.argmin(cnp.abs(self.input_array - x))
        i = cnp.clip(i, 0, self.max_i)
        return self.output_array[i]

    def _lookup_flat(self, x):
        i = cnp.where(
            x < self.input_array[1],
            0,
            cnp.argmin(x >= self.input_array) - 1,
        )
        return self.output_array[i]


class LookupTable2d(LeafSystem):
    """Interpolate the input signals into a static lookup table.

    The behavior is modeled on `scipy.interpolate.interp2d` but is implemented
    in JAX.  The only currently implemented interpolation mode is "linear". The
    input arrays must be 1D and the output array must be 2D.

    Input ports:
        (0) The first input signal, used as the first interpolation coordinate.
        (1) The second input signal, used as the second interpolation coordinate.

    Output ports:
        (0) The interpolated output signal.

    Parameters:
        input_x_array:
            The array of input values at which the output values are provided,
            corresponding to the first input signal. Must be 1D
        input_y_array:
            The array of input values at which the output values are provided,
            corresponding to the second input signal. Must be 1D
        output_table_array:
            The array of output values. Must be 2D with shape `(m, n)`, where
            `m = len(input_x_array)` and `n = len(input_y_array)`.
        interpolation:
            Only "linear" is supported.
    """

    @parameters(
        static=["input_x_array", "input_y_array", "output_table_array", "interpolation"]
    )
    def __init__(
        self,
        input_x_array,
        input_y_array,
        output_table_array,
        interpolation="linear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.declare_input_port()
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

    def initialize(
        self, input_x_array, input_y_array, output_table_array, interpolation
    ):
        xp = cnp.array(input_x_array)
        yp = cnp.array(input_y_array)
        zp = cnp.array(output_table_array)

        if len(xp.shape) != 1:
            raise ValueError(
                f"LookupTable2d block {self.name} input_x_array must be 1D, got "
                f"shape {xp.shape}"
            )

        if len(yp.shape) != 1:
            raise ValueError(
                f"LookupTable2d block {self.name} input_y_array must be 1D, got "
                f"shape {yp.shape}"
            )

        if len(zp.shape) != 2:
            raise ValueError(
                f"LookupTable2d block {self.name} output_table_array must be 2D, "
                f"got shape {zp.shape}"
            )

        if zp.shape != (len(xp), len(yp)):
            raise ValueError(
                f"LookupTable2d block {self.name} output_table_array must have "
                f"shape (len(input_x_array), len(input_y_array)), got shape {zp.shape}"
            )

        if interpolation != "linear":
            raise NotImplementedError(
                f"LookupTable2d block {self.name} only supports linear interpolation."
            )

        self._compute_output = partial(cnp.interp2d, xp, yp, zp)

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

    def _output(self, _time, _state, *inputs, **params):
        (x, y) = inputs
        return self._compute_output(x, y)


class MatrixConcatenation(ReduceBlock):
    """Concatenate two matrices along a given axis.

    Dispatches to `jax.numpy.concatenate`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.concatenate.html

    Args:
        axis: The axis along which the matrices are concatenated. 0 for vertical
            and 1 for horizontal. Default is 0.

    Input ports:
        (0, 1) The input matrices `A` and `B`

    Output ports:
        (0) The concatenation input matrices: e.g. `[A,B]`.
    """

    @parameters(static=["axis"])
    def __init__(self, n_in=2, axis=0, **kwargs):
        if n_in != 2:
            raise ValueError(
                "MatrixConcatenation block only supports two input matrices."
            )
        super().__init__(2, None, **kwargs)

    def initialize(self, axis):
        def _func(inputs):
            return cnp.concatenate((inputs[0], inputs[1]), axis=int(axis))

        self.replace_op(_func)


class MatrixInversion(FeedthroughBlock):
    """Compute the matrix inverse of the input signal.

    Dispatches to `jax.numpy.inv`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.inv.html

    Input ports:
        (0) The input matrix.

    Output ports:
        (0) The inverse of the input matrix.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(cnp.linalg.inv, *args, **kwargs)


class MatrixMultiplication(ReduceBlock):
    """Compute the matrix product of the input signals.

    Dispatches to `jax.numpy.matmul`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html

    Input ports:
        (0, 1) The input matrices `A` and `B`

    Output ports:
        (0) The matrix product of the input matrices: `A @ B`.
    """

    def __init__(
        self,
        n_in=2,
        **kwargs,
    ):
        if n_in != 2:
            raise ValueError(
                "MatrixMultiplication block only supports two input signals."
            )

        def _func(inputs):
            return cnp.matmul(inputs[0], inputs[1])

        super().__init__(2, _func, **kwargs)


class MatrixTransposition(FeedthroughBlock):
    """Compute the matrix transpose of the input signal.

    Dispatches to `jax.numpy.transpose`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html

    Input ports:
        (0) The input matrix.

    Output ports:
        (0) The transpose of the input matrix.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(cnp.transpose, *args, **kwargs)


class MinMax(ReduceBlock):
    """Return the extremum of the input signals.

    Input ports:
        (0..n_in-1) The input signals.

    Output ports:
        (0) The minimum or maximum of the input signals.

    Parameters:
        operator:
            One of "min" or "max". Determines whether the block returns the minimum
            or maximum of the input signals.

    Events:
        An event is triggered when the extreme input signal changes.  For example,
        if the block is configured as a "max" block with two inputs and the second
        signal becomes greater than the first, a zero-crossing event will be triggered.
    """

    @parameters(static=["operator"])
    def __init__(self, n_in, operator, **kwargs):
        super().__init__(n_in, None, **kwargs)

    def initialize(self, operator):
        func_lookup = {
            "max": self._max,
            "min": self._min,
        }
        if operator not in func_lookup:
            # cannot pass system=self because this error must be raised BEFORE calling super.__init__()
            # in the case of inheritting from FeedthroughBlock.
            # if we call super.__init__() first, we get missing key error for func_lookup[base].
            raise BlockParameterError(
                message=f"MinMax block {self.name} has invalid selection {operator} for 'operator'. Valid options: "
                + ", ".join([f for f in func_lookup.keys()]),
                parameter_name="operator",
            )

        self.operator = operator

        self.replace_op(func_lookup[operator])

        guard_lookup = {
            "max": self._max_guard,
            "min": self._min_guard,
        }

        self._guard = guard_lookup[operator]

    def _min(self, inputs):
        return cnp.min(cnp.array(inputs))

    def _max(self, inputs):
        return cnp.max(cnp.array(inputs))

    def _min_guard(self, _time, _state, *inputs, **_params):
        return cnp.argmin(cnp.array(inputs)).astype(float)

    def _max_guard(self, _time, _state, *inputs, **_params):
        return cnp.argmax(cnp.array(inputs)).astype(float)

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output
        # is fed to an ODE block
        if not self.has_zero_crossing_events and (self.output_ports[0]):
            self.declare_zero_crossing(self._guard, direction="edge_detection")

        return super().initialize_static_data(context)


class Multiplexer(ReduceBlock):
    """Stack the input signals into a single output signal.

    Dispatches to `jax.numpy.hstack`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.hstack.html

    Input ports:
        (0..n_in-1) The input signals.

    Output ports:
        (0) The stacked output signal.
    """

    def __init__(self, n_in, *args, **kwargs):
        super().__init__(n_in, cnp.hstack, *args, **kwargs)


class Offset(FeedthroughBlock):
    """Add a constant offset or bias to the input signal.

    Given an input signal `u` and offset value `b`, this will return `y = u + b`.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal plus the offset.

    Parameters:
        offset:
            The constant offset to add to the input signal.
    """

    @parameters(dynamic=["offset"])
    def __init__(self, offset, *args, **kwargs):
        super().__init__(lambda x, offset: x + offset, *args, **kwargs)

    def initialize(self, offset):
        pass


class PIDDiscrete(LeafSystem):
    """Discrete-time PID controller.

    This block implements a discrete-time PID controller with a first-order
    approximation to the integrated error and an optional derivative filter.
    The integrated error term is computed as:
    ```
        e_int[k+1] = e_int[k] + e[k] * dt
    ```
    where `e` is the error signal and `dt` is the sampling period.  The derivative
    term is computed in the same way as for the DerivativeDiscrete block, including
    filter options described there.  With the running error integral `e_int` and
    current estimate of the time derivative of the error `e_dot`, the output is:
    ```
        u[k] = kp * e[k] + ki * e_int[k] + kd * e_dot[k]
    ```

    Input ports:
        (0) The error signal.

    Output ports:
        (0) The control signal computed by the PID algorithm.

    Parameters:
        kp:
            The proportional gain (scalar)
        ki:
            The integral gain (scalar)
        kd:
            The derivative gain (scalar)
        dt:
            The sampling period of the block.
        initial_state:
            The initial value of the running error integral.  Default is 0.
        enable_external_initial_state:
            Source for the value used for the integrator initial state. True=from inport,
            False=from the initial_state parameter.
        filter_type:
            One of "none", "forward", "backward", or "bilinear".  Determines the type of
            filter used to estimate the derivative of the error signal.  Default is
            "none".  See DerivativeDiscrete documentation for details.
        filter_coefficient:
            The filter coefficient for the derivative filter.  Default is 1.0.  See
            DerivativeDiscrete documentation for details.
    """

    class DiscreteStateType(NamedTuple):
        integral: Array
        # Recursive filter memory for the derivative estimate
        e_prev: Array
        e_dot_prev: Array

    @parameters(
        static=["filter_type", "filter_coefficient"],
        dynamic=["kp", "ki", "kd", "initial_state"],
    )
    def __init__(
        self,
        dt,
        kp=1.0,
        ki=1.0,
        kd=1.0,
        initial_state=0.0,
        enable_external_initial_state=False,
        filter_type="none",
        filter_coefficient=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.input_index = self.declare_input_port()

        self.enable_external_initial_state = enable_external_initial_state
        self.initial_state_index = None
        if enable_external_initial_state:
            self.initial_state_index = self.declare_input_port()

        # Declare the periodic update
        self._periodic_update_idx = self.declare_periodic_update()

        # Declare an output port for the control signal
        self.control_output = self.declare_output_port()

        # NOTE:
        # An extra output port for the derivative value is not strictly necessary,
        # but the filtered estimate could be resused elsewhere.  Also, having the
        # previous value saved in the discrete output component of state would allows
        # it to be reused in the recursive filter without recomputing it as part of
        # the update step, a minor efficiency gain.  The tradeoff is an extra event
        # that has to be handled.  This implementation uses one output event and
        # re-does the derivative calculation when a recursive filter is used, but
        # we could always do it the other way in the future.

    def initialize(
        self,
        kp,
        ki,
        kd,
        initial_state,
        filter_type,
        filter_coefficient,
    ):
        # Declare an internal discrete state
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(
                integral=initial_state,
                e_prev=0.0,
                e_dot_prev=0.0,
            ),
            as_array=False,
        )

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # Determine the coefficients of the filter, if applicable
        # The filter is a pair of two-element array and the filter
        # equation is:
        # a0*y[k] + a1*y[k-1] = b0*u[k] + b1*u[k-1]
        self.filter_type = filter_type
        self.filter = derivative_filter(
            N=filter_coefficient, dt=self.dt, filter_type=filter_type
        )

        self.configure_output_port(
            self.control_output,
            self._output,
            period=self.dt,
            offset=0.0,
            default_value=initial_state,
            prerequisites_of_calc=[DependencyTicket.xd, self.input_ports[0].ticket],
        )

    def _eval_derivative(self, _time, state, *inputs, **_params):
        # Filtered derivative estimate

        e = inputs[self.input_index]  # Error signal from upstream
        e_prev = state.discrete_state.e_prev
        b, a = self.filter  # IIR filter coefficients

        # If the filter is recursive we need to reuse the previous derivative
        # estimate.
        if self.filter_type != "none":
            # Filtered estimate of the time derivative
            e_dot_prev = state.discrete_state.e_dot_prev

            # New estimate of the time derivative of the error signal
            e_dot = (b[0] * e + b[1] * e_prev - a[1] * e_dot_prev) / a[0]

        else:
            # Standard finite difference approximation - no recursion
            e_dot = (b[0] * e + b[1] * e_prev) / a[0]

        return e_dot

    def _update(self, time, state, *inputs, **params):
        e = inputs[self.input_index]  # Error signal from upstream

        # Integrated error signal
        e_int = state.discrete_state.integral

        # Update the derivative estimate if needed for a recursive filter.
        if self.filter_type != "none":
            e_dot = self._eval_derivative(time, state, *inputs, **params)
        else:
            # This state entry isn't used for the finite difference estimator.
            # Can just keep the original value as a placeholder.
            e_dot = state.discrete_state.e_dot_prev

        # Update the internal state
        return self.DiscreteStateType(
            integral=e_int + e * self.dt, e_prev=e, e_dot_prev=e_dot
        )

    def _eval_control(self, e, e_int, e_dot, **params):
        # Calculate the control signal for the PID control law
        kp, ki, kd = params["kp"], params["ki"], params["kd"]
        u = kp * e + ki * e_int + kd * e_dot
        return u

    def _output(self, time, state, *inputs, **params):
        e = inputs[self.input_index]  # Error signal from upstream
        e_int = state.discrete_state.integral
        e_dot = self._eval_derivative(time, state, *inputs, **params)
        return self._eval_control(e, e_int, e_dot, **params)

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xd = context[self.system_id].discrete_state.integral
        check_state_type(
            self,
            inp_data=u,
            state_data=xd,
            error_collector=error_collector,
        )

    def initialize_static_data(self, context):
        """Set the initial state from the input port, if specified via config"""
        if self.initial_state_index is not None:
            try:
                initial_state = self.eval_input(context, self.initial_state_index)
                default_value = self.DiscreteStateType(
                    integral=initial_state,
                    e_prev=0.0,
                    e_dot_prev=0.0,
                )
                self._default_discrete_state = default_value
                local_context = context[self.system_id].with_discrete_state(
                    default_value
                )
                context = context.with_subcontext(self.system_id, local_context)

            except UpstreamEvalError:
                # The diagram has only been partially created.  Defer the
                # inference of the initial state until the upstream block has been
                # connected.
                logger.debug(
                    "PID_Discrete.initialize_static_data: UpstreamEvalError. "
                    "Continuing without default value initialization."
                )
        return super().initialize_static_data(context)


class Power(FeedthroughBlock):
    """Raise the input signal to a constant power.

    Dispatches to `jax.numpy.power`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.power.html

    For input signal `u` with exponent `p`, the output will be `y = u ** p`.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal raised to the power of the exponent.

    Parameters:
        exponent:
            The exponent to which the input signal is raised.
    """

    @parameters(static=["exponent"])
    def __init__(self, exponent, **kwargs):
        super().__init__(self._func, **kwargs)

        # Note that the exponent here is declared as a configuration
        # parameter and not a context parameter, making it non-differentiable.
        # This is because the derivative rule for the exponent includes a log
        # of the primal input signal, which can cause NaN values during backprop
        # if the input signal is non-positive. Specifically, for `y = u ** p`, the
        # linearization with respect to `p` is `dy = y * log(u) * dp`. If we
        # eventually want to support backprop through this block, we will need
        # to handle the log of the input signal in a way that avoids NaN values.
        # (e.g. with gradient clipping). Tracked in WC-306
        self.exponent = exponent

    def initialize(self, exponent):
        self.exponent = exponent

    def _func(self, *inputs, **parameters):
        (u,) = inputs
        return u**self.exponent


class Product(ReduceBlock):
    """Compute the product and/or quotient of the input signals.

    The block will multiply or divide the input signals, depending on the specified
    operators.  For example, if the block has three inputs `u1`, `u2`, and `u3` and
    is configured with operators="**/", then the output signal will be
    `y = u1 * u2 / u3`.  By default, the block will multiply all of the input signals.

    Input ports:
        (0..n_in-1) The input signals.

    Output ports:
        (0) The product and/or quotient of the input signals.

    Parameters:
        n_in:
            The number of input ports.
        operators:
            A string of length `n_in` specifying the operators to apply to each of
            the input signals.  Each character in the string must be either "*" or "/".
            The default is "*".
        denominator_limit:
            Currently unsupported
        divide_by_zero_behavior:
            Currently unsupported
    """

    @parameters(static=["operators", "denominator_limit", "divide_by_zero_behavior"])
    def __init__(
        self,
        n_in,
        operators=None,  # Expect "**/*", etc
        denominator_limit=None,
        divide_by_zero_behavior=None,
        **kwargs,
    ):
        super().__init__(n_in, None, **kwargs)

    def initialize(
        self,
        operators=None,  # Expect "**/*", etc
        denominator_limit=None,
        divide_by_zero_behavior=None,
    ):
        if operators is not None and any(char not in {"*", "/"} for char in operators):
            raise BlockParameterError(
                message=f"Product block {self.name} has invalid operators {operators}. Can only contain '*' and '/'",
                system=self,
                parameter_name="operators",
            )

        if operators is not None and "/" in operators:
            num_indices = cnp.array(
                [idx for idx, op in enumerate(operators) if op == "*"]
            )
            den_indices = cnp.array(
                [idx for idx, op in enumerate(operators) if op == "/"]
            )

            def _func(inputs):
                ain = cnp.array(inputs)
                num = cnp.take(ain, num_indices, axis=0)
                den = cnp.take(ain, den_indices, axis=0)
                return cnp.prod(num, axis=0) / cnp.prod(den, axis=0)

        else:

            def _func(inputs):
                return cnp.prod(cnp.array(inputs), axis=0)

        self.replace_op(_func)


class ProductOfElements(FeedthroughBlock):
    """Compute the product of the elements of the input signal.

    Dispatches to `jax.numpy.prod`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.prod.html

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The product of the elements of the input signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(cnp.prod, *args, **kwargs)


class Pulse(SourceBlock):
    """A periodic pulse signal.

    Given amplitude `a`, pulse width `w`, and period `p`, the output signal is:
    ```
        y(t) = a if t % p < w else 0
    ```
    where `%` is the modulo operator.

    Input ports:
        None

    Output ports:
        (0) The pulse signal.

    Parameters:
        amplitude:
            The amplitude of the pulse signal.
        pulse_width:
            The fraction of the period during which the pulse is "high".
        period:
            The period of the pulse signal.
        phase_delay:
            Currently unsupported.
    """

    @parameters(dynamic=["amplitude", "pulse_width", "period", "phase_delay"])
    def __init__(
        self, amplitude=1.0, pulse_width=0.5, period=1.0, phase_delay=0.0, **kwargs
    ):
        super().__init__(self._func, **kwargs)

        # Initialize the floating-point tolerance.  This will be machine epsilon
        # for the floating point type of the time variable (determined in the
        # static initialization step).
        self.eps = 0.0

        if abs(phase_delay) > 1e-9:
            warnings.warn("Warning. Pulse block phase_delay not implemented.")

        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        # ad 2 events, one for the up jump, and one the down jump
        self.declare_discrete_state(default_value=False)
        self._dummy_periodic_update_idx = self.declare_periodic_update()
        self._periodic_update_idx = self.declare_periodic_update()

    def initialize(self, amplitude, pulse_width, period, phase_delay):
        if abs(phase_delay) > 1e-9:
            warnings.warn("Warning. Pulse block phase_delay not implemented.")

        self.configure_periodic_update(
            self._dummy_periodic_update_idx,
            lambda *args, **kwargs: True,
            period=period,
            offset=period,
        )

        self.configure_periodic_update(
            self._periodic_update_idx,
            lambda *args, **kwargs: True,
            period=period,
            offset=period + period * pulse_width,
        )

    def _func(self, time, **parameters):
        # Add a floating-point tolerance to the modulo operation to avoid
        # accuracy issues when the time is an "exact" multiple of the period.
        period_fraction = (
            cnp.remainder(time + self.eps, parameters["period"]) / parameters["period"]
        )
        return cnp.where(
            period_fraction >= parameters["pulse_width"],
            0.0,
            parameters["amplitude"],
        )

    def initialize_static_data(self, context):
        # Determine machine epsilon for the type of the time variable
        self.eps = 2 * cnp.finfo(cnp.result_type(context.time)).eps
        return super().initialize_static_data(context)


class Quantizer(FeedthroughBlock):
    """Discritize the input signal into a set of discrete values.

    Given an input signal `u` and a resolution `intervals`, this block will
    quantize the input signal into a set of `intervals` discrete values.
    The output signal will be `y = intervals * round(u / intervals)`.

    Input ports:
        (0) The continuous input signal. In most cases, should be scaled to the range
            `[0, intervals]`.

    Output ports:
        (0) The quantized output signal, on the same scale as the input signal.

    Parameters:
        interval:
            The number of discrete values into which the input signal is quantized.
    """

    @parameters(dynamic=["interval"])
    def __init__(self, interval, *args, **kwargs):
        super().__init__(
            lambda x, interval: interval * cnp.round(x / interval), *args, **kwargs
        )

    def initialize(self, interval):
        pass


class Relay(LeafSystem):
    """Simple state machine implementing hysteresis behavior.

    The input-output map is as follows:

    ```
            output
              |
    on_value  |          -------<------<---------------------
              |          |                    |
              |          ⌄                    ^
              |          |                    |
    off_value |----------|-------->----->-----|
              |
              |---------------------------------------------- input
                         | off_threshold      | on_threshold
    ```

    Note that the "time mode" behavior of this block will follow the input
    signal.  That is, if the input signal varies continuously in time, then
    the zero-crossing event from OFF->ON or vice versa will be localized in
    time.  On the other hand, if the input signal varies only as a result
    of periodic updates to the discrete state, the relay will only change state
    at those instants.  If the input signal is continuous, the block can
    be "forced" to this discrete-time periodic behavior by adding a ZeroOrderHold
    block before the input.

    The exception to this is the case where there are no blocks in the system
    containing either discrete or continuous state.  In this case the state changes
    will only be localized to the resolution of the major step.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The relay output signal, which is equal to either the on_value or
            the off_value, depending on the internal state of the relay.

    Parameters:
        on_threshold:
            When input rises above this value, the internal state transitions to ON.
        off_threshold:
            When input falls below this value, the internal state transitions to OFF.
        on_value:
            Value of the output signal when state is ON.
        off_value:
            Value of the output signal when state is OFF
        initial_state:
            If equal to on_value, the block will be initialized in the ON state.
            Otherwise, it will be initialized to the OFF state.

    Events:
        There are two zero-crossing events: one to transition from OFF->ON and one
        for the opposite transition from ON->OFF.
    """

    class State(IntEnum):
        OFF = 0
        ON = 1

    @parameters(
        dynamic=[
            "on_threshold",
            "off_threshold",
            "on_value",
            "off_value",
            "initial_state",
        ]
    )
    def __init__(
        self, on_threshold, off_threshold, on_value, off_value, initial_state, **kwargs
    ):
        super().__init__(**kwargs)

        self.declare_default_mode(
            self.State.ON if initial_state == on_value else self.State.OFF
        )

        self.declare_input_port()
        self.declare_output_port(
            self._output,
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.mode],
        )

        # transition to ON event
        def _on_guard(_time, _state, u, **parameters):
            return u - parameters["on_threshold"]

        self.declare_zero_crossing(
            guard=_on_guard,
            direction="negative_then_non_negative",
            start_mode=self.State.OFF,
            end_mode=self.State.ON,
        )

        # transition to OFF event
        def _off_guard(_time, _state, u, **parameters):
            return u - parameters["off_threshold"]

        self.declare_zero_crossing(
            guard=_off_guard,
            direction="positive_then_non_positive",
            start_mode=self.State.ON,
            end_mode=self.State.OFF,
        )

    def initialize(
        self, on_threshold, off_threshold, on_value, off_value, initial_state
    ):
        self.declare_default_mode(
            self.State.ON if initial_state == on_value else self.State.OFF
        )

    def _output(self, _time, state, **parameters):
        on_value = parameters["on_value"]
        off_value = parameters["off_value"]
        return cnp.where(state.mode == self.State.ON, on_value, off_value)


class Ramp(SourceBlock):
    """Output a linear ramp signal in time.

    Given a slope `m`, a start value `y0`, and a start time `t0`, the output signal is:
    ```
        y(t) = m * (t - t0) + y0 if t >= t0 else y0
    ```
    where `t` is the current simulation time.

    Input ports:
        None

    Output ports:
        (0) The ramp signal.

    Parameters:
        start_value:
            The value of the output signal at the start time.
        slope:
            The slope of the ramp signal.
        start_time:
            The time at which the ramp signal begins.
    """

    @parameters(dynamic=["start_value", "slope", "start_time"])
    def __init__(self, start_value=0.0, slope=1.0, start_time=1.0, **kwargs):
        super().__init__(self._func, **kwargs)

    def initialize(self, start_value, slope, start_time):
        pass

    def _func(self, time, **parameters):
        m = parameters["slope"]
        t0 = parameters["start_time"]
        y0 = parameters["start_value"]
        return cnp.where(time >= t0, m * (time - t0) + y0, y0)


class RateLimiter(LeafSystem):
    """Limit the time derivative of the block output.

    Given an input signal `u` computes the derivative of the output signal as:
    ```
        y_rate = (u(t) - y(Tprev))/(t - Tprev)
    ```
    Where Tprev is the last time the block was called for output update.

    When y_rate is greater than the upper_limit, the output is:
    ```
        y(t) = (t - Tprev)*upper_limit + y(Tprev)
    ```

    When y_rate is less than the lower_limit, the output is:
    ```
        y(t) = (t - Tprev)*lower_limit + y(Tprev)
    ```

    If the lower_limit is greater than the upper_limit, and both
    are being violated, the upper_limit takes precedence.

    Optionally, the block can also be configured with "dynamic" limits, which will
    add input ports for time-varying upper and lower limits.

    Presently, the block is constrainted to periodic updates.

    Input ports:
        (0) The input signal.
        (1) The upper limit, if dynamic limits are enabled.
        (2) The lower limit, if dynamic limits are enabled. (Will be indexed as 1 if
            dynamic upper limits are not enabled.)

    Output ports:
        (0) The rate limited output signal.

    Parameters:
        upper_limit:
            The upper limit of the input signal.  Default is `np.inf`.
        enable_dynamic_upper_limit:
            If True, then the upper limit can be set by an external signal. Default
            is False.
        lower_limit:
            The lower limit of the input signal.  Default is `-np.inf`.
        enable_dynamic_lower_limit:
            If True, then the lower limit can be set by an external signal. Default
            is False.
    """

    class DiscreteStateType(NamedTuple):
        y_prev: Array
        t_prev: float

    @parameters(
        static=["enable_dynamic_upper_limit", "enable_dynamic_lower_limit"],
        dynamic=["upper_limit", "lower_limit"],
    )
    def __init__(
        self,
        dt,
        upper_limit=np.inf,
        enable_dynamic_upper_limit=False,
        lower_limit=-np.inf,
        enable_dynamic_lower_limit=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.primary_input_index = self.declare_input_port()
        self.enable_dynamic_upper_limit = enable_dynamic_upper_limit
        self.enable_dynamic_lower_limit = enable_dynamic_lower_limit
        self.dt = dt

        if enable_dynamic_upper_limit:
            # If dynamic limit, simply ignore the static limit
            self.upper_limit_index = self.declare_input_port()

        if enable_dynamic_lower_limit:
            # If dynamic limit, simply ignore the static limit
            self.lower_limit_index = self.declare_input_port()

        self.output_index = self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
        )

    def initialize(
        self,
        upper_limit=np.inf,
        enable_dynamic_upper_limit=False,
        lower_limit=-np.inf,
        enable_dynamic_lower_limit=False,
    ):
        if enable_dynamic_upper_limit != self.enable_dynamic_upper_limit:
            raise ValueError(
                "RateLimiter: enable_dynamic_upper_limit cannot be changed after initialization"
            )
        if enable_dynamic_lower_limit != self.enable_dynamic_lower_limit:
            raise ValueError(
                "RateLimiter: enable_dynamic_lower_limit cannot be changed after initialization"
            )

    def _output(self, time, state, *inputs, **params):
        y_prev = state.cache[self.output_index]

        u = inputs[self.primary_input_index]

        t_diff = self.dt

        y_rate = (u - y_prev) / t_diff

        ulim = (
            inputs[self.upper_limit_index]
            if self.enable_dynamic_upper_limit
            else params["upper_limit"]
        )
        llim = (
            inputs[self.lower_limit_index]
            if self.enable_dynamic_lower_limit
            else params["lower_limit"]
        )
        y_ulim = t_diff * ulim + y_prev
        y_llim = t_diff * llim + y_prev
        y_tmp = cnp.where(y_rate < llim, y_llim, u)
        y = cnp.where(y_rate > ulim, y_ulim, y_tmp)

        return y

    def initialize_static_data(self, context):
        """Infer the size and dtype of the internal states"""
        # If building as part of a subsystem, this may not be fully connected yet.
        # That's fine, as long as it is connected by root context creation time.
        # This probably isn't a good long-term solution:
        #   see https://collimator.atlassian.net/browse/WC-51
        try:
            u = self.eval_input(context)
            self._default_cache[self.output_index] = u
            local_context = context[self.system_id].with_discrete_state(u)
            local_context = local_context.with_cached_value(self.output_index, u)
            context = context.with_subcontext(self.system_id, local_context)

        except UpstreamEvalError:
            logger.debug(
                "RateLimiter.initialize_static_data: UpstreamEvalError. "
                "Continuing without default value initialization."
            )
        return context


class Reciprocal(FeedthroughBlock):
    """Compute the reciprocal of the input signal.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The reciprocal of the input signal: `y = 1 / u`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: 1 / x, *args, **kwargs)


class Saturate(LeafSystem):
    """Clip the input signal to a specified range.

    Given an input signal `u` and upper and lower limits `ulim` and `llim`,
    the output signal is:
    ```
        y = max(llim, min(ulim, u))
    ```
    where `max` and `min` are the element-wise maximum and minimum functions.
    This is equivalent to `y = clip(u, llim, ulim)`.

    Optionally, the block can also be configured with "dynamic" limits, which will
    add input ports for time-varying upper and lower limits.

    Input ports:
        (0) The input signal.
        (1) The upper limit, if dynamic limits are enabled.
        (2) The lower limit, if dynamic limits are enabled. (Will be indexed as 1 if
            dynamic upper limits are not enabled.)

    Output ports:
        (0) The clipped output signal.

    Parameters:
        upper_limit:
            The upper limit of the input signal.  Default is `np.inf`.
        enable_dynamic_upper_limit:
            If True, then the upper limit can be set by an external signal. Default
            is False.
        lower_limit:
            The lower limit of the input signal.  Default is `-np.inf`.
        enable_dynamic_lower_limit:
            If True, then the lower limit can be set by an external signal. Default
            is False.

    Events:
        The block will trigger an event when the input signal crosses either the upper
        or lower limit.  For example, if the block is configured with static upper and
        lower limits and the input signal crosses the upper limit, then a zero-crossing
        event will be triggered.
    """

    @parameters(
        static=["enable_dynamic_upper_limit", "enable_dynamic_lower_limit"],
        dynamic=["upper_limit", "lower_limit"],
    )
    def __init__(
        self,
        upper_limit=None,
        enable_dynamic_upper_limit=False,
        lower_limit=None,
        enable_dynamic_lower_limit=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.primary_input_index = self.declare_input_port()
        self.enable_dynamic_upper_limit = enable_dynamic_upper_limit
        self.enable_dynamic_lower_limit = enable_dynamic_lower_limit

        prerequisites_of_calc = [self.input_ports[self.primary_input_index].ticket]

        if enable_dynamic_upper_limit:
            # If dynamic limit, simply ignore the static limit
            self.upper_limit_index = self.declare_input_port()
            prerequisites_of_calc.append(
                self.input_ports[self.upper_limit_index].ticket
            )
        else:
            if upper_limit is None:
                upper_limit = np.inf

        if enable_dynamic_lower_limit:
            # If dynamic limit, simply ignore the static limit
            self.lower_limit_index = self.declare_input_port()
            prerequisites_of_calc.append(
                self.input_ports[self.lower_limit_index].ticket
            )
        else:
            if lower_limit is None:
                lower_limit = -np.inf

        self.declare_output_port(
            self._compute_output, prerequisites_of_calc=prerequisites_of_calc
        )

    def initialize(
        self,
        upper_limit=None,
        enable_dynamic_upper_limit=False,
        lower_limit=None,
        enable_dynamic_lower_limit=False,
    ):
        if enable_dynamic_lower_limit != self.enable_dynamic_lower_limit:
            raise ValueError(
                "enable_dynamic_lower_limit must be the same as the value passed to the constructor"
            )
        if enable_dynamic_upper_limit != self.enable_dynamic_upper_limit:
            raise ValueError(
                "enable_dynamic_upper_limit must be the same as the value passed to the constructor"
            )

    def _lower_limit_event_value(self, _time, _state, *inputs, **params):
        u = inputs[self.primary_input_index]
        if self.enable_dynamic_lower_limit:
            lim = inputs[self.lower_limit_index]
        else:
            lim = params["lower_limit"]
        return u - lim

    def _upper_limit_event_value(self, _time, _state, *inputs, **params):
        u = inputs[self.primary_input_index]
        if self.enable_dynamic_upper_limit:
            lim = inputs[self.upper_limit_index]
        else:
            lim = params["upper_limit"]
        return u - lim

    def _compute_output(self, _time, _state, *inputs, **params):
        u = inputs[self.primary_input_index]

        ulim = (
            inputs[self.upper_limit_index]
            if self.enable_dynamic_upper_limit
            else params["upper_limit"]
        )
        llim = (
            inputs[self.lower_limit_index]
            if self.enable_dynamic_lower_limit
            else params["lower_limit"]
        )

        return cnp.clip(u, llim, ulim)

    def initialize_static_data(self, context):
        # Add zero-crossing events so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output
        # is fed to an ODE block
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(
                self._lower_limit_event_value,
                direction="positive_then_non_positive",
                name="llim",
            )
            self.declare_zero_crossing(
                self._upper_limit_event_value,
                direction="negative_then_non_negative",
                name="ulim",
            )

        return super().initialize_static_data(context)


class Sawtooth(SourceBlock):
    """Produces a modulated linear sawtooth signal.

    The signal is similar to:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html

    Given amplitude `a`, period `p`, and phase delay `phi`, the output signal is:
    ```
        y(t) = a * ((t - phi) % p)
    ```
    where `%` is the modulo operator.

    Input ports:
        None

    Output ports:
        (0) The sawtooth signal.
    """

    @parameters(dynamic=["amplitude", "frequency", "phase_delay"])
    def __init__(self, amplitude=1.0, frequency=0.5, phase_delay=1.0, **kwargs):
        super().__init__(self._func, **kwargs)

        # Initialize the floating-point tolerance.  This will be machine epsilon
        # for the floating point type of the time variable (determined in the
        # static initialization step).
        self.eps = 0.0
        self._periodic_update_idx = self.declare_periodic_update()

    def initialize(self, amplitude, frequency, phase_delay):
        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        self.declare_discrete_state(default_value=False)

        period = 1 / frequency
        self.configure_periodic_update(
            self._periodic_update_idx,
            lambda *args, **kwargs: True,
            period=period,
            offset=phase_delay,
        )

    def _func(self, time, **parameters):
        # np.mod((t - phase_delay), (1.0 / frequency)) * amplitude
        period = 1 / parameters["frequency"]
        period_fraction = cnp.mod(time - parameters["phase_delay"] + self.eps, period)
        return period_fraction * parameters["amplitude"]

    def initialize_static_data(self, context):
        # Determine machine epsilon for the type of the time variable
        self.eps = 2 * cnp.finfo(cnp.result_type(context.time)).eps
        return super().initialize_static_data(context)


class ScalarBroadcast(FeedthroughBlock):
    """Broadcast a scalar to a vector or matrix.

    Given a scalar input `u` and dimensions `m` and `n`, this block will return
    a vector or matrix of shape `(m, n)` with all elements equal to `u`.

    Input ports:
        (0) The scalar input signal.

    Output ports:
        (0) The broadcasted output signal.

    Parameters:
        m:
            The number of rows in the output matrix.  If `m` is None, then the output
            will be a vector with shape `(n,)`. To get a row vector of size `(1,n)`,
            set `m=1` expliclty.
        n:
            The number of columns in the output matrix.  If `n` is None, then the
            output will be a vector with shape `(m,)`. To get a column vector of size
            `(m,1)`, set `n=1` expliclty.
    """

    @parameters(static=["m", "n"])
    def __init__(self, m, n, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, m, n):
        if m is not None:
            m = int(m)
        else:
            m = 0
        if n is not None:
            n = int(n)
        else:
            n = 0

        if m > 0 and n > 0:
            ones_ = cnp.ones((m, n))
        elif m > 0:
            ones_ = cnp.ones((m,))
        elif n > 0:
            ones_ = cnp.ones((n,))
        else:
            raise BlockParameterError(
                message=f"ScalarBroadcast block {self.name} at least m or n must not be None or Zero"
            )
        self.replace_op(lambda x: ones_ * x)


class Sine(SourceBlock):
    """Generates a sinusoidal signal.

    Given amplitude `a`, frequency `f`, phase `phi`, and bias `b`, the output signal is:
    ```
        y(t) = a * sin(f * t + phi) + b
    ```

    Input ports:
        None

    Output ports:
        (0) The sinusoidal signal.

    Parameters:
        amplitude:
            The amplitude of the sinusoidal signal.
        frequency:
            The frequency of the sinusoidal signal.
        phase:
            The phase of the sinusoidal signal.
        bias:
            The bias of the sinusoidal signal.
    """

    @parameters(dynamic=["amplitude", "frequency", "phase", "bias"])
    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0, bias=0.0, **kwargs):
        super().__init__(self._eval, **kwargs)

    def initialize(self, amplitude=1.0, frequency=1.0, phase=0.0, bias=0.0):
        pass

    def _eval(self, t, **parameters):
        a = parameters["amplitude"]
        f = parameters["frequency"]
        phi = parameters["phase"]
        b = parameters["bias"]
        return a * cnp.sin(f * t + phi) + b


class Slice(FeedthroughBlock):
    """Slice the input signal using Python indexing rules.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The sliced output signal.

    Parameters:
        slice_:
            The slice operator to apply to the input signal.  Must be specified as a
            string input, e.g. the output `u[1:3]` would be created with the block
            `Slice("1:3")`.

    Notes:
        Currently only up to 3-dimensional slices are supported.
    """

    @parameters(static=["slice_"])
    def __init__(self, slice_, *args, **kwargs):
        super().__init__(None, *args, **kwargs)

    def initialize(self, slice_):
        # if slice was provided as numpy slice object, remove this before validating.
        if slice_.startswith("np.s_"):
            slice_ = slice_[len("np.s_") :]
        # if slice is wrapped in [], remove them temporarily.
        if slice_[0] == "[":
            slice_ = slice_[1:]
        if slice_[-1] == "]":
            slice_ = slice_[:-1]

        # validate slice_ and ensure no nefarious code.
        pattern = re.compile(r"^[0-9,:]+$")
        if not pattern.match(slice_):
            raise BlockParameterError(
                message=f"Slice block {self.name} detected invalid slice operator {slice_}. [] are optional. Valid examples: '1:3,4', '[:,4:10]'",
                parameter_name="slice_",
            )
        slice_config = str(slice_)  # copy the string
        self.static_parameters["slice_"].set(slice_config)

        # replace the [] and eval to numpy slcie object
        slice_ = "np.s_[" + slice_ + "]"
        np_slice = eval(slice_)

        def _func(inp):
            return cnp.array(inp)[np_slice]

        self.replace_op(_func)


class SquareRoot(FeedthroughBlock):
    """Compute the square root of the input signal.

    Dispatches to `jax.numpy.sqrt`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sqrt.html

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The square root of the input signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(cnp.sqrt, *args, **kwargs)


class Stack(ReduceBlock):
    """Stack the input signals into a single output signal along a new axis.

    Dispatches to `jax.numpy.stack`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html

    Input ports:
        (0..n_in-1) The input signals.

    Output ports:
        (0) The stacked output signal.

    Parameters:
        axis:
            The axis along which the input signals are stacked.  Default is 0.
    """

    @parameters(static=["axis"])
    def __init__(self, n_in, axis=0, **kwargs):
        super().__init__(n_in, None, **kwargs)

    def initialize(self, axis):
        self.replace_op(partial(cnp.stack, axis=int(axis)))


class Step(SourceBlock):
    """A step signal.

    Given start value `y0`, end value `y1`, and step time `t0`, the
    output signal is:
    ```
        y(t) = y0 if t < t0 else y1
    ```

    Input ports:
        None

    Output ports:
        (0) The step signal.

    Parameters:
        start_value:
            The value of the output signal before the step time.
        end_value:
            The value of the output signal after the step time.
        step_time:
            The time at which the step occurs.
    """

    @parameters(dynamic=["start_value", "end_value", "step_time"])
    def __init__(self, start_value=0.0, end_value=1.0, step_time=1.0, **kwargs):
        super().__init__(self._func, **kwargs)
        self._periodic_update_idx = self.declare_periodic_update()

    def initialize(self, start_value, end_value, step_time):
        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        self.declare_discrete_state(default_value=False)
        self.configure_periodic_update(
            self._periodic_update_idx,
            lambda *args, **kwargs: True,
            period=np.inf,
            offset=step_time,
        )

    def _func(self, time, **parameters):
        return cnp.where(
            time >= parameters["step_time"],
            parameters["end_value"],
            parameters["start_value"],
        )


class Stop(LeafSystem):
    """Stop the simulation early as soon as the input signal becomes True.

    If the input signal changes as a result of a discrete update, the simulation
    will terminate the major step early (before advancing continuous time).

    Input ports:
        (0): the boolean- or binary-valued termination signal

    Output ports:
        None
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.declare_input_port()

        self.declare_zero_crossing(
            guard=self._guard,
            direction="negative_then_non_negative",
            terminal=True,
        )

    def _guard(self, time, state, u, **p):
        return cnp.where(u, 1.0, -1.0)


class SumOfElements(FeedthroughBlock):
    """Compute the sum of the elements of the input signal.

    Dispatches to `jax.numpy.sum`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sum.html

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The sum of the elements of the input signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(cnp.sum, *args, **kwargs)


class Trigonometric(FeedthroughBlock):
    """Apply a trigonometric function to the input signal.

    Available functions are:
        sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh

    Dispatches to `jax.numpy.sin`, `jax.numpy.cos`, etc, so see the JAX docs for details.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The trigonometric function applied to the input signal.

    Parameters:
        function:
            The trigonometric function to apply to the input signal.  Must be one of
            "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
            "asinh", "acosh", "atanh".
    """

    @parameters(static=["function"])
    def __init__(self, function, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, function):
        func_lookup = {
            "sin": cnp.sin,
            "cos": cnp.cos,
            "tan": cnp.tan,
            "asin": cnp.arcsin,
            "acos": cnp.arccos,
            "atan": cnp.arctan,
            "sinh": cnp.sinh,
            "cosh": cnp.cosh,
            "tanh": cnp.tanh,
            "asinh": cnp.arcsinh,
            "acosh": cnp.arccosh,
            "atanh": cnp.arctanh,
        }
        if function not in func_lookup:
            raise BlockParameterError(
                message=f"Trigonometric block {self.name} has invalid selection {function} for 'function'. Valid options: "
                + ", ".join([f for f in func_lookup.keys()]),
                parameter_name="function",
            )
        self.replace_op(func_lookup[function])


class UnitDelay(LeafSystem):
    """Hold and delay the input signal by one time step.

    This block implements a "unit delay" with the following difference equation
    for internal state `x`, input signal `u`, and output signal `y`:
    ```
        x[k+1] = u[k]
        y[k] = x[k]
    ```
    Or, in a hybrid context, the discrete update advances the internal state from
    the "pre" or "minus" value x⁻ to the "post" or "plus" value x⁺ at time
    `tₖ = t0 + k * dt`.  According to the discrete update rules, this calculation
    happens using the input values computed during the update step (i.e. by computing
    upstream outputs before evaluating the inputs to this block). That is, the update
    rule can be written `x⁺(tₖ) = f(tₖ, x⁻(tₖ), u(tₖ))`.  The values of `u` are not
    distinguished as "pre" or "post" because there is only one value at the update
    time.  In the difference equation notation, x⁺(tₖ) ≡ x[k+1]`, `x⁻(tₖ) ≡ x[k],
    and u(tₖ) ≡ u[k].  The hybrid update rule is then:
    ```
        x⁺(tₖ) = u(tₖ)
        y(t) = x⁻(tₖ),       between tₖ⁺ and (tₖ+dt)⁻
    ```

    The output signal "seen" by all other blocks on the time interval (tₖ, tₖ+dt)
    is then the value of the input signal u(tₖ) at the previous update. Therefore, all
    downstream discrete-time blocks updating at the same time tₖ will still see the
    value of x⁻(tₖ), the value of the internal state prior to the update.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal delayed by one time step

    Parameters:
        dt:
            The time step of the discrete update.
        initial_state:
            The initial state of the block.  Default is 0.0.
    """

    @parameters(dynamic=["initial_state"])
    def __init__(self, dt, initial_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, initial_state):
        self.declare_discrete_state(default_value=initial_state)

        self.configure_periodic_update(
            self._periodic_update_idx, self._update, period=self.dt, offset=self.dt
        )

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=0.0,
            requires_inputs=False,
            default_value=initial_state,
            prerequisites_of_calc=[DependencyTicket.xd],
        )

    def _update(self, _time, _state, u, **_params):
        # Every dt seconds, update the state to the current input value
        return u

    def _output(self, _time, state, **parameters):
        return state.discrete_state

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        inp_data = self.eval_input(context)
        xd = context[self.system_id].discrete_state
        check_state_type(
            self,
            inp_data=inp_data,
            state_data=xd,
            error_collector=error_collector,
        )


class ZeroOrderHold(LeafSystem):
    """Implements a "zero-order hold" A/D conversion.

    https://en.wikipedia.org/wiki/Zero-order_hold

    The block implements a "zero-order hold" with the following difference equation
    for input signal `u` and output signal `y`:
    ```
        y[k] = u[k]
    ```

    The block does not maintain an internal state, but simply holds the value of the
    input signal at the previous update time.  As a result, the block is "feedthrough"
    from its inputs to outputs and cannot be used to break an algebraic loop. The data
    type of this hold value is inferred from upstream blocks.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The "hold" value of the input signal.  If the input signal is continuous,
            then the output will be the value of the input signal at the previous
            update time.  If the input signal is discrete and synchonous with the
            block, the output will be the value of the input signal at the current
            time (i.e. identical to the input signal).

    Parameters:
        dt:
            The time step of the discrete update.
    """

    def __init__(self, dt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt

        self.declare_input_port()
        self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
            prerequisites_of_calc=[self.input_ports[0].ticket, DependencyTicket.xd],
        )

    def _output(self, _time, _state, u, **_params):
        # Every dt seconds, update the state to the current input value
        return u


class SignalDatatypeConversion(FeedthroughBlock):
    """Convert the input signal to a different data type.
    Input ports:
        (0) The input signal.
    Output ports:
        (0) The input signal converted to the specified data type.
    Parameters:
        dtype:
            The data type to which the input signal is converted.  Must be a valid
            NumPy data type, e.g. "float32", "int64", etc.
    """

    @parameters(static=["convert_to_type"])
    def __init__(self, convert_to_type, *args, **kwargs):
        self.dtype = np.dtype(convert_to_type)
        super().__init__(
            lambda x: cond(
                isinstance(x, cnp.ndarray),
                lambda x: cnp.astype(x, self.dtype),
                lambda x: cnp.array(x, self.dtype),
                x,
            ),
            *args,
            **kwargs,
        )

    def initialize(self, convert_to_type):
        self.dtype = np.dtype(convert_to_type)
        self.replace_op(
            lambda x: cond(
                isinstance(x, cnp.ndarray),
                lambda x: cnp.astype(x, self.dtype),
                lambda x: cnp.array(x, self.dtype),
                x,
            )
        )
