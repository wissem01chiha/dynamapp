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

"""Blocks and utilities for working with linear systems."""

from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp

from ..framework import LeafSystem, parameters
from ..logging import logger
from collimator.lazy_loader import LazyLoader
from collimator.framework.error import StaticError, ErrorCollector
from collimator.backend import numpy_api as cnp

control = LazyLoader(
    "control", globals(), "control"
)  # For formatting state-space systems

signal = LazyLoader(
    "signal", globals(), "scipy.signal"
)  # For converting transfer functions to state-space

__all__ = [
    "LTISystem",
    "LTISystemDiscrete",
    "TransferFunction",
    "TransferFunctionDiscrete",
    "linearize",
    "PID",
    "Derivative",
    "derivative_filter",
]


def _reshape(A, B, C, D):
    # Size checks. UI allows vectors, but here we need matrices, so reshape as needed.
    if len(A.shape) == 1:
        n = 1
        A = A.reshape((n, n))
    else:
        n = A.shape[0]
    if len(B.shape) == 1:
        m = 1
        B = B.reshape((n, m))
    else:
        m = B.shape[1]
    if len(C.shape) == 1:
        p = 1
        C = C.reshape((p, n))
    else:
        p = C.shape[0]
    if len(D.shape) == 1:
        D = D.reshape((p, m))

    assert A.shape == (n, n)
    assert B.shape == (n, m)
    assert C.shape == (p, n)
    assert D.shape == (p, m)

    return A, B, C, D, n, m, p


class LTISystemBase(LeafSystem, ABC):
    """Base class for linear time-invariant systems.

    Parameters:
        A: State matrix of size n x n
        B: Input matrix of size n x m
        C: Output matrix of size p x n
        D: Feedthrough matrix of size p x m
        initialize_states: Initial state vector of size n (default: 0)
    """

    @parameters(dynamic=["initialize_states", "A", "B", "C", "D"])
    def __init__(self, A, B, C, D, initialize_states=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_input_port()  # Single input port (u)

    def initialize(self, A, B, C, D, initialize_states=None, **kwargs):
        (self.A, self.B, self.C, self.D, self.n, self.m, self.p) = _reshape(A, B, C, D)

        self.parameters["A"].set(self.A)
        self.parameters["B"].set(self.B)
        self.parameters["C"].set(self.C)
        self.parameters["D"].set(self.D)

        self.is_feedthrough = not cnp.allclose(D, 0.0)
        self.scalar_output = self.p == 1

        if initialize_states is None:
            initialize_states = cnp.zeros(self.n)
        else:
            initialize_states = cnp.array(initialize_states)
        # Broadcast to size (n,) if only a scalar-valued list, numpy array,
        # jax array, or float is provided at initialization
        if initialize_states.size == 1 and self.n > 1:
            initialize_states = cnp.ones(self.n) * initialize_states.ravel()

        self.initialize_states = initialize_states

    @abstractmethod
    def _eval_output(self, time, state, *inputs, **params):
        """Evaluate the output of the system. Contonuous and discrete systems have
        different implementations."""
        pass

    @property
    @abstractmethod
    def ss(self):
        """State-space representation of the system."""
        pass

    def __repr__(self):
        return self.ss.__repr__()

    def __str__(self):
        return self.ss.__str__()

    def _repr_latex_(self):
        return self.ss._repr_latex_()

    def get_feedthrough(self):
        return [(0, 0)] if self.is_feedthrough else []


class LTISystem(LTISystemBase):
    """Continuous-time linear time-invariant system.

    Implements the following system of ODEs:
    ```
        ẋ = Ax + Bu
        y = Cx + Du
    ```

    Input ports:
        (0) u: Input vector of size m

    Output ports:
        (0) y: Output vector of size p.  Note that this is feedthrough from the input
            port if and only if D is nonzero.

    Parameters:
        A: State matrix of size n x n
        B: Input matrix of size n x m
        C: Output matrix of size p x n
        D: Feedthrough matrix of size p x m
        initialize_states: Initial state vector of size n (default: 0)
    """

    def __init__(self, A, B, C, D, initialize_states=None, *args, **kwargs):
        super().__init__(A, B, C, D, initialize_states, *args, **kwargs)
        self._output_port_idx = self.declare_output_port(
            self._eval_output
        )  # Single output port (y)
        self._continuous_state_id = (
            self.declare_continuous_state()
        )  # Single continuous state (x)

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.configure_output_port(
            self._output_port_idx,
            self._eval_output,
            default_value=cnp.zeros(self.p) if self.p > 1 else 0.0,
            requires_inputs=self.is_feedthrough,
        )
        self.configure_continuous_state(
            self._continuous_state_id,
            ode=self.ode,
            default_value=self.initialize_states,
        )

    def _eval_output(self, time, state, *inputs, **params):
        x = state.continuous_state

        C, D = params["C"], params["D"]
        y = cnp.matmul(C, cnp.atleast_1d(x))

        if self.is_feedthrough:
            (u,) = inputs
            y += cnp.matmul(D, cnp.atleast_1d(u))

        # Handle the special case of scalar output
        if self.scalar_output:
            y = y[0]

        return y

    def ode(self, time, state, u, **params):
        x = state.continuous_state
        A, B = params["A"], params["B"]
        Ax = cnp.matmul(A, cnp.atleast_1d(x))
        Bu = cnp.matmul(B, cnp.atleast_1d(u))
        return Ax + Bu

    @property
    def ss(self):
        """State-space representation of the system."""
        return control.ss(self.A, self.B, self.C, self.D)


class TransferFunction(LTISystem):
    """Continuous-time LTI system specified as a transfer function.

    The transfer function is converted to state-space form using `scipy.signal.tf2ss`.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2ss.html

    The resulting system will be in canonical controller form with matrices
    (A, B, C, D), which are then used to create an LTISystem.  Note that this only
    supports single-input, single-output systems.

    Input ports:
        (0) u: Input vector (scalar)

    Output ports:
        (0) y: Output vector (scalar).  Note that this is feedthrough from the input
            port iff D is nonzero.

    Parameters:
        num: Numerator polynomial coefficients, in descending powers of s
        den: Denominator polynomial coefficients, in descending powers of s
    """

    @parameters(static=["num", "den"])
    def __init__(self, num, den, *args, **kwargs):
        A, B, C, D = signal.tf2ss(num, den)
        super().__init__(A, B, C, D, *args, **kwargs)

    def initialize(self, num, den, **kwargs):
        A, B, C, D = signal.tf2ss(num, den)
        kwargs.update(A=A, B=B, C=C, D=D)
        super().initialize(**kwargs)


class PID(LTISystem):
    """Continuous-time PID controller.

    The PID controller is implemented as a state-space system with matrices
    (A, B, C, D), which are then used to create a (second-order) LTISystem.
    Note that this only supports single-input, single-output PID controllers.

    The PID controller implements the following control law:
    ```
        u = kp * e + ki * ∫e + kd * ė
    ```
    where e is the error signal, and ∫e and ė are the integral and derivative
    of the error signal, respectively.

    With a filter coefficient of `n` (to make the transfer function proper), the
    state-space form of the system is:
    ```
    A = [[0, 1], [0, -n]]
    B = [[0], [1]]
    C = [[ki * n, (kp * n + ki) - (kp + kd * n) * n]]
    D = [[kp + kd * n]]
    ```

    Since D is nonzero, the block is feedthrough.

    Input ports:
        (0) e: Error signal (scalar)

    Output ports:
        (0) u: Control signal (scalar)

    Parameters:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        n: Derivative filter coefficient
        initial_state: Initial state of the integral term (default: 0)
    """

    @parameters(dynamic=["kp", "ki", "kd", "n", "initial_state"])
    def __init__(
        self,
        kp,
        ki,
        kd,
        n,
        initial_state=0.0,
        enable_external_initial_state=False,
        **kwargs,
    ):
        if enable_external_initial_state:
            raise NotImplementedError(
                "External initial state not yet implemented for PID"
            )

        A, B, C, D = self._get_abcd(kp, ki, kd, n)
        initialize_states = cnp.array([initial_state, 0.0])
        super().__init__(A, B, C, D, initialize_states=initialize_states, **kwargs)

    def _get_abcd(self, kp, ki, kd, n):
        A = cnp.array([[0.0, 1.0], [0.0, -n]])
        B = cnp.array([[0.0], [1.0]])
        C = cnp.array([(ki * n), ((kp * n + ki) - (kp + kd * n) * n)])
        D = cnp.array([(kp + kd * n)])
        return A, B, C, D

    def initialize(self, kp, ki, kd, n, initial_state, **kwargs):
        A, B, C, D = self._get_abcd(kp, ki, kd, n)
        initialize_states = cnp.array([initial_state, 0.0])
        kwargs.update(A=A, B=B, C=C, D=D, initialize_states=initialize_states)
        super().initialize(**kwargs)


class Derivative(LTISystem):
    """Causal estimate of the derivative of a signal in continuous time.

    This is implemented as a state-space system with matrices (A, B, C, D),
    which are then used to create a (first-order) LTISystem.  Note that this
    only supports single-input, single-output derivative blocks.

    The derivative is implemented as a filter with a filter coefficient of `N`,
    which is used to construct the following proper transfer function:
    ```
        H(s) = Ns / (s + N)
    ```
    As N -> ∞, the transfer function approaches a pure differentiator.  However,
    this system becomes increasingly stiff and difficult to integrate, so it is
    recommended to select a value of N based on the time scales of the system.

    From the transfer function, `scipy.signal.tf2ss` is used to convert to
    state-space form and create an LTISystem.

    Input ports:
        (0) u: Input (scalar)

    Output ports:
        (0) y: Output (scalar), estimating the time derivative du/dt
    """

    @parameters(dynamic=["filter_coefficient"])
    def __init__(self, filter_coefficient=100, *args, **kwargs):
        N = filter_coefficient
        num = [N, 0]
        den = [1, N]
        A, B, C, D = signal.tf2ss(num, den)
        super().__init__(A, B, C, D, *args, **kwargs)

    def initialize(self, filter_coefficient, **kwargs):
        N = filter_coefficient
        num = [N, 0]
        den = [1, N]

        A, B, C, D = signal.tf2ss(num, den)
        kwargs.update(A=A, B=B, C=C, D=D)
        super().initialize(**kwargs)

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        inputs = self.collect_inputs(context)
        (u,) = inputs

        if not cnp.ndim(u) == 0:
            with ErrorCollector.context(error_collector):
                raise StaticError(
                    message="Derivative must have scalar input.",
                    system=self,
                )


def linearize(system, base_context, name=None, output_index=None):
    """Linearize the system about an operating point specified by the base context.

    For now, only implemented for systems with one each (vector-valued) input and
    output. The system may have multiple output ports, but only one will be treated
    as the measurement.
    """
    assert len(system.input_ports) == 1, (
        "Linearization only implemented for systems with one input port, system "
        f"{system.name} has {len(system.input_ports)} input ports"
    )
    if len(system.output_ports) > 1:
        if output_index is None:
            logger.warning(
                "Multiple output ports detected when linearizing system %s, "
                "using first port as output",
                system.name,
            )

    # Default to zero output index if not specified (after issuing a warning)
    if output_index is None:
        output_index = 0

    input_port = system.input_ports[0]
    output_port = system.output_ports[output_index]

    xc0 = base_context.continuous_state
    u0 = input_port.eval(base_context)

    restore_fixed_val = input_port.is_fixed

    # Map from (state, inputs) to (state derivatives, outputs)
    @jax.jit
    def f(xc, u):
        context = base_context.with_continuous_state(xc)
        with input_port.fixed(u):
            xdot = system.eval_time_derivatives(context)
            y = output_port.eval(context)
        return xdot, y

    @jax.jit
    def jac(xc, u):
        primals, tangents = jax.jvp(f, (xc0, u0), (xc, u))
        return tangents

    lin_sys = LTISystem(*_jvp_to_ss(jac, xc0, u0), name=name)

    if restore_fixed_val:
        input_port.fix_value(u0)

    return lin_sys


def derivative_filter(N, dt, filter_type="forward"):
    """Discrete filter coefficients that approximate a time derivative.

    filter_type: "forward", "backward", "bilinear", or "none"

    This is used by the DerivativeDiscrete and PIDDiscrete blocks to
    create recursive filters for estimating the derivative of a signal.
    """

    # Unfiltered forward Euler (finite differencing)
    if filter_type == "none":
        b = [1, -1]
        a = [dt, 0]
    # Filtered forward Euler
    elif filter_type == "forward":
        b = [N, -N]
        a = [1, (N * dt - 1)]
    # Filtered backward Euler
    elif filter_type == "backward":
        b = [N, -N]
        a = [(1 + N * dt), -1]
    # Filtered bilinear transform
    elif filter_type == "bilinear":
        b = [2 * N, -2 * N]
        a = [(2 + N * dt), (-2 + N * dt)]
    else:
        raise ValueError(f"Unknown filter type {filter_type}")

    return b, a


@partial(jax.jit, static_argnums=(0,))
def _jvp_to_ss(J, xc0, u0):
    """Create a state-space realization from a function that evaluates the jacobian-vector product of a system.

    The ordering is not unique, but is deterministically chosen by the order of the flattened state vector produced
    by `ravel_pytree`."""
    from jax.flatten_util import ravel_pytree

    vec, unflatten = ravel_pytree(xc0)
    _, y0 = J(xc0, u0)  # One forward evaluation

    n = vec.size
    p = u0.size
    m = y0.size
    A = jnp.zeros((n, n))
    B = jnp.zeros((n, p))
    C = jnp.zeros((m, n))
    D = jnp.zeros((m, p))

    ucp = jnp.zeros_like(u0)
    for i in range(n):
        xcp = jnp.zeros_like(vec).at[i].set(1.0)
        xdot, y = J(unflatten(xcp), ucp)
        xdot, _ = ravel_pytree(xdot)
        y, _ = ravel_pytree(y)
        A = A.at[:, i].set(xdot)
        C = C.at[:, i].set(y)

    xcp = jnp.zeros_like(vec)
    for i in range(p):
        ucp = jnp.zeros_like(u0).at[i].set(1.0)
        xdot, y = J(unflatten(xcp), ucp)
        xdot, _ = ravel_pytree(xdot)
        y, _ = ravel_pytree(y)
        B = B.at[:, i].set(xdot)
        D = D.at[:, i].set(y)

    return (A, B, C, D)


class LTISystemDiscrete(LTISystemBase):
    """Discrete-time linear time-invariant system.

    Implements the following system of ODEs:
    ```
        x[k+1] = A x[k] + B u[k]
        y[k] = C x[k] + D u[k]
    ```

    Input ports:
        (0) u[k]: Input vector of size m

    Output ports:
        (0) y[k]: Output vector of size p.  Note that this is feedthrough from the
                  input port if and only if D is nonzero.

    Parameters:
        A: State matrix of size n x n
        B: Input matrix of size n x m
        C: Output matrix of size p x n
        D: Feedthrough matrix of size p x m
        dt: Sampling period
        initialize_states: Initial state vector of size n (default: 0)
    """

    def __init__(self, A, B, C, D, dt, initialize_states=None, *args, **kwargs):
        super().__init__(A, B, C, D, initialize_states, *args, **kwargs)

        self.dt = dt
        self.declare_periodic_update(
            self._update,
            period=dt,
            offset=0.0,
        )

        self._output_port_idx = self.declare_output_port(
            self._eval_output
        )  # Single output port (y)

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.declare_discrete_state(
            default_value=self.initialize_states,
        )  # Single discrete state (x)
        self.configure_output_port(
            self._output_port_idx,
            self._eval_output,
            period=self.dt,
            offset=0.0,
            default_value=jnp.zeros(self.p) if self.p > 1 else 0.0,
            requires_inputs=self.is_feedthrough,
        )

    def _eval_output(self, time, state, *inputs, **params):
        x = state.discrete_state
        C, D = params["C"], params["D"]
        y = jnp.matmul(C, jnp.atleast_1d(x))

        if self.is_feedthrough:
            (u,) = inputs
            y += jnp.matmul(D, jnp.atleast_1d(u))

        # Handle the special case of scalar output
        if self.scalar_output:
            y = y[0]

        return y

    def _update(self, time, state, u, **params):
        x = state.discrete_state
        A, B = params["A"], params["B"]
        Ax = jnp.matmul(A, jnp.atleast_1d(x))
        Bu = jnp.matmul(B, jnp.atleast_1d(u))
        return Ax + Bu

    @property
    def ss(self):
        """State-space representation of the system."""
        return control.ss(self.A, self.B, self.C, self.D, self.dt)


class TransferFunctionDiscrete(LTISystemDiscrete):
    """Implements a Discrete Time Transfer Function.

    https://en.wikipedia.org/wiki/Z-transform#Transfer_function

    The resulting system will be in canonical controller form with matrices
    (A, B, C, D), which are then used to create an LTISystem.  Note that this only
    supports single-input, single-output systems.

    Input ports:
        (0) u[k]: Input vector (scalar)

    Output ports:
        (0) y[k]: Output vector (scalar). Note that this is feedthrough from the input
            port if and only if D is nonzero.

    Parameters:
        dt:
            Sampling period of the discrete system.
        num:
            Numerator polynomial coefficients, in descending powers of z
        den:
            Denominator polynomial coefficients, in descending powers of z
        initialize_states:
            Initial state vector (default: 0)
    """

    @parameters(static=["num", "den"])
    def __init__(self, dt, num, den, initialize_states=None, *args, **kwargs):
        A, B, C, D = signal.tf2ss(num, den)
        super().__init__(A, B, C, D, dt, initialize_states, *args, **kwargs)

    def initialize(self, num, den, **kwargs):
        A, B, C, D = signal.tf2ss(num, den)
        kwargs.update(A=A, B=B, C=C, D=D)
        super().initialize(**kwargs)
