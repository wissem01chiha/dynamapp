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

"""Utilities for state estimators"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from jax.flatten_util import ravel_pytree

from .. import linearize
from ..utils import make_ode_rhs


def check_shape_compatibilities(A, B, C, D, G, Q, R):
    """
    Check shape compatibilities of matrices in a linear system.

    ```
    x[n+1] = A x[n] + B u[n] + G w[n]
    y[n]   = C x[n] + D u[n] + v[n]

    E(w[n]) = E(v[n]) = 0
    E(w[n]w'[n]) = Q
    E(v[n]v'[n] = R
    E(w[n]v'[n] = N = 0
    ```
    """

    # Retrieve the expected shapes from matrices B and C
    nx, nu = B.shape
    _, nd = G.shape
    ny = C.shape[0]

    # Define expected shapes based on system dynamics
    expected_shapes = {
        "A": (nx, nx),
        "C": (ny, nx),
        "D": (ny, nu),
        "G": (nx, nd),
        "Q": (nd, nd),
        "R": (ny, ny),
    }

    # Dictionary to hold the actual shapes for comparison
    actual_shapes = {
        "A": A.shape,
        "C": C.shape,
        "D": D.shape,
        "G": G.shape,
        "Q": Q.shape,
        "R": R.shape,
    }

    # Iterate through the expected shapes to verify each one
    for matrix_name, expected_shape in expected_shapes.items():
        actual_shape = actual_shapes[matrix_name]
        if actual_shape != expected_shape:
            error_message = (
                f"Shape mismatch for matrix '{matrix_name}': "
                f"expected {expected_shape}, got {actual_shape}."
                "\nThe above was extracted from the shapes of B and C as follows:"
                f"\n- B should have shape (nx, nu) = {B.shape}"
                f"\n- C should have shape (ny, nx) = {C.shape}"
                f"\n- 'nx' is the number of state variables "
                f"\n- 'nu' is the number of control inputs."
                f"\n- 'ny' is the number of measurement outputs."
                f"\n- and 'nd' is the number of disturbances."
            )
            raise ValueError(error_message)


def discretize_forward_zoh(A, B, dt):
    """
    Discretize a continuous-time forward model `dot_x = A x + B u` using zero-order
    hold (ZOH) discretization.

    Parameters:
    A : jnp.ndarray
        The state matrix.
    B : jnp.ndarray
        The input matrix.
    dt : float
        The time step.

    Returns:
    Ad : jnp.ndarray
        The discretized state matrix.
    Bd : jnp.ndarray
        The discretized input matrix.

    The discretized system follows the dynamics:

    x[k+1] = Ad x[k] + Bd u[k]
    """
    Ad = jsp.linalg.expm(A * dt)
    # Assume A is nonsingular (TODO: handle singular A)
    Bd = jnp.linalg.inv(A) @ (Ad - jnp.eye(A.shape[0])) @ B
    return Ad, Bd


def discretize_forward_euler(A, B, dt):
    """
    Discretize a continuous-time forward model `dot_x = A x + B u` using forward Euler
    discretization.

    Parameters:
    A : jnp.ndarray
        The state matrix.
    B : jnp.ndarray
        The input matrix.
    dt : float
        The time step.

    Returns:
    Ad : jnp.ndarray
        The discretized state matrix.
    Bd : jnp.ndarray
        The discretized input matrix.

    The discretized system follows the dynamics:

    x[k+1] = Ad x[k] + Bd u[k]
    """
    nx = A.shape[0]
    Ad = jnp.eye(nx) + A * dt
    Bd = B * dt
    return Ad, Bd


def discretize_Q_zoh(G, Q, A, dt):
    """
    For a linear system `dot_x = A x + B u + G w`, where `w` is a white noise process
    with covariance Q, if discretization is performed using zero-order hold (ZOH) with a
    sampling interval `dt`, then the discretised system is given by:
    ```
    x[k+1] = Ad x[k] + Bd u[k] + w[k],
    ```
    This function computes the covariance `Qd` of `w[k]` using Van Loan's formula [1].

    [1] Van Loan, Charles. Computing integrals involving the matrix exponential.
    IEEE transactions on automatic control, 23.3 (1978): 395-404.
    """
    nx = A.shape[0]
    GQGT = G @ Q @ G.T
    F = dt * jnp.block([[-A, GQGT], [jnp.zeros_like(A), A.T]])
    eF = jsp.linalg.expm(F)
    ur = eF[:nx, nx:]  # upper right part
    lr = eF[nx:, nx:]  # lower right part
    Qd = jnp.matmul(lr.T, ur)
    return 0.5 * (Qd + Qd.T)


def discretize_Q_euler(G, Q, dt):
    """
    For a linear system `dot_x = A x + B u + G w`, where `w` is a white noise process
    with covariance Q, if discretization is performed using a forwar Euler method with
    sampling interval `dt`, then the discretised system is given by:
    ```
    x[k+1] = Ad x[k] + Bd u[k] + w[k],
    ```
    This function computes the covariance `Qd` of `w[k]`.
    """
    GQGT = G @ Q @ G.T
    Qd = GQGT * dt
    return Qd


def linearize_plant(plant, x_eq, u_eq):
    """
    Linearize a plant around provided equilibrium point (x_eq, u_eq)

    Parameters:
    plant : LeafSystem or Diagram representing the plant
            The plant to be linearized.
    x_eq : jnp.ndarray
            The equilibrium point for the plant's state.
    u_eq : jnp.ndarray
            The equilibrium point for the plant's input.

    Returns:
    y_eq : jnp.ndarray
        The plant's output at equilibrium point.
    linear_plant : an `LTISystem` representing the linear plant.
    """
    with plant.input_ports[0].fixed(u_eq):
        base_context = plant.create_context()
        _, unravel = ravel_pytree(base_context.continuous_state)
        eq_context = base_context.with_continuous_state(unravel(x_eq))
        y_eq = plant.output_ports[0].eval(eq_context)
        linear_plant = linearize(plant, eq_context)
    return y_eq, linear_plant


@staticmethod
def linearize_and_discretize_continuous_plant(
    plant,
    x_eq,
    u_eq,
    dt,
    Q=None,
    R=None,
    G=None,  # if None, assume u = u+w, so G = B
    discretization_method="zoh",
    discretized_noise=False,
):
    """
    Utility to linearize and discretize a continuous plant.
    See documentation for methods calling this for more information; for example,
    `make_kalman_filter_for_continuous_plant`, and
    `make_infinite_horizon_kalman_filter_for_continuous_plant`
    """
    y_eq, linear_plant = linearize_plant(plant, x_eq, u_eq)

    A, B, C, D = linear_plant.A, linear_plant.B, linear_plant.C, linear_plant.D

    nx, nu = B.shape
    ny, _ = D.shape

    if G is None:
        G = B

    _, nd = G.shape

    if Q is None:
        Q = jnp.eye(nd)

    if R is None:
        R = jnp.eye(ny)

    # Convert to discrete-time
    Cd, Dd = C, D

    if discretization_method == "euler":
        Ad, Bd = discretize_forward_euler(A, B, dt)
        if discretized_noise:
            Gd = G
            Rd = R
            Qd = Q
        else:
            Gd = jnp.eye(nx)
            Rd = R / dt
            Qd = discretize_Q_euler(G, Q, dt)

    elif discretization_method == "zoh":
        Ad, Bd = discretize_forward_zoh(A, B, dt)
        if discretized_noise:
            Gd = G
            Rd = R
            Qd = Q
        else:
            Gd = jnp.eye(nx)
            Rd = R / dt
            Qd = discretize_Q_zoh(G, Q, A, dt)

    else:
        raise ValueError(
            f"Discretization method {discretization_method} not "
            f" supported. Please use `euler` or `zoh`."
        )

    return (y_eq, Ad, Bd, Cd, Dd, Gd, Qd, Rd)


@staticmethod
def prepare_continuous_plant_for_nonlinear_kalman_filter(
    plant,
    dt,
    G_func,
    Q_func,
    R_func,
    x_hat_0,
    discretization_method,
    discretized_noise,
):
    nu = Q_func(0.0).shape[0]

    nx = x_hat_0.size

    ode_rhs = make_ode_rhs(plant, nu)

    def rhs(x_k, u_k):
        # forward operator of the plant doesn't take time,
        # so use a dummy value for time argument
        return ode_rhs(x_k, u_k, 0.0)

    rhs_jac_x = jax.jacfwd(rhs)

    def forward_euler(x_k, u_k, dt=dt):
        dot_x = rhs(x_k, u_k)
        x_kp1 = x_k + dot_x * dt
        return x_kp1

    def forward_zoh(x_k, u_k, dt=dt):
        dot_x = rhs(x_k, u_k)
        A = rhs_jac_x(x_k, u_k).reshape((nx, nx))
        x_kp1 = x_k + jnp.dot(
            jnp.matmul(jnp.eye(nx) - jsp.linalg.expm(-dt * A), jnp.linalg.inv(A)),
            dot_x,
        )
        return x_kp1

    # generate the unravel utility function
    with plant.input_ports[0].fixed(jnp.zeros(nu)):
        base_context = plant.create_context()
        _, unravel = ravel_pytree(base_context.continuous_state)

    def observation(x_k, u_k):
        with plant.input_ports[0].fixed(u_k):
            base_context = plant.create_context()
            context = base_context.with_continuous_state(unravel(x_k))
            y_k = plant.output_ports[0].eval(context)
        return y_k

    # TODO: `forward` method for an RK4 integrator and/or diffrax ODE solver
    # could be beneficial for unscneted Kalman Filter.

    if discretization_method == "euler":
        forward = forward_euler
        if discretized_noise:
            Gd_func = G_func
            Rd_func = R_func

            def Qd_func(t, x_k, u_k):
                return Q_func(t)

        else:

            def Gd_func(t):
                return jnp.eye(nx)

            def Rd_func(t):
                return R_func(t) / dt

            def Qd_func(t, x_k, u_k):
                G = G_func(t)
                Q = Q_func(t)
                return discretize_Q_euler(G, Q, dt)

    elif discretization_method == "zoh":
        forward = forward_zoh
        if discretized_noise:
            Gd_func = G_func
            Rd_func = R_func

            def Qd_func(t, x_k, u_k):
                return Q_func(t)

        else:

            def Gd_func(t):
                return jnp.eye(nx)

            def Rd_func(t):
                return R_func(t) / dt

            def Qd_func(t, x_k, u_k):
                G = G_func(t)
                Q = Q_func(t)
                A = rhs_jac_x(x_k, u_k).reshape((nx, nx))
                return discretize_Q_zoh(G, Q, A, dt)  # use Van Loan's approximation

    else:
        raise ValueError(
            f"Discretization method {discretization_method} is not "
            f" supported. Please use `euler` or `zoh`."
        )

    return (forward, observation, Gd_func, Qd_func, Rd_func)


def make_global_estimator_from_local(le, x_eq, u_eq, y_eq, name=None, ui_id=None):
    """
        Make a global estimator for a local estimator working around equilibrium point
        (x_eq, u_eq, y_eq). For example, a local estimator could be a Kalman Filter
        that has been obtained for a nonlinear plant linearized at equilibrium point.

        The returned global estimator works with the global (x, u, y) variables, and
        handles transformation between local and global variables internally.

        The returned global estimator:
        (i) Takes the control vector `u` and measurement vector `y` as inputs.
        (ii) transforms `u` and `y` to `u_bar` and `y_bar`, respectively,
             where the variables `u_bar` and `y_bar` are
                u_bar = u - u_eq,
                y_bar = y - y_eq.
        (iii) Applies the local estimator to the transformed variables `u_bar` and
        `y_bar`, obtaining the state estimate `x_hat_bar`.
        (iv) Transforms `x_hat_bar` to `x_hat` and outputs this estimate:
            x_hat = x_hat_bar + x_eq.

        +------------+         +---------+
    u ->|    Adder   |>-u_bar->|         |
        | (u - u_eq) |         |         |              +--------------------+
        +------------+         |   local |              |                    |
                               |estimator|>- x_hat_bar->|       Adder        |>--> x_hat
        +------------+         |         |              | (x_hat_bar + x_eq) |
    y ->|    Adder   |>-x_bar->|         |              |                    |
        | (y - y_eq) |         |         |              +--------------------+
        +------------+         +---------+


        Parameters:
        le : local estimator
            The local estimator, such as a `KalmanFilter` object.
        x_eq : jnp.ndarray
            The equilibrium point for the plant's state.
        u_eq : jnp.ndarray
            The equilibrium point for the plant's input.
        y_eq : jnp.ndarray
            The plant's output at equilibrium point.

        Returns:
        diagram : a `Diagram` object.
            The Diagram for the Kalman filter with:

            Input ports:
                (0) u: control vector
                (1) y: measurement vector

            Output ports:
                (0) x_hat: State vector estimate

            Parameters:
                None
    """
    from collimator import DiagramBuilder
    from collimator.library import IOPort, Constant, Adder

    builder = DiagramBuilder()

    builder.add(le)

    u = builder.add(IOPort(name="u"))
    y = builder.add(IOPort(name="y"))

    u_eq = builder.add(Constant(u_eq, name="u_eq"))
    y_eq = builder.add(Constant(y_eq, name="y_eq"))

    u_bar = builder.add(Adder(2, operators="+-", name="u_bar"))
    y_bar = builder.add(Adder(2, operators="+-", name="y_bar"))

    builder.connect(u.output_ports[0], u_bar.input_ports[0])
    builder.connect(u_eq.output_ports[0], u_bar.input_ports[1])

    builder.connect(y.output_ports[0], y_bar.input_ports[0])
    builder.connect(y_eq.output_ports[0], y_bar.input_ports[1])

    builder.connect(u_bar.output_ports[0], le.input_ports[0])
    builder.connect(y_bar.output_ports[0], le.input_ports[1])

    x_eq = builder.add(Constant(x_eq, name="x_eq"))
    x_hat = builder.add(Adder(2, operators="++", name="x_hat"))

    builder.connect(x_eq.output_ports[0], x_hat.input_ports[0])
    builder.connect(le.output_ports[0], x_hat.input_ports[1])

    builder.export_input(u.input_ports[0])
    builder.export_input(y.input_ports[0])

    builder.export_output(x_hat.input_ports[0])

    diagram = builder.build(name=name, ui_id=ui_id)

    return diagram
