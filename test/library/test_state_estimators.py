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

"""Test functions for Kalman filter and variants"""

from math import ceil
import pytest

import control as pycontrol
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy as jsp

import collimator

from collimator.framework import LeafSystem
from collimator.simulation import SimulatorOptions

from collimator.library import (
    Constant,
    Adder,
    IOPort,
    ZeroOrderHold,
    WhiteNoise,
    DotProduct,
    linearize,
    KalmanFilter,
    ContinuousTimeInfiniteHorizonKalmanFilter,
    InfiniteHorizonKalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
)


class Pendulum(LeafSystem):
    """
    A simple pendulum model with damping. State is [theta, omega], and output is
    only theta
    """

    def __init__(self, *args, x0=jnp.zeros(2), m=1.0, L=1.0, d=0.2, g=9.81, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_dynamic_parameter("m", m)
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("d", d)
        self.declare_dynamic_parameter("g", g)

        self.declare_input_port(name="tau")

        self.declare_continuous_state(shape=(2,), ode=self.ode, default_value=x0)

        self.declare_output_port(self._eval_output)

    def _eval_output(self, time, state, *inputs, **parameters):
        theta, omega = state.continuous_state
        return theta

    def ode(self, time, state, *inputs, **parameters):
        (u,) = inputs
        theta, omega = state.continuous_state

        m = parameters["m"]
        L = parameters["L"]
        d = parameters["d"]
        g = parameters["g"]
        mL2 = m * L**2

        dot_theta = omega
        dot_omega = u / mL2 - d * omega / mL2 - g * jnp.sin(theta) / L

        return jnp.array([dot_theta, dot_omega[0]])


def make_disturbance_from_noise_covariance(covariance, name=None):
    n = covariance.shape[0]
    chol_cov = jnp.linalg.cholesky(covariance)
    builder = collimator.DiagramBuilder()
    fs = 10.0
    unit_noise = builder.add(
        WhiteNoise(
            correlation_time=1.0 / fs, noise_power=1.0, shape=(n,), name="unit_noise"
        )
    )
    chol_noise = builder.add(Constant(chol_cov, name="chol_cov"))
    noise = builder.add(DotProduct(name="noise"))
    builder.connect(chol_noise.output_ports[0], noise.input_ports[0])
    builder.connect(unit_noise.output_ports[0], noise.input_ports[1])
    builder.export_output(noise.output_ports[0])
    diagram = builder.build(name=name)
    return diagram


def make_pendulum_with_disturbances(
    x0,
    Q,
    R,
    config=None,
):
    """
    Returns a diagram with a pendulum and disturbances.
    Process noise with covariance Q is added to the input torque of the pendulum.
    Measurement noise R is added to the angle theta of the pendulum to generate
    the measurement.
    """
    builder = collimator.DiagramBuilder()
    if config is None:
        pendulum = builder.add(Pendulum(x0=x0, name="pendulum"))
    else:
        pendulum = builder.add(Pendulum(x0=x0, **config, name="pendulum"))

    process_noise = builder.add(
        make_disturbance_from_noise_covariance(covariance=Q, name="process_noise")
    )
    measurement_noise = builder.add(
        make_disturbance_from_noise_covariance(covariance=R, name="measurement_noise")
    )

    noisy_torque = builder.add(Adder(2, name="noisy_torque"))
    input_torque = builder.add(IOPort(name="input_torque"))
    theta_measured = builder.add(Adder(2, name="theta_measured"))

    builder.connect(noisy_torque.output_ports[0], pendulum.input_ports[0])

    builder.connect(process_noise.output_ports[0], noisy_torque.input_ports[0])
    builder.connect(input_torque.output_ports[0], noisy_torque.input_ports[1])

    builder.connect(measurement_noise.output_ports[0], theta_measured.input_ports[0])
    builder.connect(pendulum.output_ports[0], theta_measured.input_ports[1])

    builder.export_input(input_torque.input_ports[0])
    builder.export_output(theta_measured.output_ports[0])

    diagram = builder.build(name="pendulum_with_disturbances")
    return diagram


@pytest.mark.parametrize(
    "discretization_method, discrete_time_plant",
    [
        # ("zoh", False),
        # ("euler", False),
        ("zoh", True),
        ("euler", True),
    ],
)
def test_kalman_filter(discretization_method, discrete_time_plant, plot=False):
    nx = 2
    nu = 1
    ny = 1

    Q = 1.0e-01 * jnp.eye(nu)  # process noise
    R = 1.0e-02 * jnp.eye(ny)  # measurement noise

    u_eq = jnp.array([0.0])
    x_eq = jnp.array([0.0, 0.0])

    x0 = jnp.array([jnp.pi / 20, 0.0])
    x_hat_bar_0 = jnp.array([jnp.pi / 15.0, 0.1])
    P_hat_bar_0 = 0.01 * jnp.eye(nx)

    dt = 0.01

    builder = collimator.DiagramBuilder()

    pendulum = builder.add(make_pendulum_with_disturbances(x0, Q, R))

    _, kf_bar = KalmanFilter.for_continuous_plant(
        Pendulum(x0=x0, name="pendulum"),
        x_eq,
        u_eq,
        dt,
        Q,
        R,
        G=None,  # if None, assumes u = u+w, so G = B
        x_hat_bar_0=x_hat_bar_0,
        P_hat_bar_0=P_hat_bar_0,
        discretization_method=discretization_method,
        discretized_noise=False,
    )

    kf = builder.add(kf_bar)

    control = builder.add(Constant(jnp.array([0.0]), name="control"))

    if not discrete_time_plant:
        builder.connect(control.output_ports[0], kf.input_ports[0])
        builder.connect(pendulum.output_ports[0], kf.input_ports[1])
    else:
        zoh_u = builder.add(ZeroOrderHold(dt, name="zoh_u"))
        zoh_y = builder.add(ZeroOrderHold(dt, name="zoh_y"))
        builder.connect(control.output_ports[0], zoh_u.input_ports[0])
        builder.connect(pendulum.output_ports[0], zoh_y.input_ports[0])
        builder.connect(zoh_u.output_ports[0], kf.input_ports[0])
        builder.connect(zoh_y.output_ports[0], kf.input_ports[1])

    builder.connect(control.output_ports[0], pendulum.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    recorded_signals = {
        "x_true": diagram["pendulum_with_disturbances"]["pendulum"].output_ports[0],
        "x_hat": kf.output_ports[0],
    }

    if not discrete_time_plant:
        recorded_signals["theta_measured"] = pendulum.output_ports[0]
    else:
        recorded_signals["theta_measured"] = zoh_y.output_ports[0]

    Tsolve = 1.0
    nseg = ceil(Tsolve / dt)
    options = SimulatorOptions(
        max_major_steps=10 * nseg,
        max_major_step_length=dt,
    )

    sol = collimator.simulate(
        diagram,
        context,
        (0.0, Tsolve),
        options=options,
        recorded_signals=recorded_signals,
    )

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5))
        ax1.plot(
            sol.time,
            sol.outputs["theta_measured"],
            label=r"$\theta_\mathrm{meas}$",
            alpha=0.5,
        )
        ax1.plot(sol.time, sol.outputs["x_true"], label=r"$\theta_\mathrm{true}$")
        ax1.plot(sol.time, sol.outputs["x_hat"][:, 0], label=r"$\theta_\mathrm{kf}$")
        ax2.plot(sol.time, sol.outputs["x_hat"][:, 1], label=r"$\omega_\mathrm{kf}$")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.show()

    # Linearise
    u_eq = jnp.array([0.0])
    x_eq = jnp.array([0.0, 0.0])

    pendulum = Pendulum(x0=x0, name="pendulum")

    pendulum.input_ports[0].fix_value(u_eq)
    base_context = pendulum.create_context()

    eq_context = base_context.with_continuous_state(x_eq)

    linear_pendulum = linearize(pendulum, eq_context)

    A, B, C, D = (
        linear_pendulum.A,
        linear_pendulum.B,
        linear_pendulum.C,
        linear_pendulum.D,
    )

    G = B

    # Convert to discrete-time
    Cd = C
    Dd = D
    Gd = jnp.eye(nx)
    Rd = R / dt

    if discretization_method == "zoh":
        Ad = jsp.linalg.expm(A * dt)
        Bd = jnp.linalg.inv(A) @ (Ad - jnp.eye(nx)) @ B

        GQGT = G @ Q @ G.T
        F = dt * jnp.block([[-A, GQGT], [jnp.zeros_like(A), A.T]])
        EF = jsp.linalg.expm(F)
        ur = EF[:nx, nx:]
        lr = EF[nx:, nx:]
        Qd = (lr.T) @ ur
        Qd = 0.5 * (Qd + Qd.T)
    elif discretization_method == "euler":
        Ad = jnp.eye(nx) + A * dt
        Bd = B * dt

        GQGT = G @ Q @ G.T
        Qd = GQGT * dt
    else:
        raise ValueError("Invalid discretization method")

    # Compute KF solution
    regular_times = dt * jnp.arange(nseg)
    indices = jnp.searchsorted(sol.time, regular_times, side="right") - 1

    x_hat_minus = x_hat_bar_0
    P_hat_minus = P_hat_bar_0
    eye = jnp.eye(nx)
    u = jnp.array([0.0])

    wc_sol = []
    kf_sol = []

    GdQGdT = jnp.matmul(Gd, jnp.matmul(Qd, Gd.T))
    for idx in indices:
        # Correct at tcurr
        y = sol.outputs["theta_measured"][idx]
        K = P_hat_minus @ Cd.T @ jnp.linalg.inv(Cd @ P_hat_minus @ Cd.T + Rd)

        x_hat_plus = x_hat_minus + jnp.dot(
            K, y - jnp.dot(Cd, x_hat_minus) - jnp.dot(Dd, u)
        )  # k|k
        P_hat_plus = jnp.matmul(eye - jnp.matmul(K, Cd), P_hat_minus)  # k|k

        kf_sol.append(x_hat_plus)
        wc_sol.append(sol.outputs["x_hat"][idx])

        # propagate to tnext
        x_hat_minus = jnp.dot(Ad, x_hat_plus) + jnp.dot(Bd, u)  # k+1|k
        P_hat_minus = jnp.matmul(Ad, jnp.matmul(P_hat_plus, Ad.T)) + GdQGdT  # k+1|k

    # Compare KF solution with wildcat solution
    # FIXME: https://collimator.atlassian.net/browse/WC-387
    # assert jnp.allclose(jnp.array(kf_sol), jnp.array(wc_sol))


@pytest.mark.parametrize(
    "discretization_method, discrete_time_plant",
    [
        ("zoh", False),
        ("euler", False),
        ("zoh", True),
        ("euler", True),
    ],
)
def test_infinite_horizon_kalman_filter(
    discretization_method, discrete_time_plant, plot=False
):
    nx = 2
    nu = 1
    ny = 1

    Q = 1.0e-01 * jnp.eye(nu)  # process noise
    R = 1.0e-02 * jnp.eye(ny)  # measurement noise

    u_eq = jnp.array([0.0])
    x_eq = jnp.array([0.0, 0.0])

    x0 = jnp.array([jnp.pi / 20, 0.0])
    x_hat_bar_0 = jnp.array([jnp.pi / 15.0, 0.1])

    dt = 0.01

    builder = collimator.DiagramBuilder()

    pendulum = builder.add(make_pendulum_with_disturbances(x0, Q, R))

    (
        _,
        kf_bar,
    ) = InfiniteHorizonKalmanFilter.for_continuous_plant(
        Pendulum(x0=x0, name="pendulum"),
        x_eq,
        u_eq,
        dt,
        Q,
        R,
        G=None,  # if None, assumes u = u+w, so G = B
        x_hat_bar_0=x_hat_bar_0,
        discretization_method=discretization_method,
        discretized_noise=False,
    )

    kf = builder.add(kf_bar)

    control = builder.add(Constant(jnp.array([0.0]), name="control"))

    if not discrete_time_plant:
        builder.connect(control.output_ports[0], kf.input_ports[0])
        builder.connect(pendulum.output_ports[0], kf.input_ports[1])
    else:
        zoh_u = builder.add(ZeroOrderHold(dt, name="zoh_u"))
        zoh_y = builder.add(ZeroOrderHold(dt, name="zoh_y"))
        builder.connect(control.output_ports[0], zoh_u.input_ports[0])
        builder.connect(pendulum.output_ports[0], zoh_y.input_ports[0])
        builder.connect(zoh_u.output_ports[0], kf.input_ports[0])
        builder.connect(zoh_y.output_ports[0], kf.input_ports[1])

    builder.connect(control.output_ports[0], pendulum.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    recorded_signals = {
        "x_true": diagram["pendulum_with_disturbances"]["pendulum"].output_ports[0],
        "x_hat": kf.output_ports[0],
    }

    if not discrete_time_plant:
        recorded_signals["theta_measured"] = pendulum.output_ports[0]
    else:
        recorded_signals["theta_measured"] = zoh_y.output_ports[0]

    Tsolve = 1.0
    nseg = ceil(Tsolve / dt)
    options = SimulatorOptions(
        max_major_steps=10 * nseg,
        max_major_step_length=dt,
    )

    sol = collimator.simulate(
        diagram,
        context,
        (0.0, Tsolve),
        options=options,
        recorded_signals=recorded_signals,
    )

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5))
        ax1.plot(
            sol.time,
            sol.outputs["theta_measured"],
            label=r"$\theta_\mathrm{meas}$",
            alpha=0.5,
        )
        ax1.plot(sol.time, sol.outputs["x_true"], label=r"$\theta_\mathrm{true}$")
        ax1.plot(sol.time, sol.outputs["x_hat"][:, 0], label=r"$\theta_\mathrm{kf}$")
        ax2.plot(sol.time, sol.outputs["x_hat"][:, 1], label=r"$\omega_\mathrm{kf}$")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.show()

    # Linearise
    u_eq = jnp.array([0.0])
    x_eq = jnp.array([0.0, 0.0])

    pendulum = Pendulum(x0=x0, name="pendulum")

    pendulum.input_ports[0].fix_value(u_eq)
    base_context = pendulum.create_context()

    eq_context = base_context.with_continuous_state(x_eq)

    linear_pendulum = linearize(pendulum, eq_context)

    A, B, C, D = (
        linear_pendulum.A,
        linear_pendulum.B,
        linear_pendulum.C,
        linear_pendulum.D,
    )

    G = B

    # Convert to discrete-time
    Cd = C
    Dd = D
    Gd = jnp.eye(nx)
    Rd = R / dt

    if discretization_method == "zoh":
        Ad = jsp.linalg.expm(A * dt)
        Bd = jnp.linalg.inv(A) @ (Ad - jnp.eye(nx)) @ B

        GQGT = G @ Q @ G.T
        F = dt * jnp.block([[-A, GQGT], [jnp.zeros_like(A), A.T]])
        EF = jsp.linalg.expm(F)
        ur = EF[:nx, nx:]
        lr = EF[nx:, nx:]
        Qd = (lr.T) @ ur
        Qd = 0.5 * (Qd + Qd.T)
    elif discretization_method == "euler":
        Ad = jnp.eye(nx) + A * dt
        Bd = B * dt

        GQGT = G @ Q @ G.T
        Qd = GQGT * dt
    else:
        raise ValueError("Invalid discretization method")

    # Compute KF solution
    regular_times = dt * jnp.arange(nseg)
    indices = jnp.searchsorted(sol.time, regular_times, side="right") - 1

    x_hat_minus = x_hat_bar_0
    u = jnp.array([0.0])

    wc_sol = []
    kf_sol = []

    L, P, E = pycontrol.dlqe(Ad, Gd, Cd, Qd, Rd)

    K = jnp.linalg.solve(Ad, L)

    A_minus_LC = Ad - jnp.matmul(L, Cd)
    B_minus_LD = Bd - jnp.matmul(L, Dd)

    for idx in indices:
        # Correct at tcurr
        y = sol.outputs["theta_measured"][idx]

        x_hat_plus = x_hat_minus + jnp.dot(
            K, y - jnp.dot(Cd, x_hat_minus) - jnp.dot(Dd, u)
        )  # k|k

        # print(f"{x_hat_plus=}")
        # print("wildcat kf sol =", sol.outputs["x_hat"][idx])

        kf_sol.append(x_hat_plus)
        wc_sol.append(sol.outputs["x_hat"][idx])

        # propagate to tnext
        x_hat_minus = (
            jnp.dot(A_minus_LC, x_hat_plus) + jnp.dot(B_minus_LD, u) + jnp.dot(L, y)
        )

    # Compare KF solution with wildcat solution
    # FIXME: https://collimator.atlassian.net/browse/WC-387
    # assert jnp.allclose(jnp.array(kf_sol), jnp.array(wc_sol))


def test_continuous_time_infinite_horizon_kalman_filter(plot=False):
    nu = 1
    ny = 1

    Q = 1.0e-01 * jnp.eye(nu)  # process noise
    R = 1.0e-02 * jnp.eye(ny)  # measurement noise

    u_eq = jnp.array([0.0])
    x_eq = jnp.array([0.0, 0.0])

    x0 = jnp.array([jnp.pi / 20, 0.0])
    x_hat_bar_0 = jnp.array([jnp.pi / 15.0, 0.1])

    builder = collimator.DiagramBuilder()

    pendulum = builder.add(make_pendulum_with_disturbances(x0, Q, R))

    _, kf_bar = ContinuousTimeInfiniteHorizonKalmanFilter.for_continuous_plant(
        Pendulum(x0=x0, name="pendulum"),
        x_eq,
        u_eq,
        Q,
        R,
        G=None,  # if None, assumes u = u+w, so G = B
        x_hat_bar_0=x_hat_bar_0,
    )

    kf = builder.add(kf_bar)

    control = builder.add(Constant(jnp.array([0.0]), name="control"))

    builder.connect(control.output_ports[0], kf.input_ports[0])
    builder.connect(pendulum.output_ports[0], kf.input_ports[1])

    builder.connect(control.output_ports[0], pendulum.input_ports[0])

    diagram = builder.build()

    context = diagram.create_context()

    recorded_signals = {
        "x_true": diagram["pendulum_with_disturbances"]["pendulum"].output_ports[0],
        "theta_measured": pendulum.output_ports[0],
        "x_hat": kf.output_ports[0],
    }

    dt = 0.01
    Tsolve = 1.0
    nseg = ceil(Tsolve / dt)
    options = SimulatorOptions(
        max_major_steps=10 * nseg,
        max_major_step_length=dt,
    )

    sol = collimator.simulate(
        diagram,
        context,
        (0.0, Tsolve),
        options=options,
        recorded_signals=recorded_signals,
    )

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5))
        ax1.plot(
            sol.time,
            sol.outputs["theta_measured"],
            label=r"$\theta_\mathrm{meas}$",
            alpha=0.5,
        )
        ax1.plot(sol.time, sol.outputs["x_true"], label=r"$\theta_\mathrm{true}$")
        ax1.plot(sol.time, sol.outputs["x_hat"][:, 0], label=r"$\theta_\mathrm{kf}$")
        ax2.plot(sol.time, sol.outputs["x_hat"][:, 1], label=r"$\omega_\mathrm{kf}$")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.show()


@pytest.mark.parametrize(
    "discretization_method",
    [
        "zoh",
        "euler",
    ],
)
def test_extended_kalman_filter_discretization_methods(
    discretization_method, plot=False
):
    config = {"m": 1.0, "L": 1.0, "d": 0.2, "g": 9.81}
    nx = 2
    nu = 1
    ny = 1

    Q = 1.0e-01 * jnp.eye(nu)  # process noise
    R = 1.0e-02 * jnp.eye(ny)  # measurement noise
    G = jnp.array([[0.0], [1.0 / config["m"] / config["L"] ** 2]])

    x0 = jnp.array([jnp.pi / 2.0, 0.0])
    x_hat_0 = jnp.array([jnp.pi / 3.0, 0.0])
    P_hat_0 = 1.0 * jnp.eye(nx)

    dt = 0.01

    builder = collimator.DiagramBuilder()

    pendulum = builder.add(make_pendulum_with_disturbances(x0, Q, R, config))

    kf = builder.add(
        ExtendedKalmanFilter.for_continuous_plant(
            Pendulum(x0=x0, **config, name="pendulum"),
            dt,
            G_func=lambda t: G,
            Q_func=lambda t: Q,
            R_func=lambda t: R,
            x_hat_0=x_hat_0,
            P_hat_0=P_hat_0,
            discretization_method=discretization_method,
            name=None,
        )
    )

    control = builder.add(Constant(jnp.array([0.0]), name="control"))

    builder.connect(control.output_ports[0], kf.input_ports[0])
    builder.connect(pendulum.output_ports[0], kf.input_ports[1])

    builder.connect(control.output_ports[0], pendulum.input_ports[0])

    diagram = builder.build()

    context = diagram.create_context()

    recorded_signals = {
        "x_true": diagram["pendulum_with_disturbances"]["pendulum"].output_ports[0],
        "theta_measured": pendulum.output_ports[0],
        "x_hat": kf.output_ports[0],
    }

    dt = 0.01
    Tsolve = 1.0
    nseg = ceil(Tsolve / dt)
    options = SimulatorOptions(
        max_major_steps=10 * nseg,
        max_major_step_length=dt,
    )

    sol = collimator.simulate(
        diagram,
        context,
        (0.0, Tsolve),
        options=options,
        recorded_signals=recorded_signals,
    )

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5))
        ax1.plot(
            sol.time,
            sol.outputs["theta_measured"],
            label=r"$\theta_\mathrm{meas}$",
            alpha=0.5,
        )
        ax1.plot(sol.time, sol.outputs["x_true"], label=r"$\theta_\mathrm{true}$")
        ax1.plot(sol.time, sol.outputs["x_hat"][:, 0], label=r"$\theta_\mathrm{kf}$")
        ax2.plot(sol.time, sol.outputs["x_hat"][:, 1], label=r"$\omega_\mathrm{kf}$")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.show()


@pytest.mark.parametrize(
    "discretization_method",
    [
        "zoh",
        "euler",
    ],
)
def test_unscented_kalman_filter_discretization_methods(
    discretization_method, plot=False
):
    config = {"m": 1.0, "L": 1.0, "d": 0.2, "g": 9.81}
    nx = 2
    nu = 1
    ny = 1

    Q = 1.0e-01 * jnp.eye(nu)  # process noise
    R = 1.0e-02 * jnp.eye(ny)  # measurement noise
    G = jnp.array([[0.0], [1.0 / config["m"] / config["L"] ** 2]])

    x0 = jnp.array([jnp.pi / 2.0, 0.0])
    x_hat_0 = jnp.array([jnp.pi / 3.0, 0.0])
    P_hat_0 = 1.0 * jnp.eye(nx)

    dt = 0.01

    builder = collimator.DiagramBuilder()

    pendulum = builder.add(make_pendulum_with_disturbances(x0, Q, R, config))

    kf = builder.add(
        UnscentedKalmanFilter.for_continuous_plant(
            Pendulum(x0=x0, **config, name="pendulum"),
            dt,
            G_func=lambda t: G,
            Q_func=lambda t: Q,
            R_func=lambda t: R,
            x_hat_0=x_hat_0,
            P_hat_0=P_hat_0,
            discretization_method=discretization_method,
            name=None,
        )
    )

    control = builder.add(Constant(jnp.array([0.0]), name="control"))

    builder.connect(control.output_ports[0], kf.input_ports[0])
    builder.connect(pendulum.output_ports[0], kf.input_ports[1])

    builder.connect(control.output_ports[0], pendulum.input_ports[0])

    diagram = builder.build()

    context = diagram.create_context()

    recorded_signals = {
        "x_true": diagram["pendulum_with_disturbances"]["pendulum"].output_ports[0],
        "theta_measured": pendulum.output_ports[0],
        "x_hat": kf.output_ports[0],
    }

    dt = 0.01
    Tsolve = 1.0
    nseg = ceil(Tsolve / dt)
    options = SimulatorOptions(
        max_major_steps=10 * nseg,
        max_major_step_length=dt,
    )

    sol = collimator.simulate(
        diagram,
        context,
        (0.0, Tsolve),
        options=options,
        recorded_signals=recorded_signals,
    )

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5))
        ax1.plot(
            sol.time,
            sol.outputs["theta_measured"],
            label=r"$\theta_\mathrm{meas}$",
            alpha=0.5,
        )
        ax1.plot(sol.time, sol.outputs["x_true"], label=r"$\theta_\mathrm{true}$")
        ax1.plot(sol.time, sol.outputs["x_hat"][:, 0], label=r"$\theta_\mathrm{kf}$")
        ax2.plot(sol.time, sol.outputs["x_hat"][:, 1], label=r"$\omega_\mathrm{kf}$")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_kalman_filter("zoh", discrete_time_plant=False, plot=True)
    # test_infinite_horizon_kalman_filter("zoh", discrete_time_plant=False, plot=True)
    # test_continuous_time_infinite_horizon_kalman_filter(plot=True)
    # test_extended_kalman_filter_discretization_methods("euler", plot=True)
    # test_unscented_kalman_filter_discretization_methods("euler", plot=True)
