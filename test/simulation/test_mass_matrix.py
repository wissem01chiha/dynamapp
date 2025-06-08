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

"""Tests for semi-explicit index-1 DAEs specified as mass matrix ODEs."""

import pytest

import numpy as np
from scipy import linalg

import jax.numpy as jnp

import collimator
from collimator import library

A0 = np.array([[-1.0, 0.0], [0.0, -1.0]])
B0 = np.array([[0.0], [1.0]])


class VectorLinear(collimator.LeafSystem):
    #
    # M @ ẋ = A @ x + B @ u
    #

    def __init__(self, mass_matrix=None, A=None, B=None, x0=None, name=None):
        super().__init__(name=name)
        if A is None:
            A = A0
        if B is None:
            B = B0
        self.declare_dynamic_parameter("A", A)
        self.declare_dynamic_parameter("B", B)

        if x0 is None:
            x0 = np.zeros(A.shape[0])

        self.declare_continuous_state(
            default_value=x0,
            ode=self.ode,
            mass_matrix=mass_matrix,
        )
        self.declare_input_port(name="u_in")
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, u, **parameters):
        A, B = parameters["A"], parameters["B"]
        x = state.continuous_state
        return A @ x + B @ u


class TestVectorLinearLeaf:
    def _test_sim(self, tf, mass_matrix=None, u0=None, x0=None, solver="auto"):
        if u0 is None:
            u0 = np.array([0.0])
        if x0 is None:
            x0 = np.array([0.8, 0.9])

        model = VectorLinear(mass_matrix=mass_matrix)
        model.input_ports[0].fix_value(u0)

        ctx = model.create_context()
        ctx = ctx.with_continuous_state(x0)
        options = collimator.SimulatorOptions(ode_solver_method=solver)
        results = collimator.simulate(
            model,
            ctx,
            (0.0, tf),
            options=options,
        )
        A = ctx.parameters["A"]
        M = model.mass_matrix
        Minv_A = linalg.lstsq(M, A)[0]
        x_ex = linalg.expm(Minv_A * tf) @ x0
        xf = results.context.continuous_state
        assert np.allclose(xf, x_ex)

        return model

    def test_sim(self):
        # No mass matrix
        tf = 2.0
        self._test_sim(tf)

    def test_sim_identity_mass(self):
        # Identity mass matrix
        tf = 2.0
        mass_matrix = np.eye(2)
        model = self._test_sim(tf, mass_matrix)

        # Since the mass matrix is identity, the configuration
        # should recognize this as trivial and not store it
        assert not model.has_mass_matrix

    def test_sim_diag_mass(self):
        # Non-trivial mass matrix specified by diagonals
        tf = 2.0
        mass_matrix = np.diag([1.0, 2.0])

        with pytest.raises(ValueError):
            # Must use BDF solver here
            self._test_sim(tf, mass_matrix, solver="rk45")

        model = self._test_sim(tf, mass_matrix, solver="bdf")

        # The mass matrix should be stored
        assert model.has_mass_matrix

    def test_sim_square_mass(self):
        # Non-trivial mass matrix specified as square array
        tf = 2.0
        mass_matrix = np.array([[1.0, 0.5], [0.2, 2.0]])

        with pytest.raises(ValueError):
            # Must use BDF solver here
            self._test_sim(tf, mass_matrix, solver="rk45")

        model = self._test_sim(tf, mass_matrix, solver="bdf")

        # The mass matrix should be stored
        assert model.has_mass_matrix

    def test_sim_singular_mass(self):
        # Singular mass matrix.  In this case the second row
        # amounts to a constraint that the second state is equal
        # to the value of `u0`.  It's important that the initial
        # condition be consistent in this case, meaning that x0[1] = u0
        mass_matrix = np.array([1.0, 0.0])
        tf = 2.0
        u0 = np.array([2.0])
        x0 = np.array([1.0, 2.0])

        model = self._test_sim(tf, mass_matrix, u0=u0, x0=x0, solver="bdf")

        # The mass matrix should be stored
        assert model.has_mass_matrix

    def test_sim_invalid_mass(self):
        # Try several forms of invalid mass matrices
        tf = 2.0

        # Non-square mass matrix
        mass_matrix = np.array([[1.0, 0.5, 0.3], [0.2, 2.0, 0.1]])
        with pytest.raises(AssertionError):
            self._test_sim(tf, mass_matrix)

        # Wrong number of elements on diagonal
        mass_matrix = np.diag([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            self._test_sim(tf, mass_matrix)


class TestVectorLinearDiagram:
    def _make_diagram(self, blk1, blk2, source):
        # Create a simple diagram with two blocks and one common source.
        # For testing purposes, the two blocks should define equivalent systems
        builder = collimator.DiagramBuilder()
        builder.add(blk1, blk2, source)
        builder.connect(source.output_ports[0], blk1.input_ports[0])
        builder.connect(source.output_ports[0], blk2.input_ports[0])
        return builder.build()

    def _test_sim(self, tf, mass_matrix=None, u0=None, x0=None):
        # Create two VectorLinear blocks, one of which uses a mass matrix and
        # one of which does not, but both are constructed to be equivalent.
        if u0 is None:
            u0 = np.array([1.0])
        if x0 is None:
            x0 = np.array([0.8, 0.9])

        blk1 = VectorLinear(A=A0, B=B0, mass_matrix=mass_matrix, x0=x0)
        Minv = linalg.pinv(blk1.mass_matrix)
        blk2 = VectorLinear(A=Minv @ A0, B=Minv @ B0, mass_matrix=None, x0=x0)
        assert not blk2.has_mass_matrix
        source = library.Constant(u0)

        system = self._make_diagram(blk1, blk2, source)

        context = system.create_context()
        options = collimator.SimulatorOptions(ode_solver_method="bdf")
        results = collimator.simulate(
            system,
            context,
            (0.0, tf),
            options=options,
        )
        xf1 = results.context[blk1.system_id].continuous_state
        xf2 = results.context[blk2.system_id].continuous_state

        assert np.allclose(xf1, xf2)
        return system

    def test_sim_identity_mass(self):
        tf = 2.0
        M = np.eye(2)
        system = self._test_sim(tf, mass_matrix=M)
        assert not system.has_mass_matrix

    def test_sim_diag_mass(self):
        # Non-trivial mass matrix specified by diagonals
        tf = 2.0
        M = np.diag([1.0, 2.0])
        system = self._test_sim(tf, mass_matrix=M)
        assert system.has_mass_matrix

    def test_sim_square_mass(self):
        # Non-trivial mass matrix specified by diagonals
        tf = 2.0
        M = np.array([[1.0, 0.5], [0.2, 2.0]])
        system = self._test_sim(tf, mass_matrix=M)
        assert system.has_mass_matrix

    def test_sim_singular_mass(self):
        # Singular mass matrix. See TestVectorLinearLeaf for interpretation
        tf = 2.0
        M = np.array([1.0, 0.0])
        u0 = np.array([2.0])
        x0 = np.array([1.0, 2.0])
        system = self._test_sim(tf, mass_matrix=M, u0=u0, x0=x0)
        assert system.has_mass_matrix


class Robertson(collimator.LeafSystem):
    #
    # dx/dt = -0.04 * x + 1e4 * y * z
    # dy/dt = 0.04 * x - 1e4 * y * z - 3e7 * y^2
    # dz/dt = 3e7 * y^2
    #
    # There is also a conservation law constraint that x + y + z = 1,
    # so the third equation can be replaced by the constraint residual
    # along with the diagonal mass matrix [1, 1, 0]
    #
    def __init__(self, name=None):
        super().__init__(name=name)
        x0 = np.array([1.0, 0.0, 0.0])
        M = np.diag([1.0, 1.0, 0.0])
        self.declare_continuous_state(
            default_value=x0,
            mass_matrix=M,
            ode=self.ode,
        )
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state):
        x, y, z = state.continuous_state
        return jnp.array(
            [
                -0.04 * x + 1e4 * y * z,
                0.04 * x - 1e4 * y * z - 3e7 * y**2,
                (x + y + z) - 1,
            ]
        )


def test_robertson():
    # Test the Robertson system with a singular mass matrix
    model = Robertson()
    assert model.has_mass_matrix

    ctx = model.create_context()
    tf = 1e7
    recorded_signals = {"x": model.output_ports[0]}
    options = collimator.SimulatorOptions(
        ode_solver_method="bdf",
        int_time_scale=1e-10,
    )
    results = collimator.simulate(
        model,
        ctx,
        (0.0, tf),
        options=options,
        recorded_signals=recorded_signals,
    )
    xf = results.context.continuous_state
    assert np.allclose(xf, [0.0, 0.0, 1.0], atol=1e-2)

    # Check that the constraint (x + y + z) = 1 is satisfied
    x = results.outputs["x"]
    assert np.allclose(x.sum(axis=-1), 1.0)


class PlanarPendulum(collimator.LeafSystem):
    """
    ########## Final DAE equations F(x, x_dot, y)=0 ############

    Eq 0   :   -L**2 + x(t)**2 + y(t)**2
    Eq 1   :   2*d_Derivative(y(t), t)*y(t) + 2*x(t)*Derivative(x(t), t)
    Eq 2   :   d_Derivative(y(t), t) - z(t)
    Eq 3   :   -w(t) + Derivative(x(t), t)
    Eq 4   :   2*d_Derivative(x(t), (t, 2))*x(t) + 2*d_Derivative(y(t), (t, 2))*y(t) + 2*d_Derivative(y(t), t)**2 + 2*Derivative(x(t), t)**2
    Eq 5   :   d_Derivative(y(t), (t, 2)) - d_Derivative(z(t), t)
    Eq 6   :   d_Derivative(x(t), (t, 2)) - Derivative(w(t), t)
    Eq 7   :   d_Derivative(z(t), t) + g - T(t)*y(t)
    Eq 8   :   -T(t)*x(t) + Derivative(w(t), t)

    # with, x =

    w(t)                           with ic= 0.0
    x(t)                           with ic= 0.8660254037844386

    # x_dot =

    Derivative(w(t), t)            with ic= -4.243524478543744
    Derivative(x(t), t)            with ic= 0.0

    # and, y =

    d_Derivative(y(t), t)          with ic= 0.0
    T(t)                           with ic= -4.899999999999992
    y(t)                           with ic= -0.49999999999999994
    d_Derivative(x(t), (t, 2))     with ic= -4.243524478543744
    d_Derivative(y(t), (t, 2))     with ic= -7.350000000000004
    d_Derivative(z(t), t)          with ic= -7.350000000000004
    z(t)                           with ic= -9.536405264693243e-32

    ############################################################
    """

    def __init__(self, L=1.0, g0=9.8, name=None):
        super().__init__(name=name)
        x0 = np.array(
            [
                0.0,
                0.8660254037844386,
                0.0,
                -4.9,
                -0.5,
                -4.243524478543744,
                -7.35,
                -7.35,
                0.0,
            ]
        )
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("g0", g0)

        self.nx = 2
        self.nz = 7

        M = np.concatenate([np.ones(self.nx), np.zeros(self.nz)])
        self.declare_continuous_state(
            default_value=x0,
            mass_matrix=M,
            ode=self.ode,
        )
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, **parameters):
        L, g0 = parameters["L"], parameters["g0"]
        x = state.continuous_state[:2]
        z = state.continuous_state[2:]
        f = jnp.array([z[3], x[0]])
        g = jnp.array(
            [
                -(L**2) + x[1] ** 2 + z[2] ** 2,
                2 * z[0] * z[2] + 2 * x[1] * x[0],
                z[0] - z[6],
                2 * z[3] * x[1] + 2 * z[4] * z[2] + 2 * z[0] ** 2 + 2 * x[0] ** 2,
                z[4] - z[5],
                z[5] + g0 - z[1] * z[2],
                -z[1] * x[1] + z[3],
            ]
        )
        return jnp.concatenate([f, g])


def test_planar_pendulum():
    # Test the planar pendulum system with a singular mass matrix
    L = 1.0
    g0 = 9.8
    model = PlanarPendulum(L=L, g0=g0)
    assert model.has_mass_matrix

    ctx = model.create_context()

    # Check that the constraints are satisfied at the initial condition
    rhs0 = model.eval_time_derivatives(ctx)
    assert np.allclose(rhs0[model.nx :], 0.0)

    tf = 2.0
    recorded_signals = {"x": model.output_ports[0]}
    options = collimator.SimulatorOptions(
        ode_solver_method="bdf",
    )
    results = collimator.simulate(
        model,
        ctx,
        (0.0, tf),
        options=options,
        recorded_signals=recorded_signals,
    )

    # Check that the constraint (x**2 + y**2) = L**2 is satisfied
    x = results.outputs["x"][:, : model.nx]
    z = results.outputs["x"][:, model.nx :]
    assert np.allclose((x[:, 1] ** 2 + z[:, 2] ** 2), L**2)


if __name__ == "__main__":
    test_robertson()
    test_planar_pendulum()
