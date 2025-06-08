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

import pytest

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import collimator
from collimator.library import Integrator, Demultiplexer
from collimator.framework.event import IntegerTime
from collimator.simulation.types import ContinuousIntervalData

from collimator.logging import logger

from collimator.testing import fd_grad
from collimator.models import BouncingBall as BouncingBall1d

from collimator import logging


logging.set_file_handler("test.log")

pytestmark = pytest.mark.slow


class ConstantIntegrator(collimator.LeafSystem):
    def __init__(
        self,
        a=1.0,
        time_is_terminal=False,
        state_is_terminal=False,
        name=None,
    ):
        super().__init__(name=name)

        self.declare_dynamic_parameter("a", a)
        self.declare_dynamic_parameter("trigger_time", 0.5)
        self.declare_dynamic_parameter("trigger_state", 1.5)
        self.declare_continuous_state(shape=(), ode=self._ode)

        self.declare_zero_crossing(
            guard=self._time_guard,
            reset_map=self._reset if not time_is_terminal else None,
            terminal=time_is_terminal,
            name="time_reset",
        )

        self.declare_zero_crossing(
            guard=self._state_guard,
            reset_map=self._reset,
            terminal=state_is_terminal,
            name="state_reset",
        )

    def _ode(self, time, state, **params):
        return params["a"]

    def _time_guard(self, time, state, **params):
        return time - params["trigger_time"]

    def _state_guard(self, time, state, **params):
        xc = state.continuous_state
        return xc - params["trigger_state"]

    def _reset(self, time, state, **params):
        return state.with_continuous_state(0.0)


class ScalarLinear(collimator.LeafSystem):
    def __init__(self, *args, a=1.0, xr=1.0, xt=1.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_dynamic_parameter("a", a)
        self.declare_dynamic_parameter("reset_state", xr)  # State reset value
        self.declare_dynamic_parameter("trigger_state", xt)  # Trigger state
        self.declare_continuous_state(shape=(), ode=self._ode)

        self.declare_zero_crossing(
            guard=self._state_guard, reset_map=self._reset, name="state_reset"
        )

    def _ode(self, time, state, **params):
        x = state.continuous_state
        return params["a"] * x

    def _state_guard(self, time, state, **params):
        xc = state.continuous_state
        xt = params["trigger_state"]
        return xc - xt

    def _reset(self, time, state, **params):
        xr = params["reset_state"]
        return state.with_continuous_state(xr)


class BouncingBall2d(collimator.LeafSystem):
    def __init__(self, *args, g=9.81, e=1.0, **kwargs):
        super().__init__(*args, name="ball", **kwargs)

        self.declare_continuous_state(4, ode=self.ode)  # [x, y, vx, vy]
        self.declare_continuous_state_output(name="ball:state")
        self.declare_dynamic_parameter("g", g)
        self.declare_dynamic_parameter(
            "e", e
        )  # Resitiution coefficent (0.0 <= e <= 1.0)

        self.declare_zero_crossing(
            guard=self._x_signed_distance,
            reset_map=self._x_reset,
            name="x_bounce",
            direction="positive_then_non_positive",
        )

        self.declare_zero_crossing(
            guard=self._y_signed_distance,
            reset_map=self._y_reset,
            name="y_bounce",
            direction="positive_then_non_positive",
        )

    def ode(self, time, state, **parameters):
        g = parameters["g"]
        x, y, vx, vy = state.continuous_state
        return jnp.array([vx, vy, 0.0, -g])

    def _x_signed_distance(self, time, state, **parameters):
        x, y, vx, vy = state.continuous_state
        return x

    def _y_signed_distance(self, time, state, **parameters):
        x, y, vx, vy = state.continuous_state
        return y

    def _x_reset(self, time, state, **parameters):
        # Update velocity using Newtonian restitution model.
        x, y, vx, vy = state.continuous_state
        e = parameters["e"]

        return state.with_continuous_state(jnp.array([jnp.abs(x), y, -e * vx, vy]))

    def _y_reset(self, time, state, **parameters):
        # Update velocity using Newtonian restitution model.
        x, y, vx, vy = state.continuous_state
        e = parameters["e"]
        return state.with_continuous_state(jnp.array([x, jnp.abs(y), vx, -e * vy]))


class TestGuardLocalization:
    def test_constant_dynamics(self):
        # this test no longer tests _localize_zero_crossings() directly because
        # calls to sim.solve no longer exist inside that function.
        # calls to sim.solve have been moved to guarded_integrate, so the
        # previous way of testing would no longer work.
        collimator.set_backend("jax")

        model = ConstantIntegrator()
        context = model.create_context()

        trigger_time = context.parameters["trigger_time"]

        simulator_options = collimator.SimulatorOptions(rtol=1e-3)
        sim = collimator.Simulator(model, options=simulator_options)
        solver_state = sim.ode_solver.initialize(context)

        guarded_integrate = sim._advance_continuous_time
        if sim.enable_tracing:
            guarded_integrate = jax.jit(guarded_integrate)

        tf = 5.0
        int_tf = IntegerTime.from_decimal(tf)

        cdata = ContinuousIntervalData(
            context=context,
            triggered=False,
            terminate_early=False,
            t0=0,
            tf=int_tf,
            results_data=None,
            ode_solver_state=solver_state,
        )

        cdata = guarded_integrate(cdata)

        # Unpack results from the integration interval
        triggered = cdata.triggered
        context = cdata.context

        tc = context.time

        assert jnp.allclose(tc, trigger_time)
        assert triggered, "Should have triggered a guard"

    def test_constant_dynamics_adjoint(self):
        collimator.set_backend("jax")
        # Integrate to the final time and compare the autodiff results to
        # the analytic solution.  For trigger time tt, if tf > tt, then the
        # final state will be:
        #   xf = a * (tf - tt)
        #   -->  dxf/dx0 = 0.0
        #   -->  dxf/dtf = a
        #   -->  dxf/dtt = -a
        #   -->  dxf/da = (tf - tt)

        model = ConstantIntegrator(a=0.0)
        context = model.create_context()

        rtol = 1e-3

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        simulator_options = collimator.SimulatorOptions(
            rtol=rtol,
            enable_autodiff=True,
            max_major_steps=100,
        )
        sim = collimator.Simulator(model, options=simulator_options)

        x0 = context.continuous_state

        def forward(context, x0, tf, tt, a):
            context = context.with_continuous_state(x0)
            context = context.with_parameters(
                {"a": a, "trigger_time": tt, "trigger_state": jnp.inf}
            )
            sim_state = sim.advance_to(tf, context)
            return sim_state.context.continuous_state

        fwd_grad = jax.value_and_grad(forward, argnums=(1, 2, 3, 4))
        if sim.enable_tracing:
            fwd_grad = jax.jit(fwd_grad)

        x0 = 0.5
        tt = 0.5  # Trigger time
        tf = 0.8  # Interval end time
        a = 1.5
        xf, grads = fwd_grad(context, x0, tf, tt, a)
        dx0, dtf, dtt, da = grads

        # Check forward solution against analytic result
        assert jnp.allclose(xf, a * (tf - tt), rtol=rtol)

        # Check that dxf/dx0 = 0.0
        assert jnp.allclose(dx0, 0.0, rtol=rtol)

        # Check that dxf/dtf = a
        assert jnp.allclose(dtf, a, rtol=rtol)

        # Check that dxf/dtt = -a
        assert jnp.allclose(dtt, -a, rtol=rtol)

        # Check that dxf/da = (tf - tt)
        assert jnp.allclose(da, tf - tt, rtol=rtol)

    def test_constant_dynamics_terminal(self):
        collimator.set_backend("jax")
        model = ConstantIntegrator(time_is_terminal=True)
        context = model.create_context()

        trigger_time = context.parameters["trigger_time"]

        rtol = 1e-3
        options = collimator.SimulatorOptions(rtol=rtol, max_major_steps=100)
        sim = collimator.Simulator(model, options=options)
        advance_to = jax.jit(sim.advance_to)

        # Check that the simulation stops short at the triggered time
        tf = 2 * trigger_time
        results = advance_to(tf, context)
        assert jnp.allclose(results.context.time, trigger_time, rtol=rtol)

    def test_constant_dynamics_terminal_adjoint(self):
        collimator.set_backend("jax")

        # This tests parametric dependency in the guard function for a
        # terminal event.
        #
        # Analytic solution
        #   xf = x0 + a * tt,  where tt is the triggered time
        #   -->  dxf/dx0 = 1.0
        #   -->  dxf/dtf = 0.0
        #   -->  dxf/dtt = a
        #   -->  dxf/da = tt

        model = ConstantIntegrator(time_is_terminal=True)
        context = model.create_context()

        rtol = 1e-3

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        options = collimator.SimulatorOptions(
            enable_autodiff=True,
            math_backend="jax",
            max_major_steps=100,
            rtol=rtol,
        )
        sim = collimator.Simulator(model, options=options)

        def forward(context, x0, tf, tt, a):
            context = context.with_continuous_state(x0)
            context = context.with_parameters({"a": a, "trigger_time": tt})
            sim_state = sim.advance_to(tf, context)
            final_context = sim_state.context
            return final_context.time, final_context.continuous_state

        # Extract the state as the only output
        def forward_state_only(*args):
            _, xf = forward(*args)
            return xf

        fwd_grad = jax.grad(forward_state_only, argnums=(1, 2, 3, 4))
        if sim.enable_tracing:
            forward = jax.jit(forward)
            fwd_grad = jax.jit(fwd_grad)

        # Check that the simulation stops short at the triggered time
        x0 = 0.5
        tt = 0.5  # Trigger time
        tf0 = 0.8  # Requested interval end time
        a = 1.5

        # Check the forward solution is correct
        tf, xf = forward(context, x0, tf0, tt, a)
        assert jnp.allclose(tf, tt, rtol=rtol)
        assert jnp.allclose(xf, x0 + a * tt, rtol=rtol)

        # Check the gradients
        (dx0, dtf0, dtt, da) = fwd_grad(context, x0, tf0, tt, a)

        # Check that dxf/dx0 = 1.0
        assert jnp.allclose(dx0, 1.0, rtol=rtol)

        # Check that dxf/dtf = 0.0 (since the simulation stops short)
        assert jnp.allclose(dtf0, 0.0, rtol=rtol)

        # Check that dxf/dtt = a
        assert jnp.allclose(dtt, a, rtol=rtol)

        # Check that dxf/da = tt
        assert jnp.allclose(da, tt, rtol=rtol)

    def test_linear_dynamics(self):
        collimator.set_backend("jax")

        # this test no longer tests _localize_zero_crossings() directly because
        # calls to sim.solve no longer exist inside that function.
        # calls to sim.solve have been moved to guarded_integrate, so the
        # previous way of testing would no longer work.
        xr = 1.2  # State reset value
        xt = 1.5  # Trigger state
        x0 = 1.0  # Initial state
        a = 1.0  # Linear coefficient
        model = ScalarLinear(xr=xr, xt=xt)
        context = model.create_context()
        context = context.with_continuous_state(x0)

        simulator_options = collimator.SimulatorOptions(
            rtol=1e-6,
            max_major_steps=100,
        )
        sim = collimator.Simulator(model, options=simulator_options)

        advance_to = sim.advance_to
        if sim.enable_tracing:
            advance_to = jax.jit(advance_to)

        # When does x cross xt?
        tt = jnp.log(xt / x0) / a
        tf = tt + 1e-12
        sim_state = advance_to(tf, context)

        # Unpack results from the integration interval
        context = sim_state.context
        xc = context.continuous_state

        print(f"xc={xc} xt={xt} xr={xr}")
        print(f"tf={context.time}")

        # Check that the final state is approximately the reset value
        assert jnp.allclose(xc, xr)

    # @pytest.mark.xfail(reason="cant get the vjp_fun to work")
    def test_linear_dynamics_adjoint(self):
        collimator.set_backend("jax")

        # At first the solution will be:
        #   xf = x0 * exp(a * tf)
        # until the state reaches the trigger state xt=1.5, at which point
        # the state will reset to xr.  This will happen at time tt=ln(xt/x0)/a.
        # Thereafter the solution will be:
        #   xf = xr * exp(a * (tf - tt))
        # Analytic derivatives (after reset):
        #   -->  dxf/dx0 = (xr / xt) * exp(a * tf)
        #   -->  dxf/dtf = a * (xr * x0 / xt) * exp(a * tf)
        #   -->  dxf/dxt = -(xr * x0 / xt^2) * exp(a * tf)
        #   -->  dxf/dxr = exp(a * (tf - tt))
        #   -->  dxf/da = (x0 * xr / xt) * tf * exp(a * tf)

        model = ScalarLinear()
        context = model.create_context()
        context = context.with_continuous_state(1.0)

        rtol = 1e-8
        atol = 1e-10

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        simulator_options = collimator.SimulatorOptions(
            rtol=rtol,
            atol=atol,
            enable_autodiff=True,
            max_major_steps=100,
        )
        sim = collimator.Simulator(model, options=simulator_options)

        x0 = context.continuous_state

        def forward(context, x0, tf, xt, xr, a):
            context = context.with_continuous_state(x0)
            context = context.with_parameters(
                {"a": a, "trigger_state": xt, "reset_state": xr}
            )
            sim_state = sim.advance_to(tf, context)
            return sim_state.context.continuous_state

        fwd_grad = jax.value_and_grad(forward, argnums=(1, 2, 3, 4, 5))

        if sim.enable_tracing:
            fwd_grad = jax.jit(fwd_grad)

        tf = 0.5  # Requested end time (after reset)
        a = 1.5
        x0 = 1.0
        xr = 1.0
        xt = 1.5  # Trigger state

        # Expected trigger time (about t=0.27)
        tt = jnp.log(xt / x0) / a

        # Check results of forward sim
        xf, grads = fwd_grad(context, x0, tf, xt, xr, a)

        print(f"{xf=}, {tt=}")

        dx0, dtf, dxt, dxr, da = grads
        print("Autodiff results:")
        print(f"{dx0=}, {dtf=}, {dxt=}, {dxr=}, {da=}")

        print("Expected:")
        print(f"xf={xr * jnp.exp(a * (tf - tt))}")
        print(f"dx0={(xr / xt) * jnp.exp(a * tf)}")
        print(f"dtf={a * (xr * x0 / xt) * jnp.exp(a * tf)}")
        print(f"dxt={-(xr * x0 / xt**2) * jnp.exp(a * tf)}")
        print(f"dxr={jnp.exp(a * (tf - tt))}")
        print(f"da={(x0 * xr / xt) * tf * jnp.exp(a * tf)}")

        # Check the final state after the reset
        assert jnp.allclose(xf, xr * jnp.exp(a * (tf - tt)))

        # Check that dxf/dx0 = (xr / xt) * exp(a * tf)
        assert jnp.allclose(dx0, (xr / xt) * jnp.exp(a * tf))

        # Check that dxf/dtf = a * (xr * x0 / xt) * exp(a * tf)
        assert jnp.allclose(dtf, a * (xr * x0 / xt) * jnp.exp(a * tf))

        # Check that dxf/dxt = -(xr * x0 / xt^2) * exp(a * tf)
        assert jnp.allclose(dxt, -(xr * x0 / xt**2) * jnp.exp(a * tf))

        # Check that dxf/dxr = exp(a * (tf - tt))
        assert jnp.allclose(dxr, jnp.exp(a * (tf - tt)))

        # Check that dxf/da = (x0 * xr / xt) * tf * exp(a * tf)
        assert jnp.allclose(da, (x0 * xr / xt) * tf * jnp.exp(a * tf))


class TestResetAdjointLeafSystem:
    def test_linear_single_reset(self):
        collimator.set_backend("jax")

        g0 = 10.0
        model = BouncingBall1d(g=g0)
        context = model.create_context()

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        options = collimator.SimulatorOptions(
            enable_autodiff=True,
            max_major_steps=100,
            atol=1e-10,
            rtol=1e-8,
        )
        sim = collimator.Simulator(model, options=options)

        def forward(context, y0, e, h, tf):
            context = context.with_continuous_state(jnp.array([y0, 0.0]))
            context = context.with_parameter("e", e)
            context = context.with_parameter("h0", h)

            sim_state = sim.advance_to(tf, context)
            final_context = sim_state.context
            return final_context.continuous_state[0]

        if sim.enable_tracing:
            forward = jax.jit(forward)

        # Nominal values
        e = 0.7
        y0 = 5.0  # Initial height -> t1=1.0 when y0=5.0
        h = 0.1  # Floor height
        t1 = np.sqrt(2 * (y0 - h) / g0)

        # Check value before transition
        tf = t1 - 0.1
        yf = forward(context, y0, e, h, tf)
        y_exact = y0 - 0.5 * g0 * tf**2
        logger.debug(f"{yf=}, {y_exact=}")
        assert jnp.allclose(yf, y_exact)

        # Check value after transition
        tf = t1 + 0.1
        yf = forward(context, y0, e, h, tf)
        v_plus = e * g0 * t1
        y_exact = h + v_plus * (tf - t1) - 0.5 * g0 * (tf - t1) ** 2
        logger.debug(f"{yf=}, {y_exact=}")
        assert jnp.allclose(yf, y_exact)

        # Check gradients before transition
        tf = t1 - 0.1
        dy0, de, dh, dtf = jax.grad(forward, argnums=(1, 2, 3, 4))(
            context, y0, e, h, tf
        )
        logger.debug(f"{dy0=}, {de=}, {dh=}, {dtf=}")
        assert jnp.allclose(dy0, 1.0)
        assert jnp.allclose(de, 0.0)
        assert jnp.allclose(dh, 0.0)
        assert jnp.allclose(dtf, -g0 * tf)

        # Check gradients after transition
        tf = t1 + 0.1
        dy0, de, dh, dtf = jax.grad(forward, argnums=(1, 2, 3, 4))(
            context, y0, e, h, tf
        )
        logger.debug(f"{dy0=}, {de=}, {dh=}, {dtf=}")
        dy0_exact = (1 + e) * (tf / t1) - (1 + 2 * e)
        dtf_exact = g0 * ((1 + e) * t1 - tf)
        de_exact = g0 * t1 * (tf - t1)
        dh_exact = (1 + e) * (2 - tf / t1)
        logger.debug(f"{dy0_exact=}, {de_exact=}, {dh_exact=}, {dtf_exact=}")
        assert jnp.allclose(dy0, dy0_exact)
        assert jnp.allclose(de, de_exact)
        assert jnp.allclose(dh, dh_exact)
        assert jnp.allclose(dtf, dtf_exact)

    def test_linear_multiple_reset(self):
        collimator.set_backend("jax")

        model = BouncingBall1d()
        context = model.create_context()

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        simulator_options = collimator.SimulatorOptions(
            enable_autodiff=True,
            max_major_steps=100,
            atol=1e-12,
            rtol=1e-10,
            max_major_step_length=0.1,  # required to localize events with sufficient precision
        )
        sim = collimator.Simulator(model, options=simulator_options)

        def forward(context, y0, g, e, tf):
            context = context.with_continuous_state(jnp.array([y0, 0.0]))
            context = context.with_parameters({"g": g, "e": e})

            sim_state = sim.advance_to(tf, context)
            final_context = sim_state.context
            return final_context.continuous_state[0]

        grad_fwd = jax.grad(forward, argnums=(1, 2, 3, 4))
        if sim.enable_tracing:
            forward = jax.jit(forward)
            grad_fwd = jax.jit(grad_fwd)

        # Nominal values
        e = 0.7
        y0 = 5.0  # Initial height -> t1=1.0 when y0=5.0
        g0 = 10.0
        t1 = np.sqrt(2 * y0 / g0)

        # Check gradients after transition
        tf = 1.5 * t1  # Multiple resets

        yf = forward(context, y0, g0, e, tf)
        print(f"{yf=}")

        dy0, dg, de, dtf = grad_fwd(context, y0, g0, e, tf)
        print(f"{dy0=}, {dtf=}, {dg=}, {de=}")

        eps_fd = 1e-6
        dy0_fd, dg_fd, de_fd, dtf_fd = fd_grad(
            partial(forward, context), y0, g0, e, tf, eps=eps_fd
        )
        print(f"{dy0_fd=}, {dtf_fd=}, {dg_fd=}, {de_fd=}")

        test_atol = 1e-4
        assert jnp.allclose(dy0, dy0_fd, atol=test_atol)
        assert jnp.allclose(dtf, dtf_fd, atol=test_atol)
        assert jnp.allclose(dg, dg_fd, atol=test_atol)
        assert jnp.allclose(de, de_fd, atol=test_atol)

    def test_nonlinear_multiple_reset(self):
        collimator.set_backend("jax")

        model = BouncingBall1d()
        context = model.create_context()

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        simulator_options = collimator.SimulatorOptions(
            enable_autodiff=True,
            max_major_steps=100,
            atol=1e-12,
            rtol=1e-10,
        )
        sim = collimator.Simulator(model, options=simulator_options)

        def forward(context, y0, g, b, e, tf):
            context = context.with_continuous_state(jnp.array([y0, 0.0]))
            context = context.with_parameters({"g": g, "e": e, "b": b})

            sim_state = sim.advance_to(tf, context)
            final_context = sim_state.context
            return final_context.continuous_state[0]

        grad_forward = jax.grad(forward, argnums=(1, 2, 3, 4, 5))
        if sim.enable_tracing:
            forward = jax.jit(forward)
            grad_forward = jax.jit(grad_forward)

        # Nominal values
        e = 0.7
        b = 0.1
        y0 = 5.0
        g0 = 10.0
        t1 = np.sqrt(2 * y0 / g0)

        # Check gradients after transition
        tf = 3.0 * t1  # Multiple resets

        # yf = forward(context, y0, g0, b, e, tf)
        # print(f"{yf=}")

        dy0, dg, db, de, dtf = grad_forward(context, y0, g0, b, e, tf)
        print(f"{dy0=}, {dtf=}, {dg=}, {db=}, {de=}")

        eps_fd = 1e-6
        dy0_fd, dg_fd, db_fd, de_fd, dtf_fd = fd_grad(
            partial(forward, context), y0, g0, b, e, tf, eps=eps_fd
        )
        print(f"{dy0_fd=}, {dtf_fd=}, {dg_fd=}, {db_fd=}, {de_fd=}")

        test_atol = 1e-4
        assert jnp.allclose(dy0, dy0_fd, atol=test_atol)
        assert jnp.allclose(dtf, dtf_fd, atol=test_atol)
        assert jnp.allclose(dg, dg_fd, atol=test_atol)
        assert jnp.allclose(db, db_fd, atol=test_atol)
        assert jnp.allclose(de, de_fd, atol=test_atol)

    def test_multiple_guards_multiple_resets(self):
        collimator.set_backend("jax")

        model = BouncingBall2d()
        context = model.create_context()

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        simulator_options = collimator.SimulatorOptions(
            enable_autodiff=True,
            max_major_steps=400,
            atol=1e-12,
            rtol=1e-10,
            max_major_step_length=0.1,
        )
        sim = collimator.Simulator(model, options=simulator_options)

        def forward(context, x0, g, e, tf, out_index=0):
            context = context.with_continuous_state(x0)
            context = context.with_parameters({"g": g, "e": e})

            sim_state = sim.advance_to(tf, context)
            final_context = sim_state.context
            return final_context.continuous_state[out_index]

        if sim.enable_tracing:
            forward = jax.jit(forward)

        # Nominal values
        e = 0.7
        x0 = jnp.array([1.0, 1.0, -1.0, 0.0])
        g0 = 10.0
        tf = 1.2

        # Loop over (x, y, vx, vy) and compare gradients to finite differencing
        for out_index in range(4):
            func = partial(forward, context, out_index=out_index)
            grad_func = jax.grad(func, argnums=(0, 1, 2, 3))

            if sim.enable_tracing:
                func = jax.jit(func)
                grad_func = jax.jit(grad_func)

            dx0, dg, de, dtf = grad_func(x0, g0, e, tf)
            print(f"{dx0=}, {dtf=}, {dg=}, {de=}")

            eps_fd = 1e-6
            dx0_fd, dg_fd, de_fd, dtf_fd = fd_grad(func, x0, g0, e, tf, eps=eps_fd)
            print(f"{dx0_fd=}, {dtf_fd=}, {dg_fd=}, {de_fd=}")

            test_atol = 1e-4
            assert jnp.allclose(dx0, dx0_fd, atol=test_atol)
            assert jnp.allclose(dtf, dtf_fd, atol=test_atol)
            assert jnp.allclose(dg, dg_fd, atol=test_atol)
            assert jnp.allclose(de, de_fd, atol=test_atol)

    def test_time_varying_guard_reset(self):
        collimator.set_backend("jax")

        model = BouncingBall1d()
        context = model.create_context()

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        simulator_options = collimator.SimulatorOptions(
            enable_autodiff=True,
            max_major_steps=100,
            atol=1e-12,
            rtol=1e-10,
        )
        sim = collimator.Simulator(model, options=simulator_options)

        def forward(context, x0, g, hdot, e, tf, out_index=0):
            context = context.with_continuous_state(x0)
            context = context.with_parameters({"g": g, "e": e, "hdot": hdot})

            sim_state = sim.advance_to(tf, context)
            final_context = sim_state.context
            return final_context.continuous_state[out_index]

        fwd_y = partial(forward, context, out_index=0)
        fwd_v = partial(forward, context, out_index=1)

        grad_y = jax.grad(fwd_y, argnums=(0, 1, 2, 3, 4))
        grad_v = jax.grad(fwd_v, argnums=(0, 1, 2, 3, 4))

        if sim.enable_tracing:
            fwd_y = jax.jit(fwd_y)
            fwd_v = jax.jit(fwd_v)
            grad_y = jax.jit(grad_y)
            grad_v = jax.jit(grad_v)

        # Nominal values
        e = 0.7
        hdot = 0.2
        x0 = jnp.array([5.0, 0.0])
        g0 = 10.0

        # Check gradients after transition
        tf = 4.0  # Multiple resets

        for fwd, grad_fwd in zip((fwd_y, fwd_v), (grad_y, grad_v)):
            dx0, dg, dhdot, de, dtf = grad_fwd(x0, g0, hdot, e, tf)
            print(f"{dx0=}, {dtf=}, {dg=}, {dhdot=}, {de=}")

            eps_fd = 1e-6
            dx0_fd, dg_fd, dhdot_fd, de_fd, dtf_fd = fd_grad(
                fwd, x0, g0, hdot, e, tf, eps=eps_fd
            )
            print(f"{dx0_fd=}, {dtf_fd=}, {dg_fd=}, {dhdot_fd=}, {de_fd=}")

            test_atol = 1e-3
            assert jnp.allclose(dx0, dx0_fd, atol=test_atol)
            assert jnp.allclose(dtf, dtf_fd, atol=test_atol)
            assert jnp.allclose(dg, dg_fd, atol=test_atol)
            assert jnp.allclose(dhdot, dhdot_fd, atol=test_atol)
            assert jnp.allclose(de, de_fd, atol=test_atol)


class TestResetAdjointDiagram:
    def test_nonlinear_multiple_reset(self):
        collimator.set_backend("jax")

        builder = collimator.DiagramBuilder()

        ball = builder.add(BouncingBall1d(name="ball"))
        demux = builder.add(Demultiplexer(2, name="demux"))
        integrator = builder.add(Integrator(0.0, name="integrator"))

        builder.connect(ball.output_ports[0], demux.input_ports[0])
        builder.connect(demux.output_ports[0], integrator.input_ports[0])

        system = builder.build()

        # Have to speciy max_major_steps here because we're calling `advance_to`
        # directly rather than using the `simulate` interface, which would estimate
        # it automatically. Also we're differentiating with respect to the end
        # time, so the automatic estimation of max_major_steps won't work because
        # the final time is a JAX tracer and not a concrete float value.
        simulator_options = collimator.SimulatorOptions(
            enable_autodiff=True,
            max_major_steps=100,
            atol=1e-12,
            rtol=1e-10,
        )
        sim = collimator.Simulator(system, options=simulator_options)

        def forward(context, x0, g, b, e, tf):
            # Update the leaf contexts
            int_context = context[integrator.system_id].with_continuous_state(
                jnp.array(0.0)
            )
            ball_context = context[ball.system_id].with_continuous_state(x0)
            ball_context = ball_context.with_parameters({"g": g, "e": e, "b": b})

            # Store the updated leaf contexts in the root context
            context = context.with_subcontext(integrator.system_id, int_context)
            context = context.with_subcontext(ball.system_id, ball_context)

            sim_state = sim.advance_to(tf, context)
            final_context = sim_state.context
            # return final_context["ball"].continuous_state[0]
            return final_context[integrator.system_id].continuous_state

        context = system.create_context()
        forward = partial(forward, context)
        grad_fwd = jax.grad(forward, argnums=(0, 1, 2, 3, 4))

        if sim.enable_tracing:
            forward = jax.jit(forward)
            grad_fwd = jax.jit(grad_fwd)

        # Nominal values
        e = 0.7
        b = 0.01
        x0 = jnp.array([5.0, 0.0])
        g0 = 10.0
        tf = 4.0  # Multiple resets

        # yf = forward(x0, g0, b, e, tf)
        # logger.debug(f"{yf=}")

        # Check gradients after transitions
        dx0, dg, db, de, dtf = grad_fwd(x0, g0, b, e, tf)

        eps_fd = 1e-6
        dx0_fd, dg_fd, db_fd, de_fd, dtf_fd = fd_grad(
            forward, x0, g0, b, e, tf, eps=eps_fd
        )
        logger.debug(f"{dx0=}, {dtf=}, {dg=}, {db=}, {de=}")
        logger.debug(f"{dx0_fd=}, {dtf_fd=}, {dg_fd=}, {db_fd=}, {de_fd=}")

        test_atol = 1e-2
        assert jnp.allclose(dx0, dx0_fd, atol=test_atol)
        assert jnp.allclose(dtf, dtf_fd, atol=test_atol)
        assert jnp.allclose(dg, dg_fd, atol=test_atol)
        assert jnp.allclose(db, db_fd, atol=test_atol)
        assert jnp.allclose(de, de_fd, atol=test_atol)


if __name__ == "__main__":
    # TestGuardLocalization().test_constant_dynamics()
    # TestGuardLocalization().test_constant_dynamics_adjoint()
    # TestGuardLocalization().test_constant_dynamics_terminal()
    # TestGuardLocalization().test_constant_dynamics_terminal_adjoint()
    # TestGuardLocalization().test_linear_dynamics()
    # TestGuardLocalization().test_linear_dynamics_adjoint()

    # TestResetAdjointLeafSystem().test_linear_single_reset()
    # TestResetAdjointLeafSystem().test_linear_multiple_reset()
    # TestResetAdjointLeafSystem().test_nonlinear_multiple_reset()
    TestResetAdjointLeafSystem().test_multiple_guards_multiple_resets()
    # TestResetAdjointLeafSystem().test_time_varying_guard_reset()

    # TestResetAdjointDiagram().test_nonlinear_multiple_reset()
