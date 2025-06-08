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
from enum import IntEnum
import dataclasses

import numpy as np
import matplotlib.pyplot as plt
import collimator

# from collimator.common import *
from collimator.library import Sine, Integrator
from collimator.framework.event import (
    IntegerTime,
    ZeroCrossingEvent,
    ZeroCrossingEventData,
)
from collimator.simulation import SimulatorOptions
from collimator.simulation.simulator import (
    ContinuousIntervalData,
    StepEndReason,
    guard_interval_start,
)
from collimator.backend import jit


# from collimator import logging


pytestmark = pytest.mark.minimal


class ConstantIntegrator(collimator.LeafSystem):
    def __init__(self, *args, a=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.trigger_time = 0.5
        self.trigger_state = 1.5
        self.declare_dynamic_parameter("a", a)
        self.declare_continuous_state(shape=(), ode=self._ode)

        self.declare_zero_crossing(
            guard=self._time_guard, reset_map=self._reset, name="time_reset"
        )

        self.declare_zero_crossing(
            guard=self._state_guard, reset_map=self._reset, name="state_reset"
        )

    def _ode(self, time, state, **params):
        return params["a"]

    def _time_guard(self, time, state, **params):
        return time - self.trigger_time

    def _state_guard(self, time, state, **params):
        xc = state.continuous_state
        return xc - self.trigger_state

    def _reset(self, time, state, **params):
        return state.with_continuous_state(0.0)


class ScalarLinear(collimator.LeafSystem):
    def __init__(self, *args, a=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.trigger_state = 1.5

        self.declare_dynamic_parameter("a", a)
        self.declare_continuous_state(shape=(), ode=self._ode)

        self.declare_zero_crossing(
            guard=self._state_guard, reset_map=self._reset, name="state_reset"
        )

    def _ode(self, time, state, **params):
        x = state.continuous_state
        return params["a"] * x

    def _state_guard(self, time, state, **params):
        xc = state.continuous_state
        return xc - self.trigger_state

    def _reset(self, time, state, **params):
        return state.with_continuous_state(0.0)


class SimpleStateMachine(collimator.LeafSystem):
    class Mode(IntEnum):
        A = 1
        B = 2
        C = 3

    def __init__(self, name=None):
        super().__init__(name=name)

        self.declare_input_port()

        self.declare_mode_output()

        # Set up transition map
        self.declare_default_mode(self.Mode.A)
        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._reset,
            name="AB",
            direction="crosses_zero",
            start_mode=self.Mode.A,
            end_mode=self.Mode.B,
        )

        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._reset,
            name="BC",
            direction="crosses_zero",
            start_mode=self.Mode.B,
            end_mode=self.Mode.C,
        )

        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._reset,
            name="CA",
            direction="crosses_zero",
            start_mode=self.Mode.C,
            end_mode=self.Mode.A,
        )

    def _guard(self, time, state, u, **p):
        return u

    def _reset(self, time, state, u, **p):
        return state


def test_eval_guards():
    model = ConstantIntegrator()
    context = model.create_context()

    assert model.zero_crossing_events.num_events == 2

    transitions = model.determine_active_guards(context)
    event1 = transitions.events[0]
    assert event1.name == "time_reset"

    event2 = transitions.events[1]
    assert event2.name == "state_reset"

    context = context.with_time(0.0)
    value = event1.guard(context)
    assert np.allclose(value, -model.trigger_time)

    value = event2.guard(context)
    assert np.allclose(value, -model.trigger_state)

    # Set one to be inactive
    event1.event_data = dataclasses.replace(event1.event_data, active=False)
    value = event1.guard(context)
    assert not event1.should_trigger()

    value = event2.guard(context)
    assert np.allclose(value, -model.trigger_state)


def test_positive_then_non_positive_trigger():
    event = ZeroCrossingEvent(
        system_id=0,
        guard=None,
        reset_map=None,
        direction="positive_then_non_positive",
        event_data=ZeroCrossingEventData(True),
    )
    assert event.event_data.active

    wp = 1.0
    wn = -1.0

    # Check the flag logic
    assert event._should_trigger(wp, wn)
    assert not event._should_trigger(wn, wp)
    assert not event._should_trigger(0.0, wn)
    assert event._should_trigger(wp, 0.0)
    assert not event._should_trigger(wp, wp)
    assert not event._should_trigger(wn, wn)

    # Check that it will always be false if marked inactive
    event.event_data = dataclasses.replace(event.event_data, active=False)
    assert not event._should_trigger(wp, wn)
    assert not event._should_trigger(wn, wp)
    assert not event._should_trigger(0.0, wn)
    assert not event._should_trigger(wp, 0.0)


def test_negative_then_non_negative_trigger():
    event = ZeroCrossingEvent(
        system_id=0,
        guard=None,
        reset_map=None,
        direction="negative_then_non_negative",
        event_data=ZeroCrossingEventData(True),
    )

    assert event.event_data.active

    wp = 1.0
    wn = -1.0

    # Check the flag logic
    assert not event._should_trigger(wp, wn)
    assert event._should_trigger(wn, wp)
    assert not event._should_trigger(0.0, wn)
    assert event._should_trigger(wn, 0.0)
    assert not event._should_trigger(wp, wp)
    assert not event._should_trigger(wn, wn)

    # Check that it will always be false if marked inactive
    event.event_data = dataclasses.replace(event.event_data, active=False)
    assert not event._should_trigger(wp, wn)
    assert not event._should_trigger(wn, wp)
    assert not event._should_trigger(0.0, wn)
    assert not event._should_trigger(wn, 0.0)


def test_crosses_zero_trigger():
    event = ZeroCrossingEvent(
        system_id=0,
        guard=None,
        reset_map=None,
        direction="crosses_zero",
        event_data=ZeroCrossingEventData(True),
    )

    assert event.event_data.active

    wp = 1.0
    wn = -1.0

    # Check the flag logic
    assert event._should_trigger(wp, wn)
    assert event._should_trigger(wn, wp)
    assert not event._should_trigger(wp, wp)
    assert not event._should_trigger(wn, wn)
    assert not event._should_trigger(0.0, wn)
    assert not event._should_trigger(0.0, wp)
    assert event._should_trigger(wn, 0.0)
    assert event._should_trigger(wp, 0.0)

    # Check that it will always be false if marked inactive
    event.event_data = dataclasses.replace(event.event_data, active=False)
    assert not event._should_trigger(wp, wn)
    assert not event._should_trigger(wn, wp)
    assert not event._should_trigger(wp, wp)
    assert not event._should_trigger(wn, wn)
    assert not event._should_trigger(0.0, wn)
    assert not event._should_trigger(0.0, wp)
    assert not event._should_trigger(wn, 0.0)
    assert not event._should_trigger(wp, 0.0)


def test_guard_interval_start():
    model = ConstantIntegrator()
    context = model.create_context()

    transitions = model.determine_active_guards(context)
    transitions = guard_interval_start(transitions, context)

    assert np.allclose(
        transitions.events[0].event_data.w0,
        -model.trigger_time,
    )
    assert np.allclose(
        transitions.events[1].event_data.w0,
        -model.trigger_state,
    )

    # Check eval to NaN on inactive
    event1 = transitions.events[0]
    event1.event_data = dataclasses.replace(event1.event_data, active=False)

    transitions = guard_interval_start(transitions, context)
    assert not transitions.events[0].should_trigger()
    assert np.allclose(
        transitions.events[1].event_data.w0,
        -model.trigger_state,
    )


def test_guarded_integrate():
    from collimator.simulation.simulator import _determine_step_end_reason

    model = ConstantIntegrator()
    context = model.create_context()

    sim = collimator.Simulator(model)

    update_time = np.inf

    # JIT-compile the advance function for speed
    guarded_integrate = jit(sim._advance_continuous_time)
    solver_state = sim.ode_solver.initialize(context)

    # Integrate to a final time which is exactly a multiple of the trigger time
    boundary_time = 2.0
    int_boundary_time = IntegerTime.from_decimal(boundary_time)

    for _ in range(10):
        cdata = ContinuousIntervalData(
            context=context,
            triggered=False,
            terminate_early=False,
            t0=IntegerTime.from_decimal(context.time),
            tf=int_boundary_time,
            results_data=None,
            ode_solver_state=solver_state,
        )
        cdata = guarded_integrate(cdata)
        triggered = cdata.triggered
        context = cdata.context
        solver_state = cdata.ode_solver_state

        terminate_early = False
        end_reason = _determine_step_end_reason(
            triggered, terminate_early, boundary_time, update_time
        )

        if end_reason == StepEndReason.GuardTriggered:
            assert triggered
            assert np.allclose(context.time, model.trigger_time)
            break

        assert not triggered
        assert end_reason == StepEndReason.NothingTriggered
        assert context.time < model.trigger_time

    assert np.allclose(
        context.time, model.trigger_time
    ), "Should have triggered a guard"


def test_advance_to():
    a = 1.0
    model = ConstantIntegrator(a=a)
    context = model.create_context()

    options = collimator.SimulatorOptions(
        max_major_steps=100,
        atol=1e-8,
        rtol=1e-6,
    )
    sim = collimator.Simulator(model, options=options)

    # Test single reset at t=0.5
    tf = 1.0
    xf = (tf - model.trigger_time) * a
    sim_state = sim.advance_to(1.0, context)  # Should trigger a time reset at t=0.5
    context = sim_state.context
    assert np.allclose(context.time, tf)
    assert np.allclose(context.continuous_state, xf)
    assert sim_state.step_end_reason == StepEndReason.NothingTriggered

    # Test multiple resets at t=0.5, t=2.0
    context = model.create_context()
    tf = 2.5
    t1 = model.trigger_time
    t2 = t1 + (model.trigger_state / a)
    xf = (tf - t2) * a

    sim_state = sim.advance_to(tf, context)
    context = sim_state.context
    assert np.allclose(context.time, tf)
    assert np.allclose(context.continuous_state, xf)
    assert sim_state.step_end_reason == StepEndReason.NothingTriggered


def test_simple_state_machine(show_plot=False):
    # The state machine should change modes from A->B, B->C, C->A
    # at every zero crossing.  Given a sine wave input u=sin(pi*t),
    # the mode should change at t=1, t=2, t=3, etc.
    builder = collimator.DiagramBuilder()
    state_machine = builder.add(SimpleStateMachine())
    source = builder.add(Sine(frequency=np.pi))
    int_ = builder.add(Integrator(initial_state=0.0))
    builder.connect(source.output_ports[0], state_machine.input_ports[0])
    builder.connect(source.output_ports[0], int_.input_ports[0])

    system = builder.build()
    context = system.create_context()

    tf = 6.0
    dt = np.pi / 4
    options = SimulatorOptions(
        max_major_step_length=dt,
    )
    recorded_signals = {
        "s": state_machine.output_ports[0],
        "u": source.output_ports[0],
    }
    results = collimator.simulate(
        system,
        context,
        (0.0, tf),
        options=options,
        recorded_signals=recorded_signals,
    )
    # Check that the simulation finished
    assert np.allclose(results.time[-1], tf)

    # Check that the mode changes at t=1, t=2, t=3, etc.
    t = results.time
    s = results.outputs["s"]

    if show_plot:
        fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
        ax1.plot(t, results.outputs["u"], label="sine wave")
        ax1.plot(t, s, label="s")
        ax1.grid(True)
        ax1.legend()
        plt.show()

    # The zero crossing detection is only accurate to within a certain tolerance.  By default,
    # this is about rtol=1e-6.  In other words, the exact transition time is likely to overshoot
    # the analytic value by about 1e-6 with default settings.
    # @am. since simulator.py now requires num_continuous_states>0 in order to handle "transoitions",
    # this test results in slightly different user facing results, and hence the pass conditions
    # change slightly.
    zc_tol = 1e-6
    assert all(
        np.where(
            (t > 0.0 + zc_tol) * (t < 1.0 - zc_tol),
            s == SimpleStateMachine.Mode.A,
            True,
        )
    )
    assert all(
        np.where(
            (t > 1.0 + zc_tol) * (t < 2.0 - zc_tol),
            s == SimpleStateMachine.Mode.B,
            True,
        )
    )
    assert all(
        np.where(
            (t > 2.0 + zc_tol) * (t < 3.0 - zc_tol),
            s == SimpleStateMachine.Mode.C,
            True,
        )
    )
    assert all(
        np.where(
            (t > 3.0 + zc_tol) * (t < 4.0 - zc_tol),
            s == SimpleStateMachine.Mode.A,
            True,
        )
    )
    assert all(
        np.where(
            (t > 4.0 + zc_tol) * (t < 5.0 - zc_tol),
            s == SimpleStateMachine.Mode.B,
            True,
        )
    )
    assert all(
        np.where(
            (t > 5.0 + zc_tol) * (t < 6.0 - zc_tol),
            s == SimpleStateMachine.Mode.C,
            True,
        )
    )


if __name__ == "__main__":
    test_advance_to()
    # test_simple_state_machine(show_plot=True)
