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
import numpy as np
import matplotlib.pyplot as plt
import collimator
from collimator.framework import LeafSystem, IntegerTime
from collimator.backend import jit
from collimator.library import (
    Adder,
    Gain,
    Clock,
    DiscreteClock,
    Integrator,
    Sine,
    Constant,
    Comparator,
    Ramp,
    Abs,
    DeadZone,
    Saturate,
    IfThenElse,
    LogicalOperator,
    MinMax,
)

from collimator.simulation.simulator import (
    ContinuousIntervalData,
    StepEndReason,
    _determine_step_end_reason,
)

pytestmark = pytest.mark.minimal


def run_test(
    model,
    evt_data,
    boundary_time=1.0,
):
    triggered_times, triggered_names = evt_data
    context = model.create_context()

    options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
    sim = collimator.Simulator(model, options=options)

    # JIT-compile the advance function for speed (if using JAX backend)
    guarded_integrate = jit(sim._advance_continuous_time)
    solver_state = sim.ode_solver.initialize(context)

    int_boundary_time = IntegerTime.from_decimal(boundary_time)
    int_update_time = IntegerTime.from_decimal(np.inf)

    triggered_idx = 0
    step_idx = 0
    while context.time < boundary_time:
        print(f"\nstep_idx={step_idx} start time={context.time}")
        step_idx = step_idx + 1

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
            triggered, terminate_early, int_boundary_time, int_update_time
        )

        if end_reason == StepEndReason.GuardTriggered:
            event_time = triggered_times[triggered_idx]
            event_name = triggered_names[triggered_idx]
            print(f"event: {event_name}. Tevt={event_time} Tctx={context.time}")
            # see comment in _unguarded_integrate for why we do not assert guards.has_triggered
            assert np.allclose(context.time, event_time)
            triggered_idx = triggered_idx + 1
        else:
            assert not triggered
            assert end_reason == StepEndReason.NothingTriggered

    assert np.allclose(context.time, boundary_time)
    assert triggered_idx == len(triggered_times)


def test_events_abs():
    triggered_times = [
        0.45,  # abs_up
        0.55,  # abs_down
    ]
    triggered_names = [
        "abs_up",
        "abs_down",
    ]
    evt_data = (triggered_times, triggered_names)

    builder = collimator.DiagramBuilder()
    ramp_abs_up = builder.add(
        Ramp(name="ramp_abs_up", start_time=0.0, start_value=-0.45)
    )
    ramp_abs_down = builder.add(
        Ramp(name="ramp_abs_down", start_time=0.0, start_value=0.55, slope=-1.0)
    )
    abs_up = builder.add(Abs(name="abs_up"))
    abs_down = builder.add(Abs(name="abs_down"))
    builder.connect(ramp_abs_up.output_ports[0], abs_up.input_ports[0])
    builder.connect(ramp_abs_down.output_ports[0], abs_down.input_ports[0])

    # Combine the signals and send to Integrator so both have zero-crossing events
    adder = builder.add(Adder(2, name="adder"))
    int_ = builder.add(Integrator(name="int_", initial_state=0.0))
    builder.connect(abs_down.output_ports[0], adder.input_ports[0])
    builder.connect(abs_up.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], int_.input_ports[0])

    model = builder.build()
    run_test(model, evt_data)


def test_events_sat():
    upper_limit = 0.1
    lower_limit = -0.2
    triggered_times = [
        upper_limit,  # sat_upper
        np.abs(lower_limit),  # sat_lower
    ]

    triggered_names = [
        "sat_upper",
        "sat_lower",
    ]
    evt_data = (triggered_times, triggered_names)

    builder = collimator.DiagramBuilder()
    ramp_up = builder.add(Ramp(name="ramp_up", start_time=0.0))
    ramp_down = builder.add(Ramp(name="ramp_down", start_time=0.0, slope=-1.0))
    sat_upr = builder.add(
        Saturate(name="sat_upr", upper_limit=upper_limit, lower_limit=-1e6)
    )
    sat_lwr = builder.add(
        Saturate(name="sat_lwr", upper_limit=1e6, lower_limit=lower_limit)
    )
    builder.connect(ramp_up.output_ports[0], sat_upr.input_ports[0])
    builder.connect(ramp_down.output_ports[0], sat_lwr.input_ports[0])

    # Combine the signals and send to Integrator so both have zero-crossing events
    int_ = builder.add(Integrator(name="int_", initial_state=0.0))
    adder = builder.add(Adder(2, name="adder"))
    builder.connect(sat_lwr.output_ports[0], adder.input_ports[0])
    builder.connect(sat_upr.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], int_.input_ports[0])

    model = builder.build()
    run_test(model, evt_data, boundary_time=0.3)


def test_events_cmp():
    triggered_times = [
        np.arcsin(0.3),  # cmp_gr
        np.arcsin(0.5),  # cmp_greq
        np.arccos(0.5),  # cmp_lseq
        np.arccos(0.3),  # cmp_ls
    ]

    triggered_names = [
        "cmp_gr",
        "cmp_greq",
        "cmp_lseq",
        "cmp_ls",
    ]
    evt_data = (triggered_times, triggered_names)

    builder = collimator.DiagramBuilder()
    sinewave = builder.add(Sine(name="sinwave"))
    coswave = builder.add(Sine(name="coswave", phase=np.pi / 2.0))
    int0 = builder.add(Integrator(name="int0", initial_state=0.0))
    point5 = builder.add(Constant(name="point5", value=0.5))
    point3 = builder.add(Constant(name="point3", value=0.3))
    cmp_greq = builder.add(Comparator(name="cmp_greq", operator=">="))
    cmp_lseq = builder.add(Comparator(name="cmp_lseq", operator="<="))
    cmp_gr = builder.add(Comparator(name="cmp_gr", operator=">"))
    cmp_ls = builder.add(Comparator(name="cmp_ls", operator="<"))

    builder.connect(sinewave.output_ports[0], cmp_greq.input_ports[0])
    builder.connect(point5.output_ports[0], cmp_greq.input_ports[1])
    builder.connect(coswave.output_ports[0], cmp_lseq.input_ports[0])
    builder.connect(point5.output_ports[0], cmp_lseq.input_ports[1])
    builder.connect(sinewave.output_ports[0], cmp_gr.input_ports[0])
    builder.connect(point3.output_ports[0], cmp_gr.input_ports[1])
    builder.connect(coswave.output_ports[0], cmp_ls.input_ports[0])
    builder.connect(point3.output_ports[0], cmp_ls.input_ports[1])

    # Connect the comparators to the integrator so that the
    # blocks have zero-crossing events
    adder = builder.add(Adder(4, name="adder"))
    builder.connect(cmp_greq.output_ports[0], adder.input_ports[0])
    builder.connect(cmp_lseq.output_ports[0], adder.input_ports[1])
    builder.connect(cmp_gr.output_ports[0], adder.input_ports[2])
    builder.connect(cmp_ls.output_ports[0], adder.input_ports[3])

    gain = builder.add(Gain(1.0, name="gain"))  # Convert to float

    builder.connect(adder.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], int0.input_ports[0])

    model = builder.build()
    run_test(model, evt_data, boundary_time=2.0)


def test_events_dz():
    hr_up = 0.2
    hr_down = 0.3
    triggered_times = [
        hr_up,  # dz_up
        hr_down,  # dz_down
    ]

    triggered_names = [
        "dz_up",
        "dz_down",
    ]
    evt_data = (triggered_times, triggered_names)

    builder = collimator.DiagramBuilder()
    ramp_dz_up = builder.add(Ramp(name="ramp_dz_up", start_time=0.0))
    ramp_dz_down = builder.add(Ramp(name="ramp_dz_down", start_time=0.0, slope=-1.0))
    dz_up = builder.add(DeadZone(name="dz_up", half_range=hr_up))
    dz_down = builder.add(DeadZone(name="dz_down", half_range=hr_down))

    builder.connect(ramp_dz_up.output_ports[0], dz_up.input_ports[0])
    builder.connect(ramp_dz_down.output_ports[0], dz_down.input_ports[0])

    # Combine the signals and send to Integrator so both have zero-crossing events
    adder = builder.add(Adder(2, name="adder"))
    int0 = builder.add(Integrator(name="int0", initial_state=0.0))
    builder.connect(dz_down.output_ports[0], adder.input_ports[0])
    builder.connect(dz_up.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], int0.input_ports[0])

    model = builder.build()
    run_test(model, evt_data, boundary_time=0.5)


def test_events_IfThenElse():
    triggered_times = [
        0.3,
    ]

    triggered_names = [
        "if",
    ]
    evt_data = (triggered_times, triggered_names)

    builder = collimator.DiagramBuilder()
    ramp_up = builder.add(Ramp(name="ramp_up", start_time=0.0))
    point3 = builder.add(Constant(name="point3", value=0.3))
    cmp_greq = builder.add(Comparator(name="cmp_greq", operator=">="))
    if_ = builder.add(IfThenElse(name="if_"))
    int0 = builder.add(Integrator(name="int0", initial_state=0.0))

    builder.connect(ramp_up.output_ports[0], cmp_greq.input_ports[0])
    builder.connect(point3.output_ports[0], cmp_greq.input_ports[1])
    builder.connect(cmp_greq.output_ports[0], if_.input_ports[0])
    builder.connect(ramp_up.output_ports[0], if_.input_ports[1])
    builder.connect(point3.output_ports[0], if_.input_ports[2])
    builder.connect(if_.output_ports[0], int0.input_ports[0])

    model = builder.build()
    run_test(model, evt_data, boundary_time=0.5)


def test_events_logicalOp():
    """
    Note: did not test all possible ops, as they should all work the same.
    """
    triggered_times = [
        0.3,  # lo_or
        0.5,  # lo_and
    ]

    triggered_names = ["lo_or", "lo_and"]
    evt_data = (triggered_times, triggered_names)

    builder = collimator.DiagramBuilder()
    ramp_up = builder.add(Ramp(name="ramp_up", start_time=0.0))
    point3 = builder.add(Constant(name="point3", value=0.3))
    point5 = builder.add(Constant(name="point5", value=0.5))
    cmp_greq_p3 = builder.add(Comparator(name="cmp_greq_p3", operator=">="))
    cmp_greq_p5 = builder.add(Comparator(name="cmp_greq_p5", operator=">="))
    lo_or = builder.add(LogicalOperator(name="lo_or", function="or"))
    lo_and = builder.add(LogicalOperator(name="lo_and", function="and"))
    int0 = builder.add(Integrator(name="int0", initial_state=0.0))

    builder.connect(ramp_up.output_ports[0], cmp_greq_p3.input_ports[0])
    builder.connect(point3.output_ports[0], cmp_greq_p3.input_ports[1])

    builder.connect(ramp_up.output_ports[0], cmp_greq_p5.input_ports[0])
    builder.connect(point5.output_ports[0], cmp_greq_p5.input_ports[1])

    builder.connect(cmp_greq_p3.output_ports[0], lo_or.input_ports[0])
    builder.connect(cmp_greq_p5.output_ports[0], lo_or.input_ports[1])

    builder.connect(cmp_greq_p3.output_ports[0], lo_and.input_ports[0])
    builder.connect(cmp_greq_p5.output_ports[0], lo_and.input_ports[1])

    # Connect the logical operators to the Integrator so that the
    # blocks have zero-crossing events
    adder = builder.add(Adder(2, name="adder"))
    builder.connect(lo_or.output_ports[0], adder.input_ports[0])
    builder.connect(lo_and.output_ports[0], adder.input_ports[1])
    gain = builder.add(Gain(1.0, name="gain"))  # Convert to float
    builder.connect(adder.output_ports[0], gain.input_ports[0])

    builder.connect(gain.output_ports[0], int0.input_ports[0])

    model = builder.build()
    run_test(model, evt_data, boundary_time=1.0)


def test_events_minmax():
    triggered_times = [
        0.3,  # min
        0.5,  # max
    ]

    triggered_names = ["min", "max"]
    evt_data = (triggered_times, triggered_names)

    builder = collimator.DiagramBuilder()
    ramp_up = builder.add(Ramp(name="ramp_up", start_time=0.0))
    point3 = builder.add(Constant(name="point3", value=0.3))
    point5 = builder.add(Constant(name="point5", value=0.5))
    min_ = builder.add(MinMax(name="min_", n_in=2, operator="min"))
    max_ = builder.add(MinMax(name="max_", n_in=2, operator="max"))
    int0 = builder.add(Integrator(name="int0", initial_state=0.0))

    builder.connect(ramp_up.output_ports[0], min_.input_ports[0])
    builder.connect(point3.output_ports[0], min_.input_ports[1])

    builder.connect(ramp_up.output_ports[0], max_.input_ports[0])
    builder.connect(point5.output_ports[0], max_.input_ports[1])

    builder.connect(ramp_up.output_ports[0], int0.input_ports[0])

    model = builder.build()
    run_test(model, evt_data, boundary_time=1.0)


# Block that just ends the simulation early
class Terminator(LeafSystem):
    def __init__(self, terminal_value, name: str = None):
        super().__init__(name=name)

        self.declare_input_port()

        def _guard(time, state, *inputs, **params):
            (u,) = inputs
            return terminal_value - u

        self.declare_zero_crossing(
            _guard,
            direction="positive_then_non_positive",
            terminal=True,
        )


class TestTerminalEvents:
    @pytest.mark.minimal
    def test_terminal_event(self):
        # Check that a system with no continuous states and a terminal event
        # terminates at the right time.

        dt = 0.1  # Effective discrete step length

        # Trigger the event between the first and second major steps.
        # Since the system doesn't have continuous events, the zero-crossing
        # is only localized to within one major step.
        terminal_time = 1.5 * dt
        t_final = 2 * dt  # Expected actual end time

        builder = collimator.DiagramBuilder()
        terminator = builder.add(Terminator(terminal_value=terminal_time))
        clock = builder.add(Clock())
        builder.connect(clock.output_ports[0], terminator.input_ports[0])
        system = builder.build()

        context = system.create_context()

        options = collimator.simulation.SimulatorOptions(max_major_step_length=dt)
        results = collimator.simulate(system, context, (0.0, 5 * dt), options=options)

        assert np.allclose(results.context.time, t_final)

    def test_terminate_on_discrete(self):
        # Check that a termination event triggered by a discrete update
        # will stop before advancing continuous time.
        # The system is a discrete clock connected to a terminator block, along
        # with a constant input and an integrator.  The integrator state should
        # be the same as the simulation time.
        # The output of the discrete block will change from 0.1 to 0.2 during
        # the discrete update at t=0.2, triggering an event monitoring for
        # value 0.15 on the output.  The event should then halt simulation
        # before the rest of the major step, which would otherwise advance
        # simulation to t=0.3.

        dt = 0.1  # Effective discrete step length
        terminal_value = 0.15

        builder = collimator.DiagramBuilder()
        terminator = builder.add(Terminator(terminal_value=terminal_value))
        clock = builder.add(DiscreteClock(dt))
        const = builder.add(Constant(1.0))
        integrator = builder.add(Integrator(0.0))
        builder.connect(clock.output_ports[0], terminator.input_ports[0])
        builder.connect(const.output_ports[0], integrator.input_ports[0])
        system = builder.build()

        context = system.create_context()
        results = collimator.simulate(system, context, (0.0, 5 * dt))

        # Expected results
        t_final = 0.2
        xf = 0.2  # Final integrator state

        assert np.allclose(results.context.time, t_final)
        assert np.allclose(results.context.continuous_state[0], xf)

    def test_terminate_on_continuous(self):
        # Check that a termination event triggered by a continuous update
        # will properly localize the event and then terminate.
        # The system is a discrete clock, along with a constant input and an
        # integrator.  The integrator state should be the same as the simulation
        # time.  The integrator state is connected to the terminator block,
        # so that the event will trigger when the integrator state reaches
        # the terminal value.

        dt = 0.1  # Effective discrete step length
        terminal_value = 0.15

        builder = collimator.DiagramBuilder()
        terminator = builder.add(Terminator(terminal_value=terminal_value))
        builder.add(DiscreteClock(dt))
        const = builder.add(Constant(1.0))
        integrator = builder.add(Integrator(0.0))
        builder.connect(const.output_ports[0], integrator.input_ports[0])
        builder.connect(integrator.output_ports[0], terminator.input_ports[0])
        system = builder.build()

        context = system.create_context()

        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        results = collimator.simulate(system, context, (0.0, 5 * dt), options=options)

        # Expected results
        t_final = 0.15
        xf = 0.15  # Final integrator state

        assert np.allclose(results.context.time, t_final)
        assert np.allclose(results.context.continuous_state[0], xf)


def test_result_event_detection(show_plot=False):
    # a bouncing ball model.
    # this test ensures that the comparator block output has some 'True' samples.

    builder = collimator.DiagramBuilder()
    accel = builder.add(Constant(-9.81, name="accel"))
    floor = builder.add(Constant(0.0, name="floor"))
    vel = builder.add(
        Integrator(
            initial_state=0.0,
            enable_reset=True,
            enable_external_reset=True,
            name="vel",
        )
    )
    pos = builder.add(
        Integrator(
            initial_state=1.0,
            enable_reset=True,
            enable_external_reset=True,
            name="pos",
        )
    )
    impact = builder.add(Comparator(name="impact", operator="<"))
    restitution = builder.add(Gain(-0.6, name="restitution"))

    builder.connect(accel.output_ports[0], vel.input_ports[0])
    builder.connect(vel.output_ports[0], pos.input_ports[0])
    builder.connect(pos.output_ports[0], impact.input_ports[0])
    builder.connect(floor.output_ports[0], impact.input_ports[1])
    builder.connect(impact.output_ports[0], vel.input_ports[1])
    builder.connect(impact.output_ports[0], pos.input_ports[1])
    builder.connect(vel.output_ports[0], restitution.input_ports[0])
    builder.connect(restitution.output_ports[0], vel.input_ports[2])
    builder.connect(floor.output_ports[0], pos.input_ports[2])

    diagram = builder.build()
    context = diagram.create_context()

    options = collimator.SimulatorOptions(rtol=1e-10, atol=1e-8)
    recorded_signals = {
        "pos": pos.output_ports[0],
        "impact": impact.output_ports[0],
    }
    r = collimator.simulate(
        diagram,
        context,
        (0.0, 1.0),
        options=options,
        recorded_signals=recorded_signals,
    )

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(r.time, r.outputs["pos"], label="pos")
        ax.plot(r.time, r.outputs["impact"], label="impact")
        ax.legend()
        plt.show()

    assert np.any(r.outputs["impact"])


if __name__ == "__main__":
    # test_events_abs()
    # test_events_cmp()
    # test_events_minmax()
    # test_events_logicalOp()
    test_result_event_detection(show_plot=True)
