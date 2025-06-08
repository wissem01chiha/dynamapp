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
import collimator
import jaxlib
from collimator.library import (
    Sine,
    Constant,
    Comparator,
    StateMachine,
    Integrator,
    Clock,
)
from collimator.library.state_machine import (
    StateMachineData,
    StateMachineState,
    StateMachineTransition,
)
from collimator.framework.error import StaticError, BlockInitializationError
import matplotlib.pyplot as plt

from collimator.simulation import SimulatorOptions

# from collimator import logging


# logging.set_log_handlers(to_file="test.log")

pytestmark = pytest.mark.minimal


def _build_sm_all_ops():
    # 3 states: off=0, a2=2, a3=3
    # this state machine does not do anything, its
    # only purpose is to ensure wildcat does fail due to
    # any of the guard/action strings
    off_t0 = StateMachineTransition(guard="in_0==0.5 and in_1", dst=2)

    a2_t0 = StateMachineTransition(
        guard="in_0 <= 10 or in_1",
        actions=["out_0 = in_0 > 7; out_1 = in_0 == 9; out_2=in_2[1]"],
        dst=3,
    )
    a2_t1 = StateMachineTransition(guard="not in_1", dst=0)

    a3_t0 = StateMachineTransition(guard="in_0>=0.5 and not in_1", dst=0)
    off = StateMachineState(name="off", transitions=[off_t0])
    a2 = StateMachineState(name="a2", transitions=[a2_t0, a2_t1])
    a3 = StateMachineState(name="a3", transitions=[a3_t0])

    sm_data = StateMachineData(
        states={0: off, 2: a2, 3: a3},
        intial_state=0,
        inital_actions=["out_0=0", "out_1=0.", "out_2=0."],
    )

    inputs = ["in_0", "in_1", "in_2"]
    outputs = ["out_0", "out_1", "out_2"]

    return sm_data, inputs, outputs


def test_state_machine_all_ops():
    builder = collimator.DiagramBuilder()

    dt = 0.1
    sw = builder.add(Sine(name="sw"))
    z = builder.add(Constant(value=0.0, name="z"))
    cmp = builder.add(Comparator(name="cmp", operator=">"))
    arr = builder.add(Constant(value=np.array([1.0, 2.0, 3.0]), name="arr"))
    sm_data, sm_inputs, sm_outputs = _build_sm_all_ops()
    sm = builder.add(
        StateMachine(
            name="sm",
            sm_data=sm_data,
            dt=dt,
            inputs=sm_inputs,
            outputs=sm_outputs,
            time_mode="discrete",
        )
    )
    builder.connect(sw.output_ports[0], sm.input_ports[0])
    builder.connect(sw.output_ports[0], cmp.input_ports[0])
    builder.connect(z.output_ports[0], cmp.input_ports[1])
    builder.connect(cmp.output_ports[0], sm.input_ports[1])
    builder.connect(arr.output_ports[0], sm.input_ports[2])

    diagram = builder.build()
    context = diagram.create_context()
    collimator.simulate(diagram, context, (0.0, 0.2))


def test_state_machine_unguarded_exit(show_plot=False):
    # 3 states: off=0, on=1, wait=2
    # off guarded to on, on not guarded wait, wait not guarded back to off
    off_t0 = StateMachineTransition(guard="in_0>0.5", actions=["out_0=1"], dst=1)
    on_t0 = StateMachineTransition(actions=["out_0=2"], dst=2)
    wait_t0 = StateMachineTransition(guard="True", actions=["out_0=0"], dst=0)

    off = StateMachineState(name="off", transitions=[off_t0])
    on = StateMachineState(name="on", transitions=[on_t0])
    wait = StateMachineState(name="wait", transitions=[wait_t0])

    sm_data = StateMachineData(
        states={0: off, 1: on, 2: wait}, intial_state=0, inital_actions=["out_0=0"]
    )

    sm_inputs = ["in_0"]
    sm_outputs = ["out_0"]

    builder = collimator.DiagramBuilder()

    dt = 0.1
    sw = builder.add(Sine(name="sw"))
    sm = builder.add(
        StateMachine(
            name="sm", sm_data=sm_data, dt=dt, inputs=sm_inputs, outputs=sm_outputs
        )
    )
    builder.connect(sw.output_ports[0], sm.input_ports[0])

    recorded_signals = {"sw": sw.output_ports[0], "sm": sm.output_ports[0]}
    diagram = builder.build()
    context = diagram.create_context()
    res = collimator.simulate(
        diagram, context, (0.0, 1.0), recorded_signals=recorded_signals
    )

    time = np.array(res.time)
    sw = np.array(res.outputs["sw"])
    sm = np.array(res.outputs["sm"])

    sm_sol = np.zeros_like(time)
    for idx, t in enumerate(time):
        sw_ = sw[idx]
        if idx > 0:
            sm_sol[idx] = sm_sol[idx - 1]
        rem = np.remainder(t, dt)
        if rem < 1e-3 or np.abs(rem - dt) < 1e-3:
            if sw_ > 0.5 and sm_sol[idx - 1] == 0:
                sm_sol[idx] = 1
            elif sm_sol[idx - 1] == 1:
                sm_sol[idx] = 2
            elif sm_sol[idx - 1] == 2:
                sm_sol[idx] = 0

    print(f"time=\n{time}")
    print(f"sw=\n{sw}")
    print(f"sm=\n{sm}")
    print(f"sm_sol=\n{sm_sol}")

    if show_plot:
        fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))
        ax1.plot(time, sm_sol, label="sm_sol", marker="x")
        ax1.plot(time, sm, label="sm", marker="o")
        ax1.plot(time, sw, label="sw", marker="_")
        ax1.grid(True)
        ax1.legend()
        ax2.plot(time, sm - sm_sol, label="err", marker="o")
        ax2.grid(True)
        ax2.legend()
        plt.show()

    # see wc-128. there are duplciate time sample at 0.6sec. this causes smach and sm_sol mismatch at t=0.6
    # so we cannot try to compare signals until after that.
    idx_compare = 0  # np.argmin(np.abs(time - 0.6)) + 10

    print(f"idx_compare={idx_compare}")
    print(f"time at idx_compare={time[idx_compare]}")

    assert np.allclose(sm_sol[idx_compare:], sm[idx_compare:])


def test_state_machine_entry_point_action(show_plot=False):
    # states: off=0
    # ep[out=99.] -> off[do nothing]

    off = StateMachineState(name="off")

    sm_data = StateMachineData(
        states={0: off}, intial_state=0, inital_actions=["out_0=99."]
    )

    sm_outputs = ["out_0"]

    builder = collimator.DiagramBuilder()
    sm = builder.add(
        StateMachine(name="sm", dt=0.1, sm_data=sm_data, outputs=sm_outputs)
    )

    recorded_signals = {"sm": sm.output_ports[0]}
    diagram = builder.build()
    context = diagram.create_context()
    res = collimator.simulate(
        diagram, context, (0.0, 1.0), recorded_signals=recorded_signals
    )

    time = np.array(res.time)
    sm = np.array(res.outputs["sm"])

    sm_sol = np.ones_like(time) * 99.0

    print(f"time=\n{time}")
    print(f"sm=\n{sm}")
    print(f"sm_sol=\n{sm_sol}")

    if show_plot:
        fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
        ax1.plot(time, sm_sol, label="sm_sol", marker="x")
        ax1.plot(time, sm, label="sm", marker="o")
        ax1.grid(True)
        ax1.legend()
        plt.show()

    assert np.allclose(sm_sol, sm)


def test_state_machine_too_many_unguarded_exit():
    # states: off=0, on=1
    # off has 2 unguarded exits
    off_t0 = StateMachineTransition(actions=["out_0=1"], dst=1)
    off_t1 = StateMachineTransition(actions=["out_0=2"], dst=1)

    off = StateMachineState(name="off", transitions=[off_t0, off_t1])
    on = StateMachineState(name="on")

    sm_data = StateMachineData(states={0: off, 1: on}, intial_state=0)

    sm_inputs = []
    sm_outputs = ["out_0"]

    builder = collimator.DiagramBuilder()

    dt = 0.1

    with pytest.raises(StaticError) as e:
        builder.add(
            StateMachine(
                name="sm", sm_data=sm_data, dt=dt, inputs=sm_inputs, outputs=sm_outputs
            )
        )
    # Success! The test failed as expected.
    assert (
        "StateMachine sm state[off,0] has more than one unguarded exit transition."
        in str(e)
    )


def test_state_machine_no_states():
    sm_data = StateMachineData(states={}, intial_state=0)

    sm_inputs = []
    sm_outputs = ["out_0"]

    builder = collimator.DiagramBuilder()

    dt = 0.1

    with pytest.raises(StaticError) as e:
        builder.add(
            StateMachine(
                name="sm", sm_data=sm_data, dt=dt, inputs=sm_inputs, outputs=sm_outputs
            )
        )
    # Success! The test failed as expected.
    assert "StateMachine sm must have at least one state." in str(e)


def test_state_machine_invalid_code():
    invalid0 = "__import__('os').system('rm -rf /')"
    invalid1 = "__import__('subprocess').check_output(['ls', '/etc'])"

    off_t0 = StateMachineTransition(guard="in_0>0.5", actions=[invalid0], dst=1)
    on_t0 = StateMachineTransition(guard=invalid1, actions=["out_0=0"], dst=0)

    off = StateMachineState(name="off", transitions=[off_t0])
    on = StateMachineState(name="on", transitions=[on_t0])

    sm_data = StateMachineData(states={0: off, 1: on}, intial_state=0)

    sm_inputs = []
    sm_outputs = ["out_0"]

    builder = collimator.DiagramBuilder()

    dt = 0.1

    with pytest.raises(StaticError) as e:
        builder.add(
            StateMachine(
                name="sm", sm_data=sm_data, dt=dt, inputs=sm_inputs, outputs=sm_outputs
            )
        )
    # Success! The test failed as expected.
    print(e)
    assert "exit transitions have the following invalid entries:" in str(e.value)


def test_state_machine_continuous(show_plot=False):
    # states: off=0, on=1
    # sine -> state_machine -> integrator
    # integrator input will be [0.0, 1.0, 0.0 ... ] as state machine changes state
    off_t0 = StateMachineTransition(guard="in_0>0.5", actions=["out_0=1."], dst=1)
    on_t0 = StateMachineTransition(guard="in_0<-0.5", actions=["out_0=0."], dst=0)

    off = StateMachineState(name="off", transitions=[off_t0])
    on = StateMachineState(name="on", transitions=[on_t0])

    sm_data = StateMachineData(
        states={0: off, 1: on}, intial_state=0, inital_actions=["out_0=0."]
    )

    sm_inputs = ["in_0"]
    sm_outputs = ["out_0"]

    builder = collimator.DiagramBuilder()
    sw = builder.add(Sine(name="sw"))
    sm = builder.add(
        StateMachine(
            name="sm",
            sm_data=sm_data,
            inputs=sm_inputs,
            outputs=sm_outputs,
        )
    )
    int_sm = builder.add(Integrator(name="int_sm", initial_state=0.0))
    int_sw = builder.add(Integrator(name="int_sw", initial_state=0.0))

    builder.connect(sw.output_ports[0], sm.input_ports[0])
    builder.connect(sm.output_ports[0], int_sm.input_ports[0])
    builder.connect(sw.output_ports[0], int_sw.input_ports[0])

    recorded_signals = {
        "sw": sw.output_ports[0],
        "sm": sm.output_ports[0],
        "sm_md": sm.output_ports[1],
        "int_sm": int_sm.output_ports[0],
    }
    diagram = builder.build()
    context = diagram.create_context()
    options = SimulatorOptions(max_major_step_length=0.1)
    res = collimator.simulate(
        diagram,
        context,
        (0.0, 6.0),
        recorded_signals=recorded_signals,
        options=options,
    )

    time = np.array(res.time)
    sw = np.array(res.outputs["sw"])
    sm = np.array(res.outputs["sm"])
    sm_md = np.array(res.outputs["sm_md"])
    int_sm = np.array(res.outputs["int_sm"])

    # this pass condition is kind hacky. but it's necessary because of
    # the event occurred results sample recorded just before applying reset maps.
    sm_sol = np.zeros_like(time)
    dly_up = 0
    dly_down = 0
    for idx, val in enumerate(sw):
        if val >= 0.5:
            dly_down = 0
            if dly_up < 1:
                dly_up += 1
                sm_sol[idx] = sm_sol[idx - 1]
            else:
                sm_sol[idx] = 1.0
        elif val <= -0.5:
            dly_up = 0
            if dly_down < 1:
                dly_down += 1
                sm_sol[idx] = sm_sol[idx - 1]
            else:
                sm_sol[idx] = 0.0
        elif idx > 0:
            sm_sol[idx] = sm_sol[idx - 1]

    print(f"time=\n{time}")
    print(f"sw=\n{sw}")
    print(f"sm=\n{sm}")
    print(f"sm_sol=\n{sm_sol}")

    if show_plot:
        fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
        ax1.plot(time, sw, label="sw", marker="_")
        ax1.plot(time, sm, label="sm", marker="o")
        ax1.plot(time, sm_md, label="sm_md", marker="o")
        ax1.plot(time, int_sm, label="int_sm", marker="x")
        ax1.plot(time, sm_sol, label="sm_sol")
        ax1.grid(True)
        ax1.legend()
        plt.show()

    assert np.allclose(sm, sm_sol)


def test_state_machine_output_mismatch_dtype():
    off_t0 = StateMachineTransition(guard="in_0>1.0", actions=["out_0=in_0"], dst=1)
    off = StateMachineState(name="off", transitions=[off_t0])
    on_t0 = StateMachineTransition(guard="in_0>1.5", actions=["out_0=0"], dst=0)
    on = StateMachineState(name="on", transitions=[on_t0])

    sm_data = StateMachineData(
        states={0: off, 1: on}, intial_state=0, inital_actions=["out_0=0"]
    )
    sm_inputs = ["in_0", "in_1"]
    sm_outputs = ["out_0"]

    builder = collimator.DiagramBuilder()
    clk = builder.add(Clock(name="clk"))
    k = builder.add(Constant(name="k", value=np.array([1.0, 2.0])))
    sm = builder.add(
        StateMachine(
            name="sm",
            dt=0.1,
            sm_data=sm_data,
            inputs=sm_inputs,
            outputs=sm_outputs,
            time_mode="discrete",
        )
    )
    builder.connect(clk.output_ports[0], sm.input_ports[0])
    builder.connect(k.output_ports[0], sm.input_ports[1])

    recorded_signals = {"sm": sm.output_ports[0]}
    diagram = builder.build()
    context = diagram.create_context()

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError) as e:
        collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
    # Success! The test failed as expected.
    print(e)
    # it's some big jax.xla error. no point in validating it.


def test_state_machine_uninit_output():
    off_t0 = StateMachineTransition(guard="in_0>1.0", actions=["out_0=in_0"], dst=0)
    off = StateMachineState(name="off", transitions=[off_t0])

    sm_data = StateMachineData(states={0: off}, intial_state=0, inital_actions=[])
    sm_outputs = ["out_0"]

    builder = collimator.DiagramBuilder()

    with pytest.raises(BlockInitializationError) as e:
        builder.add(
            StateMachine(name="sm", dt=0.1, sm_data=sm_data, outputs=sm_outputs)
        )
    # Success! The test failed as expected.
    print(e)
    assert (
        "StateMachine sm does not initialize the following output values in the entry point actions:"
        in str(e)
    )
    assert "out_0" in str(e)


def test_state_machine_guarded_counter(show_plot=False):
    # intentionally set such that t0 is higher priority initially.
    # the priority should be updated at block creation.
    counting_t1 = StateMachineTransition(guard="tmr<15", actions=["tmr=20"], dst=1)
    counting_t0 = StateMachineTransition(guard="True", actions=["tmr=tmr-1"], dst=0)

    counting = StateMachineState(
        name="counting", transitions=[counting_t0, counting_t1]
    )
    done = StateMachineState(name="done")

    sm_data = StateMachineData(
        states={0: counting, 1: done}, intial_state=0, inital_actions=["tmr=20"]
    )
    sm_outputs = ["tmr"]

    builder = collimator.DiagramBuilder()

    dt = 0.1
    sm = builder.add(
        StateMachine(
            name="sm",
            sm_data=sm_data,
            outputs=sm_outputs,
            dt=dt,
            time_mode="discrete",
        )
    )

    recorded_signals = {"sm": sm.output_ports[0]}
    diagram = builder.build()
    context = diagram.create_context()
    res = collimator.simulate(
        diagram, context, (0.0, 0.8), recorded_signals=recorded_signals
    )

    time = np.array(res.time)
    sm = np.array(res.outputs["sm"])

    sm_sol = np.ones_like(time) * 20
    st = 0
    for idx, t in enumerate(time):
        if idx > 0:
            if (sm_sol[idx - 1] < 15 and st == 0) or st == 1:
                sm_sol[idx] = 20
                st = 1
            else:
                sm_sol[idx] = sm_sol[idx - 1] - 1.0

    print(f"time=\n{time}")
    print(f"sm=\n{sm}")
    print(f"sm_sol=\n{sm_sol}")

    if show_plot:
        fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
        ax1.plot(time, sm_sol, label="sm_sol", marker="x")
        ax1.plot(time, sm, label="sm", marker="o")
        ax1.grid(True)
        ax1.legend()
        plt.show()

    assert np.allclose(sm, sm_sol)


if __name__ == "__main__":
    # test_state_machine_invalid_code()
    test_state_machine_continuous(show_plot=True)
    # test_state_machine_entry_point_action(show_plot=False)
    # test_state_machine_no_states()
    # test_state_machine_all_ops()
    # test_state_machine_unguarded_exit(show_plot=False)
    # test_state_machine_too_many_unguarded_exit()
    # test_state_machine_output_mismatch_dtype()
    # test_state_machine_uninit_output()
    # test_state_machine_guarded_counter(show_plot=True)
