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

import numpy as np
from matplotlib import pyplot as plt
import pytest
import os

# acausal imports
from collimator.experimental import AcausalCompiler, AcausalDiagram, EqnEnv
from collimator.experimental import electrical as elec
from collimator.experimental import rotational as rot
from collimator.experimental import thermal as ht

# collimator imports
import collimator
from collimator import library as lib

import collimator.logging as logging

logging.set_log_level(logging.DEBUG)


def test_basic_RC(show_plot=False):
    # basic test of self contained system, the output is the state of the system.
    # ConstantVoltageSource-Resistor-Capacitor in a loop, connect to Ground.
    # in the simulation, the capacitor charges until its voltage matches that of
    # the voltage source.

    # make acausal diagram
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    c1 = elec.Capacitor(
        ev, name="c1", C=1.0, initial_voltage=0.0, initial_voltage_fixed=True
    )
    ref1 = elec.Ground(ev, name="ref1")
    print(f"{ref1.ports=}")
    print(f"{ref1.syms=}")
    print(f"{ref1.eqs=}")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    lpf = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    lpf = builder.add(lpf)

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    lpf_ctx = context[lpf.system_id]

    # verify acausal diagram params are in the acausal_system context
    params = context[lpf.system_id].parameters
    assert params["v1_v"] == 1.0
    assert params["r1_R"] == 1.0
    assert params["c1_C"] == 1.0

    # run the simulation
    recorded_signals = {
        "x": lpf.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    x = results.outputs["x"]

    def rc_filter(t, v=1, r=1, c=1):
        return v * (1 - np.exp(-t / (r * c)))

    cv_sol = rc_filter(t)

    atol = 0.0002
    rtol = 0

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        for i in range(len(lpf_ctx.state.continuous_state)):
            label = "x" + str(i)
            ax.plot(t, x[:, i], label=label)
        ax.plot(t, cv_sol, label="cv_sol")
        ax.legend()
        plt.show()

    assert np.allclose(cv_sol, x[:, 0], atol=atol, rtol=rtol)


def test_basic_RC_with_outputs(show_plot=False):
    # Basic test of self contained system, the output is from sensors as opposed to the state.
    # ConstantVoltageSource-Resistor-CurrentSensor-Capacitor in a loop, connect to Ground.
    # In the simulation, the capacitor charges until its voltage matches that of
    # the voltage source.
    # The voltage sensor measures the capacitor voltage.
    # The current sensor measures the current in the loop.
    # This test also has parallel resistors, which tests the elimination of mathematically
    # equivalent polynomial expressions. in this case, the parallel resistors results in
    # mathematcialy equivalent flow equations for the nodes on ether side of the resistors.

    # make acausal diagram
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=2.0)
    r2 = elec.Resistor(ev, name="r2", R=2.0)
    c1 = elec.Capacitor(
        ev, name="c1", C=1.0, initial_voltage=0.0, initial_voltage_fixed=True
    )
    ref1 = elec.Ground(ev, name="ref1")
    sensV = elec.VoltageSensor(ev, name="sensV")
    sensI = elec.CurrentSensor(ev, name="sensI")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", sensI, "n")
    ad.connect(sensI, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    ad.connect(c1, "p", sensV, "p")
    ad.connect(c1, "n", sensV, "n")
    ad.connect(r1, "p", r2, "p")
    ad.connect(r1, "n", r2, "n")
    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    lpf = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    lpf = builder.add(lpf)

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    v_idx = lpf.outsym_to_portid[sensV.get_sym_by_port_name("v")]
    i_idx = lpf.outsym_to_portid[sensI.get_sym_by_port_name("i")]
    recorded_signals = {
        "sensV": lpf.output_ports[v_idx],
        "sensI": lpf.output_ports[i_idx],
    }
    options = collimator.SimulatorOptions(ode_solver_method="bdf")
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
        options=options,
    )
    t = results.time
    sensV = results.outputs["sensV"]
    sensI = results.outputs["sensI"]

    def rc_filter(t, v=1, r=1, c=1):
        return v * (1 - np.exp(-t / (r * c)))

    cv_sol = rc_filter(t)

    atol = 0.0002
    rtol = 0

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(t, sensV, label="sensV")
        ax1.plot(t, cv_sol, label="cv_sol")
        ax1.grid()
        ax1.legend()

        ax2.plot(t, sensI, label="sensI")
        ax2.grid()
        ax2.legend()
        plt.show()

    assert np.allclose(cv_sol, sensV, atol=atol, rtol=rtol)


def test_basic_RC_with_voltage_input(show_plot=False):
    # basic test of self contained system, the output is the state of the system.
    # ConstantVoltageSource-Resistor-Capacitor in a loop, connect to Ground.
    # in the simulation, the capacitor charges until its voltage matches that of
    # the voltage source.
    # the only difference between this test and test_basic_RC() is that the
    # voltage source contant is passed in through a acausal_system input, as opposed to
    # set as a acausal_system parameter value.
    # this difference may seem trivial, but
    # that's what this test is intended to show, how trivial difference doesn't
    # produce the same results.

    # make acausal diagram
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", enable_voltage_port=True)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    c1 = elec.Capacitor(
        ev, name="c1", C=1.0, initial_voltage=0.0, initial_voltage_fixed=True
    )
    ref1 = elec.Ground(ev, name="ref1")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    lpf = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    lpf = builder.add(lpf)
    Vconst = builder.add(lib.Constant(value=1.0, name="Vconst"))
    builder.connect(Vconst.output_ports[0], lpf.input_ports[0])

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    lpf_ctx = context[lpf.system_id]

    # Ensure the algebraic equations are satisfied by the initial condition
    rhs = lpf.eval_time_derivatives(context)
    assert np.allclose(rhs[lpf.n_ode :], 0.0)  # noqa

    # run the simulation
    recorded_signals = {
        "x": lpf.output_ports[0],
    }
    options = collimator.SimulatorOptions(ode_solver_method="bdf")
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
        options=options,
    )
    t = results.time
    x = results.outputs["x"]

    def rc_filter(t, v=1, r=1, c=1):
        return v * (1 - np.exp(-t / (r * c)))

    cv_sol = rc_filter(t)

    atol = 0.0002
    rtol = 0

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        for i in range(len(lpf_ctx.state.continuous_state)):
            label = "x" + str(i)
            ax.plot(t, x[:, i], label=label)
        ax.plot(t, cv_sol, label="cv_sol")
        ax.legend()
        plt.show()

    assert np.allclose(cv_sol, x[:, 0], atol=atol, rtol=rtol)


def test_RLC_circuit(show_plot=False):
    # https://www.science.smith.edu/~jcardell/Courses/EGR326/RLC_SSModelIEEE.pdf
    # section 3.3.there has a StateSpace model for RLC circuit.
    # there is actually an error in the A matrix, 1/L should be -1/L.
    R = 1.0
    L = 1.0
    C = 1.0
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", enable_voltage_port=False)
    r1 = elec.Resistor(ev, name="r1", R=R)
    c1 = elec.Capacitor(
        ev, name="c1", C=C, initial_voltage=0.0, initial_voltage_fixed=True
    )
    l1 = elec.Inductor(
        ev, name="l1", L=L, initial_current=0.0, initial_current_fixed=True
    )
    ref1 = elec.Ground(ev, name="ref1")
    sensV = elec.VoltageSensor(ev, name="sensV")
    sensI = elec.CurrentSensor(ev, name="sensI")
    ad.connect(v1, "p", r1, "p")
    ad.connect(r1, "n", l1, "p")
    ad.connect(l1, "n", sensI, "p")
    ad.connect(sensI, "n", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    ad.connect(sensV, "p", c1, "p")
    ad.connect(sensV, "n", c1, "n")
    # compile to acausal system
    ac = AcausalCompiler(ev, ad)
    acausal_system = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    A = np.array([[-R / L, -1 / L], [1 / C, 0]])
    B = np.array([1 / L, 0])
    C_ = np.array([[1, 0], [0, 1]])
    D = np.array([0, 0])
    ss = builder.add(lib.LTISystem(A=A, B=B, C=C_, D=D))
    volt = builder.add(lib.Constant(value=1.0))
    builder.connect(volt.output_ports[0], ss.input_ports[0])

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    v_idx = acausal_system.outsym_to_portid[sensV.get_sym_by_port_name("v")]
    i_idx = acausal_system.outsym_to_portid[sensI.get_sym_by_port_name("i")]
    recorded_signals = {
        "sensV": acausal_system.output_ports[v_idx],
        "sensI": acausal_system.output_ports[i_idx],
        "ss": ss.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    sensV = results.outputs["sensV"]
    sensI = results.outputs["sensI"]
    ssV = results.outputs["ss"][:, 1]
    ssI = results.outputs["ss"][:, 0]

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(t, sensV, label="sensV")
        ax.plot(t, sensI, label="sensI")
        ax.plot(t, ssV, label="ssV")
        ax.plot(t, ssI, label="ssI")
        ax.legend()
        plt.show()

    assert np.allclose(sensV, ssV)
    assert np.allclose(sensI, ssI)


def test_lowpass_filter(show_plot=False):
    R = 1.0
    C = 1.0
    builder = collimator.DiagramBuilder()

    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", enable_voltage_port=True)
    r1 = elec.Resistor(ev, name="r1", R=R)
    c1 = elec.Capacitor(
        ev, name="c1", C=C, initial_voltage=0.0, initial_voltage_fixed=True
    )
    ref1 = elec.Ground(ev, name="ref1")
    sensV = elec.VoltageSensor(ev, name="sensV")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    ad.connect(c1, "p", sensV, "p")
    ad.connect(c1, "n", sensV, "n")

    ac = AcausalCompiler(ev, ad)
    lpf = ac()
    lpf = builder.add(lpf)

    sinewave = builder.add(
        lib.Sine(name="sw", amplitude=0.5, frequency=10, bias=1, phase=np.pi / 2)
    )
    integrator = builder.add(lib.Integrator(initial_state=0.0))

    builder.connect(sinewave.output_ports[0], lpf.input_ports[0])
    builder.connect(sinewave.output_ports[0], integrator.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    params = context[lpf.system_id].parameters

    # assert params["v1_V(t)"] == 5.0
    assert params["r1_R"] == R
    assert params["c1_C"] == C

    recorded_signals = {
        "sinewave": sinewave.output_ports[0],
        "lpf": lpf.output_ports[0],
    }
    r = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )

    time = r.time
    sw = r.outputs["sinewave"]
    sensV = r.outputs["lpf"]

    # naively use a discrete low pass filter as the 'expected solution'
    # https://en.wikipedia.org/wiki/Low-pass_filter#Difference_equation_through_discrete_time_sampling
    lfp_sol = np.zeros_like(r.time)
    RC = R * C
    for idx in range(len(r.time)):
        if idx > 0:
            T = r.time[idx] - r.time[idx - 1]
            B = np.exp(-T / RC)
            lfp_sol[idx] = lfp_sol[idx - 1] * B + (1 - B) * r.outputs["sinewave"][idx]

    rel_err = (lfp_sol - sensV) / np.abs(sensV)
    err_cmp_idx = np.argmin(np.abs(time - 2.0))

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(time, sw, label="sw")
        ax1.plot(time, sensV, label="lpf")
        ax1.plot(time, lfp_sol, label="lfp_sol")
        ax1.legend()
        ax1.grid()

        ax2.plot(time, lfp_sol - sensV, label="lfp err")
        ax2.plot(time, rel_err, label="rel_err")
        ax2.legend()
        ax2.grid()
        plt.show()

    # we only check the numerical condition after some time, where the signals correlate better
    assert np.allclose(sensV[err_cmp_idx:], lfp_sol[err_cmp_idx:], atol=0.0, rtol=0.05)


def test_parallel_caps(show_plot=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    c1 = elec.Capacitor(
        ev, name="c1", C=0.5, initial_voltage=0.0, initial_voltage_fixed=True
    )
    c2 = elec.Capacitor(
        ev, name="c2", C=0.5, initial_voltage=0.0, initial_voltage_fixed=True
    )
    ref1 = elec.Ground(ev, name="ref1")
    sensV = elec.VoltageSensor(ev, name="sensV")
    sensI = elec.CurrentSensor(ev, name="sensI")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", sensI, "n")
    ad.connect(sensI, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    ad.connect(c1, "p", sensV, "p")
    ad.connect(c1, "n", sensV, "n")
    ad.connect(c1, "p", c2, "p")
    ad.connect(c1, "n", c2, "n")
    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    lpf = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    lpf = builder.add(lpf)

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    v_idx = lpf.outsym_to_portid[sensV.get_sym_by_port_name("v")]
    i_idx = lpf.outsym_to_portid[sensI.get_sym_by_port_name("i")]
    recorded_signals = {
        "sensV": lpf.output_ports[v_idx],
        "sensI": lpf.output_ports[i_idx],
    }
    options = collimator.SimulatorOptions(ode_solver_method="bdf")
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
        options=options,
    )
    t = results.time
    sensV = results.outputs["sensV"]
    sensI = results.outputs["sensI"]

    def rc_filter(t, v=1, r=1, c=1):
        return v * (1 - np.exp(-t / (r * c)))

    cv_sol = rc_filter(t)

    atol = 0.0002
    rtol = 0

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(t, sensV, label="sensV")
        ax1.plot(t, cv_sol, label="cv_sol")
        ax1.grid()
        ax1.legend()

        ax2.plot(t, sensI, label="sensI")
        ax2.grid()
        ax2.legend()
        plt.show()

    assert np.allclose(cv_sol, sensV, atol=atol, rtol=rtol)


def test_parallel_caps_and_resistors(show_plot=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=2.0)
    r2 = elec.Resistor(ev, name="r2", R=2.0)
    c1 = elec.Capacitor(
        ev, name="c1", C=0.5, initial_voltage=0.0, initial_voltage_fixed=True
    )
    c2 = elec.Capacitor(
        ev, name="c2", C=0.5, initial_voltage=0.0, initial_voltage_fixed=True
    )
    ref1 = elec.Ground(ev, name="ref1")
    sensV = elec.VoltageSensor(ev, name="sensV")
    sensI = elec.CurrentSensor(ev, name="sensI")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", sensI, "n")
    ad.connect(sensI, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    ad.connect(c1, "p", sensV, "p")
    ad.connect(c1, "n", sensV, "n")
    ad.connect(r1, "p", r2, "p")
    ad.connect(r1, "n", r2, "n")
    ad.connect(c1, "p", c2, "p")
    ad.connect(c1, "n", c2, "n")
    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    lpf = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    lpf = builder.add(lpf)

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    v_idx = lpf.outsym_to_portid[sensV.get_sym_by_port_name("v")]
    i_idx = lpf.outsym_to_portid[sensI.get_sym_by_port_name("i")]
    recorded_signals = {
        "sensV": lpf.output_ports[v_idx],
        "sensI": lpf.output_ports[i_idx],
    }
    options = collimator.SimulatorOptions(ode_solver_method="bdf")
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
        options=options,
    )
    t = results.time
    sensV = results.outputs["sensV"]
    sensI = results.outputs["sensI"]

    def rc_filter(t, v=1, r=1, c=1):
        return v * (1 - np.exp(-t / (r * c)))

    cv_sol = rc_filter(t)

    atol = 0.0002
    rtol = 0

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(t, sensV, label="sensV")
        ax1.plot(t, cv_sol, label="cv_sol")
        ax1.grid()
        ax1.legend()

        ax2.plot(t, sensI, label="sensI")
        ax2.grid()
        ax2.legend()
        plt.show()

    assert np.allclose(cv_sol, sensV, atol=atol, rtol=rtol)


def make_circuit_act1(ev):
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1")
    r2 = elec.Resistor(ev, name="r2")
    c1 = elec.Capacitor(
        ev,
        name="c1",
        initial_voltage=0.0,
        initial_voltage_fixed=True,
    )
    l1 = elec.Inductor(
        ev,
        name="l1",
        initial_current=0.0,
        initial_current_fixed=True,
    )
    ref1 = elec.Ground(ev, name="ref1")
    ad.connect(v1, "p", r1, "p")
    ad.connect(v1, "p", l1, "p")
    ad.connect(r1, "n", r2, "p")
    ad.connect(l1, "n", r2, "n")
    ad.connect(r1, "n", c1, "p")
    ad.connect(c1, "n", l1, "n")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")

    return ad


def make_circuit_act2(ev):
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1")
    r2 = elec.Resistor(ev, name="r2")
    r3 = elec.Resistor(ev, name="r3")
    l1 = elec.Inductor(ev, name="l1", initial_current=0.0)
    ref1 = elec.Ground(ev, name="ref1")
    ad.connect(v1, "p", r1, "p")
    ad.connect(v1, "p", l1, "p")
    ad.connect(r1, "n", r2, "p")
    ad.connect(l1, "n", r2, "n")
    ad.connect(r1, "n", r3, "p")
    ad.connect(r3, "n", l1, "n")
    ad.connect(r3, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")

    return ad


def make_circuit_act3(ev):
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1")
    r2 = elec.Resistor(ev, name="r2")
    c1 = elec.Capacitor(ev, name="c1", initial_voltage=0.0, initial_voltage_fixed=True)
    l1 = elec.Inductor(ev, name="l1")
    ref1 = elec.Ground(ev, name="ref1")
    ad.connect(v1, "p", r1, "p")
    ad.connect(v1, "p", c1, "p")
    ad.connect(r1, "n", r2, "p")
    ad.connect(c1, "n", r2, "n")
    ad.connect(r1, "n", l1, "p")
    ad.connect(l1, "n", c1, "n")
    ad.connect(l1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")

    return ad


def run_circuit_act(ev, ad, show_plot=False):
    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    asys = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    asys = builder.add(asys)

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    asys_ctx = context[asys.system_id]

    # run the simulation
    recorded_signals = {
        "x": asys.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    x = results.outputs["x"]

    if show_plot:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 3))
        for i in range(len(asys_ctx.state.continuous_state)):
            label = "x" + str(i)
            ax1.plot(t, x[:, i], label=label)

        ax1.grid()
        ax1.legend()
        plt.show()


def test_simple_circuit_act1(show_plot=False):
    # this is from doc known at collimator as dae1.pdf. The title is:
    #   Lecture 1 – Simulation of differential-algebraic equations
    #   DAE models and differential index
    # this doc has 3 different electrical circuits act 1,2,3.
    # this test implements act1 circuit. tests below implement act2, and act3.
    ev = EqnEnv()
    ad = make_circuit_act1(ev)
    run_circuit_act(ev, ad, show_plot=show_plot)


def test_simple_circuit_act2(show_plot=False):
    # see commnent in act1 test
    ev = EqnEnv()
    ad = make_circuit_act2(ev)
    run_circuit_act(ev, ad, show_plot=show_plot)


@pytest.mark.xfail(reason="not clear how diagram processing should handle this.")
def test_simple_circuit_act3(show_plot=False):
    # see commnent in act1 test
    ev = EqnEnv()
    ad = make_circuit_act3(ev)
    run_circuit_act(ev, ad, show_plot=show_plot)


def make_lc_oscillator(ev):
    ad = AcausalDiagram()
    c1 = elec.Capacitor(
        ev,
        name="c1",
        initial_voltage=1.0,
        initial_voltage_fixed=True,
    )
    l1 = elec.Inductor(
        ev,
        name="l1",
        initial_current=0.0,
        initial_current_fixed=True,
    )
    ref1 = elec.Ground(ev, name="ref1")
    ad.connect(c1, "p", l1, "p")
    ad.connect(c1, "n", l1, "n")
    ad.connect(c1, "n", ref1, "p")

    return ad


def make_lc_oscillator_damped(ev):
    ad = AcausalDiagram()
    c1 = elec.Capacitor(
        ev,
        name="c1",
        initial_voltage=1.0,
        initial_voltage_fixed=True,
    )
    l1 = elec.Inductor(
        ev,
        name="l1",
        initial_current=0.0,
        initial_current_fixed=True,
    )
    r1 = elec.Resistor(ev, name="r1", R=0.2)
    ref1 = elec.Ground(ev, name="ref1")
    ad.connect(c1, "p", r1, "p")
    ad.connect(r1, "n", l1, "p")
    ad.connect(c1, "n", l1, "n")
    ad.connect(c1, "n", ref1, "p")

    return ad


def test_lc_oscillator(show_plot=False):
    # https://en.wikipedia.org/wiki/LC_circuit
    ev = EqnEnv()
    builder = collimator.DiagramBuilder()

    lc_ad = make_lc_oscillator(ev)
    lc_ac = AcausalCompiler(ev, lc_ad, verbose=True)
    lc_sys = lc_ac(name="lc")
    lc_sys = builder.add(lc_sys)

    lcr_ad = make_lc_oscillator_damped(ev)
    lcr_ac = AcausalCompiler(ev, lcr_ad, verbose=True)
    lcr_sys = lcr_ac(name="lcr")
    lcr_sys = builder.add(lcr_sys)

    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    lc_ctx = context[lc_sys.system_id]
    lcr_ctx = context[lcr_sys.system_id]

    recorded_signals = {
        "lc": lc_sys.output_ports[0],
        "lcr": lcr_sys.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    lc = results.outputs["lc"]
    lcr = results.outputs["lcr"]

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        for i in range(len(lc_ctx.state.continuous_state)):
            label = "lc" + str(i)
            ax1.plot(t, lc[:, i], label=label)
        ax1.grid()
        ax1.legend()

        for i in range(len(lcr_ctx.state.continuous_state)):
            label = "lcr" + str(i)
            ax2.plot(t, lcr[:, i], label=label)
        ax2.grid()
        ax2.legend()
        plt.show()


def test_ideal_motor(show_plot=False):
    # voltage_source-motor
    # motor-heat_capacitor
    ev = EqnEnv()
    ad = AcausalDiagram()
    mot = elec.IdealMotor(
        ev,
        R=0.1,
        K=0.5,
        J=2.0,
        enable_heat_port=True,
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
        initial_current=0.0,
        initial_current_fixed=True,
    )
    v = elec.VoltageSource(ev, name="v1", v=50.0, enable_voltage_port=False)
    gnd = elec.Ground(ev, name="gnd")
    hc = ht.HeatCapacitor(
        ev,
        name="hc",
        C=100,
        initial_temperature=300,
        initial_temperature_fixed=True,
    )
    rotSpd = rot.MotionSensor(ev, name="rotSpd", enable_flange_b=False)
    sensI = elec.CurrentSensor(ev, name="sensI")
    ts = ht.TemperatureSensor(ev, name="ts", enable_port_b=False)
    ad.connect(v, "p", sensI, "p")
    ad.connect(sensI, "n", mot, "pos")
    ad.connect(v, "n", mot, "neg")
    ad.connect(v, "n", gnd, "p")
    ad.connect(mot, "shaft", rotSpd, "flange_a")
    ad.connect(mot, "heat", hc, "port")
    ad.connect(mot, "heat", ts, "port_a")

    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    rotSpd_idx = acausal_system.outsym_to_portid[rotSpd.get_sym_by_port_name("w_rel")]
    temp_idx = acausal_system.outsym_to_portid[ts.get_sym_by_port_name("T_rel")]
    I_idx = acausal_system.outsym_to_portid[sensI.get_sym_by_port_name("i")]
    recorded_signals = {
        "rotSpd": acausal_system.output_ports[rotSpd_idx],
        "temp": acausal_system.output_ports[temp_idx],
        "I": acausal_system.output_ports[I_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 4.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    rotSpd = results.outputs["rotSpd"]
    temp = results.outputs["temp"]
    I = results.outputs["I"]  # noqa
    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 3))
        ax1.plot(t, rotSpd, label="rotSpd")
        ax1.legend()
        ax1.grid()
        ax2.plot(t, temp, label="temp")
        ax2.legend()
        ax2.grid()
        ax3.plot(t, I, label="I")
        ax3.legend()
        ax3.grid()
        plt.show()

    assert np.allclose(rotSpd[-1], 100, atol=0.0, rtol=0.05)
    assert np.allclose(temp[-1], 400, atol=0.0, rtol=0.05)
    assert np.allclose(I[-1], 0, atol=4.0, rtol=0.0)


def test_ideal_motor_with_circuit(show_plot=False):
    # squarewave(PWM)->voltage_source->lowpass_filter->ideal_motor->damped_interia
    ev = EqnEnv()
    ad = AcausalDiagram()
    mot = elec.IdealMotor(
        ev,
        R=0.1,
        K=0.5,
        J=0.1,
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
        initial_current=0.0,
        initial_current_fixed=True,
    )
    d1 = rot.Damper(ev, name="d1", D=0.01)
    ref1 = rot.FixedAngle(ev, name="ref1")
    v = elec.VoltageSource(ev, name="v1", enable_voltage_port=True)
    r1 = elec.Resistor(ev, name="r1", R=0.1)
    c1 = elec.Capacitor(
        ev,
        name="c1",
        C=2.0,
        initial_voltage=0.0,
        initial_voltage_fixed=True,
    )
    gnd = elec.Ground(ev, name="gnd")
    rotSpd = rot.MotionSensor(ev, name="rotSpd", enable_flange_b=False)
    sensV = elec.VoltageSensor(ev, name="sensv")
    sensI = elec.CurrentSensor(ev, name="sensI")
    # circuit
    ad.connect(v, "p", r1, "p")
    ad.connect(r1, "n", c1, "p")
    ad.connect(c1, "n", v, "n")
    ad.connect(v, "n", gnd, "p")
    # motor electrical
    ad.connect(c1, "p", sensI, "p")
    ad.connect(sensI, "n", mot, "pos")
    ad.connect(mot, "neg", c1, "n")
    ad.connect(c1, "p", sensV, "p")
    ad.connect(c1, "n", sensV, "n")
    # mot mech
    # ad.connect(mot, "shaft", J, "flange")
    ad.connect(mot, "shaft", rotSpd, "flange_a")
    ad.connect(mot, "shaft", d1, "flange_a")
    ad.connect(d1, "flange_b", ref1, "flange")

    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    vin = builder.add(lib.Sine(amplitude=50.0, bias=50.0, frequency=60))
    builder.connect(vin.output_ports[0], acausal_system.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    rotSpd_idx = acausal_system.outsym_to_portid[rotSpd.get_sym_by_port_name("w_rel")]
    sensV_idx = acausal_system.outsym_to_portid[sensV.get_sym_by_port_name("v")]
    I_idx = acausal_system.outsym_to_portid[sensI.get_sym_by_port_name("i")]
    recorded_signals = {
        "rotSpd": acausal_system.output_ports[rotSpd_idx],
        "sensV": acausal_system.output_ports[sensV_idx],
        "I": acausal_system.output_ports[I_idx],
        "vin": vin.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 4.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    rotSpd = results.outputs["rotSpd"]
    sensV = results.outputs["sensV"]
    I = results.outputs["I"]  # noqa
    vin = results.outputs["vin"]
    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 3))
        ax1.plot(t, rotSpd, label="rotSpd")
        ax1.legend()
        ax1.grid()
        ax2.plot(t, vin, label="vin")
        ax2.plot(t, sensV, label="sensV")
        ax2.legend()
        ax2.grid()
        ax3.plot(t, I, label="I")
        ax3.legend()
        ax3.grid()
        plt.show()

    assert rotSpd[-1] > 95


def test_elec_resistor_thermal_port(show_plot=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", enable_voltage_port=True)
    r1 = elec.Resistor(ev, name="r1", R=1.0, enable_heat_port=True)
    c1 = elec.Capacitor(
        ev, name="c1", C=1.0, initial_voltage=0.0, initial_voltage_fixed=True
    )
    ref1 = elec.Ground(ev, name="ref1")
    hc = ht.HeatCapacitor(
        ev, name="hc", initial_temperature=300, initial_temperature_fixed=True, C=0.1
    )
    sensT = ht.TemperatureSensor(ev, name="sensT", enable_port_b=False)
    ad.connect(v1, "p", r1, "p")
    ad.connect(r1, "n", c1, "p")
    ad.connect(c1, "n", ref1, "p")
    ad.connect(v1, "n", ref1, "p")
    ad.connect(r1, "heat", hc, "port")
    ad.connect(sensT, "port_a", hc, "port")
    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    sinewave = builder.add(
        lib.Sine(name="sw", amplitude=0.5, frequency=10, bias=1, phase=np.pi / 2)
    )
    acausal_system = builder.add(acausal_system)
    builder.connect(sinewave.output_ports[0], acausal_system.input_ports[0])

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    recorded_signals = {
        "sensT": acausal_system.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    sensT = results.outputs["sensT"]

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(t, sensT, label="sensT")
        ax.legend()
        plt.show()

    assert np.allclose(sensT[0], 300.0)
    assert sensT[-1] > 315.0
    assert np.all(np.diff(sensT) >= 0.0)


def test_battery_pulse(show_plot=False):
    # current pulses and rest for battery.
    ev = EqnEnv()
    ad = AcausalDiagram()
    batt = elec.Battery(
        ev,
        name="batt",
        AH=0.1,
        R=0.01,
        Rp=0.01,
        Cp=30.0,
        enable_soc_port=True,
        initial_soc=0.5,
        initial_soc_fixed=True,
        enable_ocv_port=True,
        enable_Up_port=True,
    )
    r1 = elec.Resistor(ev, name="r1", R=0.1)
    cs = elec.CurrentSource(ev, name="cs", enable_current_port=True)
    sensV = elec.VoltageSensor(ev, name="sensV")
    sensI = elec.CurrentSensor(ev, name="sensI")
    gnd = elec.Ground(ev, name="gnd")

    ad.connect(cs, "p", sensI, "p")
    ad.connect(sensI, "n", r1, "p")
    ad.connect(r1, "n", batt, "p")
    ad.connect(batt, "n", gnd, "p")
    ad.connect(batt, "n", cs, "n")
    ad.connect(batt, "p", sensV, "p")
    ad.connect(batt, "n", sensV, "n")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    pulse = builder.add(lib.Pulse(period=4.0))
    poffset = builder.add(lib.Offset(offset=-1.0))
    pgain = builder.add(lib.Step(start_value=-1.0, end_value=1.0, step_time=5.0))
    curr = builder.add(lib.Product(n_in=2))
    builder.connect(pulse.output_ports[0], poffset.input_ports[0])
    builder.connect(poffset.output_ports[0], curr.input_ports[0])
    builder.connect(pgain.output_ports[0], curr.input_ports[1])
    builder.connect(curr.output_ports[0], acausal_system.input_ports[0])

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    soc_idx = acausal_system.outsym_to_portid[batt.get_sym_by_port_name("soc")]
    Up_idx = acausal_system.outsym_to_portid[batt.get_sym_by_port_name("Up")]
    ocv_idx = acausal_system.outsym_to_portid[batt.get_sym_by_port_name("ocv")]
    v_idx = acausal_system.outsym_to_portid[sensV.get_sym_by_port_name("v")]
    i_idx = acausal_system.outsym_to_portid[sensI.get_sym_by_port_name("i")]
    recorded_signals = {
        "soc": acausal_system.output_ports[soc_idx],
        "Up": acausal_system.output_ports[Up_idx],
        "ocv": acausal_system.output_ports[ocv_idx],
        "sensV": acausal_system.output_ports[v_idx],
        "sensI": acausal_system.output_ports[i_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    soc = results.outputs["soc"]
    Up = results.outputs["Up"]
    ocv = results.outputs["ocv"]
    sensV = results.outputs["sensV"]
    sensI = results.outputs["sensI"]

    if show_plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12))
        ax1.plot(t, soc, label="SOC")
        ax1.legend()
        ax1.grid()

        ax2.plot(t, sensI, label="Current")
        ax2.legend()
        ax2.grid()

        ax3.plot(t, sensV, label="Terminal Voltage")
        ax3.plot(t, ocv, label="Open Circuit Voltage")
        ax3.legend()
        ax3.grid()

        ax4.plot(t, Up, label="RC voltage")
        ax4.legend()
        ax4.grid()
        plt.show()

    soc_sol_t = np.array([0, 2, 4, 6, 8, 10])
    soc_sol = np.array([0.5, 0.5, 0.45, 0.45, 0.5, 0.5])
    soc_sol = np.interp(t, soc_sol_t, soc_sol)
    # print(f"{soc-soc_sol}")
    assert np.allclose(soc, soc_sol, rtol=0.05, atol=0.05)

    discharge_idx = np.where(sensI < 0)
    charge_idx = np.where(sensI > 0)
    assert np.all(sensV[discharge_idx] <= ocv[discharge_idx])
    assert np.all(sensV[charge_idx] >= ocv[charge_idx])


def test_bldc(show_plot=False):
    # voltage_source-motor
    # motor-heat_capacitor
    ev = EqnEnv()
    ad = AcausalDiagram()
    mot = elec.BLDC(ev, enable_heat_port=True)
    volts = 500
    v = elec.VoltageSource(ev, name="v1", v=volts, enable_voltage_port=False)
    gnd = elec.Ground(ev, name="gnd")
    hc = ht.HeatCapacitor(
        ev,
        name="hc",
        C=100,
        initial_temperature=300,
        initial_temperature_fixed=True,
    )
    jj = rot.Inertia(
        ev,
        "jj",
        I=0.1,
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    dd = rot.Damper(ev, name="dd", D=0.01)
    rotRef = rot.FixedAngle(ev)
    rotSpd = rot.MotionSensor(ev, name="rotSpd", enable_flange_b=False)
    rotTrq = rot.TorqueSensor(ev, "rotTrq")
    sensI = elec.CurrentSensor(ev, name="sensI")
    ts = ht.TemperatureSensor(ev, name="ts", enable_port_b=False)
    ad.connect(v, "p", sensI, "p")
    ad.connect(sensI, "n", mot, "pos")
    ad.connect(v, "n", mot, "neg")
    ad.connect(v, "n", gnd, "p")
    ad.connect(mot, "shaft", rotSpd, "flange_a")
    ad.connect(mot, "shaft", rotTrq, "flange_a")
    ad.connect(rotTrq, "flange_b", jj, "flange")
    ad.connect(jj, "flange", dd, "flange_a")
    ad.connect(dd, "flange_b", rotRef, "flange")
    ad.connect(mot, "heat", hc, "port")
    ad.connect(mot, "heat", ts, "port_a")

    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    trq_req_norm = builder.add(lib.Step(start_value=0.5, end_value=-0.2, step_time=2.0))
    builder.connect(trq_req_norm.output_ports[0], acausal_system.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    rotSpd_idx = acausal_system.outsym_to_portid[rotSpd.get_sym_by_port_name("w_rel")]
    rotTrq_idx = acausal_system.outsym_to_portid[rotTrq.get_sym_by_port_name("tau")]
    temp_idx = acausal_system.outsym_to_portid[ts.get_sym_by_port_name("T_rel")]
    I_idx = acausal_system.outsym_to_portid[sensI.get_sym_by_port_name("i")]
    recorded_signals = {
        "rotSpd": acausal_system.output_ports[rotSpd_idx],
        "rotTrq": acausal_system.output_ports[rotTrq_idx],
        "temp": acausal_system.output_ports[temp_idx],
        "I": acausal_system.output_ports[I_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 4.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    rotSpd = results.outputs["rotSpd"]
    rotTrq = results.outputs["rotTrq"]
    temp = results.outputs["temp"]
    print("computing plot results")
    I = results.outputs["I"]  # noqa
    elec_pwr = I * volts
    mech_pwr = rotSpd * rotTrq
    eff = np.abs(mech_pwr) / np.clip(np.abs(elec_pwr), 1.0, None)

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    file_name = "bldc_pass_conditions.npz"
    file_path = os.path.join(current_directory, file_name)
    sols = np.load(file_path)

    t_sol = sols["t"]
    rotSpd_sol = sols["rotSpd"]
    rotTrq_sol = sols["rotTrq"]
    I_sol = sols["I"]
    eff_sol = sols["eff"]
    if show_plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12))
        ax1.plot(t_sol, rotSpd_sol, label="rotSpd_sol")
        ax1.plot(t, rotSpd, label="rotSpd")
        ax1.legend()
        ax1.grid()
        ax2.plot(t, temp, label="temp")
        ax2.plot(t_sol, rotTrq_sol, label="rotTrq_sol")
        ax2.plot(t, rotTrq, label="rotTrq")
        ax2.legend()
        ax2.grid()
        ax3.plot(t_sol, I_sol, label="I_sol")
        ax3.plot(t, I, label="I")
        ax3.legend()
        ax3.grid()
        ax4.plot(t_sol, eff_sol, label="eff_sol")
        ax4.plot(t, eff, label="eff")
        ax4.legend()
        ax4.grid()
        plt.show()

    # get expected sol on same time samples as results
    rotSpd_sol = np.interp(t, t_sol, rotSpd_sol)
    rotTrq_sol = np.interp(t, t_sol, rotTrq_sol)
    I_sol = np.interp(t, t_sol, I_sol)
    eff_sol = np.interp(t, t_sol, eff_sol)

    assert np.allclose(rotSpd, rotSpd_sol, rtol=1e-2, atol=1)
    # FIXME: these pass conditions below are not behaving.
    # assert np.allclose(rotTrq[10:-10], rotTrq_sol[10:-10], rtol=1e-2, atol=1)
    # assert np.allclose(I[10:-10], I_sol[10:-10], rtol=1e-2, atol=1)
    # t2_idx = np.argmin(np.abs(t - 2.0))
    # idx_buf = 10
    # idx1 = t2_idx - idx_buf
    # idx2 = t2_idx + idx_buf
    # assert np.allclose(eff[idx_buf:idx1], eff_sol[idx_buf:idx1], rtol=1e-2, atol=0.05)
    # assert np.allclose(eff[idx2:-idx_buf], eff_sol[idx2:-idx_buf], rtol=1e-2, atol=0.05)
    # np.savez(
    #     file_path, t=t, rotSpd=rotSpd, rotTrq=rotTrq, I=I, eff=eff
    # )


def test_ideal_diode(show_plot=False, ideal=False, run_sim=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", enable_voltage_port=True, v=-5.0)
    r1 = elec.Resistor(ev, name="r1", R=1)
    c1 = elec.Capacitor(ev, name="c1", initial_voltage=0.0, initial_voltage_fixed=True)
    if ideal:
        d1 = elec.IdealDiode(ev, name="d1", Ron=0.01, Roff=100.0)
    else:
        d1 = elec.Diode(ev, name="d1")
    # use a resistor in place of the diode to test the rest of the circuit.
    # d1 = elec.Resistor(ev, name="d1", R=1.0)
    sensI = elec.CurrentSensor(ev, name="sensI")
    sensV = elec.VoltageSensor(ev, name="sensV")
    capV = elec.VoltageSensor(ev, name="capV")
    rcV = elec.VoltageSensor(ev, name="rcV")
    allV = elec.VoltageSensor(ev, name="allV")
    ref1 = elec.Ground(ev, name="ref1")

    ad.connect(v1, "p", d1, "p")
    ad.connect(d1, "n", sensI, "n")
    ad.connect(sensI, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    # voltage sensors
    # diode voltage
    ad.connect(d1, "p", sensV, "p")
    ad.connect(d1, "n", sensV, "n")
    # cap voltage
    ad.connect(c1, "p", capV, "p")
    ad.connect(c1, "n", capV, "n")
    # RC voltage
    ad.connect(r1, "n", rcV, "p")
    ad.connect(c1, "n", rcV, "n")
    # diode+R+C voltage, should be 5v
    ad.connect(d1, "p", allV, "p")
    ad.connect(ref1, "p", allV, "n")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    asys = ac()

    if run_sim:
        # FIXME: see FIXME in IdealDiode and Diode classes.
        # make wildcat diagram
        builder = collimator.DiagramBuilder()
        asys = builder.add(asys)
        vin = builder.add(lib.Sine(frequency=10.0, amplitude=10.0))
        builder.connect(vin.output_ports[0], asys.input_ports[0])

        # 'compile' wildcat diagram
        diagram = builder.build()
        context = diagram.create_context(check_types=True)

        # run the simulation
        I_idx = asys.outsym_to_portid[sensI.get_sym_by_port_name("i")]
        V_idx = asys.outsym_to_portid[sensV.get_sym_by_port_name("v")]
        capV_idx = asys.outsym_to_portid[capV.get_sym_by_port_name("v")]
        rcV_idx = asys.outsym_to_portid[rcV.get_sym_by_port_name("v")]
        allV_idx = asys.outsym_to_portid[allV.get_sym_by_port_name("v")]
        recorded_signals = {
            "I": asys.output_ports[I_idx],
            "V": asys.output_ports[V_idx],
            "capV": asys.output_ports[capV_idx],
            "rcV": asys.output_ports[rcV_idx],
            "allV": asys.output_ports[allV_idx],
        }
        results = collimator.simulate(
            diagram,
            context,
            (0.0, 10.0),
            recorded_signals=recorded_signals,
        )
        t = results.time
        I_ = results.outputs["I"]
        V_ = results.outputs["V"]
        capV_ = results.outputs["capV"]
        rcV = results.outputs["rcV"]
        allV = results.outputs["allV"]

        if show_plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            ax1.plot(t, I_, label="current")
            ax1.legend()
            ax1.grid()
            ax2.plot(t, V_, label="voltage across diode", marker="o")
            ax2.plot(t, capV_, label="voltage across capacitor")
            ax2.plot(t, rcV, label="voltage across resistor+capacitor")
            ax2.plot(t, allV, label="voltage across D+R+C")
            ax2.legend()
            ax2.grid()
            plt.show()


if __name__ == "__main__":
    show_plot = True
    # test_basic_RC(show_plot=show_plot)
    # test_basic_RC_with_outputs(show_plot=True)
    # test_basic_RC_with_voltage_input(show_plot=show_plot)
    # test_RLC_circuit(show_plot=show_plot)
    # test_lowpass_filter(show_plot=show_plot)
    # test_parallel_caps(show_plot=True)
    # test_parallel_caps_and_resistors(show_plot=True)
    # test_simple_circuit_act1(show_plot=True)
    # test_simple_circuit_act2(show_plot=False)
    # test_simple_circuit_act3(show_plot=True)
    # test_lc_oscillator(show_plot=True)
    # test_ideal_motor(show_plot=show_plot)
    # test_ideal_motor_with_circuit(show_plot=True)
    # test_elec_resistor_thermal_port(show_plot=show_plot)
    test_battery_pulse(show_plot=True)
    # test_bldc(show_plot=show_plot)
    # test_ideal_diode(show_plot=show_plot)
