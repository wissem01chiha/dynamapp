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

# acausal imports
from collimator.experimental import AcausalCompiler, AcausalDiagram, EqnEnv
from collimator.experimental import translational as trans
from collimator.experimental import rotational as rot

# collimator imports
import collimator
from collimator import library as lib
from collimator.backend.typing import ArrayLike

import collimator.logging as logging

logging.set_log_level(logging.DEBUG)


@pytest.mark.parametrize("fixed_angle_ic", [0.0, -1.0])
def test_rot_oscillator_with_outputs(fixed_angle_ic, show_plot=False):
    # basic test of self contained system.
    # mass-spring-force_sensor-wall. spring is initially strecthed.
    # force sensors measures force of spring, which is same as force everywhere else.
    # speed sensor measures speed of mass.

    # make acausal diagram
    ev = EqnEnv()
    ad = AcausalDiagram()
    m1 = rot.Inertia(
        ev,
        name="m1",
        I=1.0,
        initial_angle=1.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    sp1 = rot.Spring(ev, name="sp1", K=1.0)
    r1 = rot.FixedAngle(ev, name="r1", initial_angle=fixed_angle_ic)
    spdsnsr = rot.MotionSensor(
        ev,
        name="spdsnsr",
        enable_flange_b=True,
        enable_angle_port=True,
        enable_acceleration_port=True,
    )
    frcsnsr = rot.TorqueSensor(ev, name="trqsnsr")
    ad.connect(m1, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", frcsnsr, "flange_a")
    ad.connect(frcsnsr, "flange_b", r1, "flange")
    ad.connect(m1, "flange", spdsnsr, "flange_a")
    ad.connect(r1, "flange", spdsnsr, "flange_b")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    mech_oscillator = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    mech_oscillator = builder.add(mech_oscillator)

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run simulation
    alpha_idx = mech_oscillator.outsym_to_portid[
        spdsnsr.get_sym_by_port_name("alpha_rel")
    ]
    spd_idx = mech_oscillator.outsym_to_portid[spdsnsr.get_sym_by_port_name("w_rel")]
    ang_idx = mech_oscillator.outsym_to_portid[spdsnsr.get_sym_by_port_name("ang_rel")]
    frc_idx = mech_oscillator.outsym_to_portid[frcsnsr.get_sym_by_port_name("tau")]
    recorded_signals = {
        "alpha": mech_oscillator.output_ports[alpha_idx],
        "spd": mech_oscillator.output_ports[spd_idx],
        "ang": mech_oscillator.output_ports[ang_idx],
        "frc": mech_oscillator.output_ports[frc_idx],
    }
    t0, tf = 0.0, 10.0
    results = collimator.simulate(
        diagram,
        context,
        (t0, tf),
        recorded_signals=recorded_signals,
    )
    t = results.time
    alpha = results.outputs["alpha"]
    spd = results.outputs["spd"]
    ang = results.outputs["ang"]
    frc = results.outputs["frc"]

    if fixed_angle_ic == 0.0:
        frc_sol = np.cos(t)
        spd_sol = np.sin(t) * -1
        ang_sol = np.cos(t)
        alpha_sol = np.cos(t) * -1
        atol = 3e-5
        rtol = 0.0
    elif fixed_angle_ic == -1.0:
        frc_sol = np.cos(t) * 2.0
        spd_sol = np.sin(t) * -2.0
        ang_sol = np.cos(t) * 2.0
        alpha_sol = np.cos(t) * -2.0
        atol = 4e-5
        rtol = 0.0
    else:
        raise ValueError(
            f"test test_rot_oscillator_with_outputs does not have pass conditions for {fixed_angle_ic=}."
        )

    if show_plot:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(
            8, 1, figsize=(8, 8)
        )
        ax1.plot(t, frc_sol, label="frc_sol")
        ax1.plot(t, frc, label="frc")
        ax1.grid()
        ax1.legend()
        ax2.plot(t, frc_sol - frc)
        ax2.grid()

        ax3.plot(t, spd_sol, label="spd_sol")
        ax3.plot(t, spd, label="spd")
        ax3.grid()
        ax3.legend()
        ax4.plot(t, spd_sol - spd)
        ax4.grid()

        ax5.plot(t, alpha_sol, label="alpha_sol")
        ax5.plot(t, alpha, label="alpha")
        ax5.grid()
        ax5.legend()
        ax6.plot(t, alpha_sol - alpha)
        ax6.grid()

        ax7.plot(t, ang_sol, label="ang_sol")
        ax7.plot(t, ang, label="ang")
        ax7.grid()
        ax7.legend()
        ax8.plot(t, ang_sol - ang)
        ax8.grid()

        plt.show()

    assert np.allclose(frc_sol, frc, atol=atol, rtol=rtol)
    assert np.allclose(spd_sol, spd, atol=atol, rtol=rtol)
    assert np.allclose(alpha_sol, alpha, atol=atol, rtol=rtol)
    assert np.allclose(ang_sol, ang, atol=atol, rtol=rtol)


def test_basic_engine(show_plot=False):
    # engine-trq_sensor-inertia-rot_damper-fixed_angle
    ev = EqnEnv()
    ad = AcausalDiagram()

    ice = rot.BasicEngine(ev)
    jj = rot.Inertia(
        ev,
        name="J",
        I=0.1,
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=200.0,
        initial_velocity_fixed=True,
    )
    d1 = rot.Damper(ev, D=0.05)
    ref1 = rot.FixedAngle(ev)
    rotSpd = rot.MotionSensor(ev, name="rotSpd", enable_flange_b=False)
    sensTrq = rot.TorqueSensor(ev, name="sensTrq")
    ad.connect(ice, "flange", sensTrq, "flange_a")
    ad.connect(sensTrq, "flange_b", jj, "flange")
    ad.connect(jj, "flange", d1, "flange_a")
    ad.connect(d1, "flange_b", ref1, "flange")
    ad.connect(jj, "flange", rotSpd, "flange_a")

    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac(leaf_backend="jax")
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    thrIn = builder.add(lib.Constant(value=1.0))
    builder.connect(thrIn.output_ports[0], acausal_system.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # verify acausal diagram params are in the acausal_system context
    params = context[acausal_system.system_id].parameters
    print(params)
    assert isinstance(params["BasicEngine_peak_trq_w"], ArrayLike)
    assert isinstance(params["BasicEngine_peak_trq_t"], ArrayLike)

    rotSpd_idx = acausal_system.outsym_to_portid[rotSpd.get_sym_by_port_name("w_rel")]
    sensTrq_idx = acausal_system.outsym_to_portid[sensTrq.get_sym_by_port_name("tau")]
    recorded_signals = {
        "rotSpd": acausal_system.output_ports[rotSpd_idx],
        "sensTrq": acausal_system.output_ports[sensTrq_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 4.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    rotSpd = results.outputs["rotSpd"]
    sensTrq = results.outputs["sensTrq"]

    print(f"{rotSpd[0]=}")
    print(f"{rotSpd[-1]=}")
    print(f"{sensTrq[0]=}")
    print(f"{sensTrq[-1]=}")
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(t, rotSpd, label="rotSpd")
        ax1.legend()
        ax1.grid()
        ax2.plot(t, sensTrq, label="sensTrq")
        ax2.legend()
        ax2.grid()
        plt.show()

    assert np.allclose(rotSpd[0], 200)
    assert np.allclose(rotSpd[-1], 571, atol=0.0, rtol=0.05)
    assert np.allclose(sensTrq[0], 100)
    assert np.allclose(sensTrq[-1], 28.57, atol=0.0, rtol=0.05)


def test_ideal_wheel(show_plot=False):
    # torque_source-inertia-wheel-mass-damper-ref
    ev = EqnEnv()
    ad = AcausalDiagram()
    trq = rot.TorqueSource(ev, name="trq", tau=1, enable_flange_b=False)
    inertia = rot.Inertia(
        ev,
        name="inertia",
        initial_angle=0.0,
        initial_angle_fixed=True,
    )
    whl = rot.IdealWheel(ev, name="whl")
    mass = trans.Mass(
        ev,
        name="mass",
        initial_position=0.0,
        initial_position_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    td = trans.Damper(ev, name="td")
    transRef = trans.FixedPosition(ev, name="transRef")
    rspd = rot.MotionSensor(ev, name="rspd", enable_flange_b=False)
    tspd = trans.MotionSensor(ev, name="tspd", enable_flange_b=False)
    ad.connect(trq, "flange_a", inertia, "flange")
    ad.connect(trq, "flange_a", whl, "shaft")
    ad.connect(whl, "flange", mass, "flange")
    ad.connect(mass, "flange", td, "flange_a")
    ad.connect(td, "flange_b", transRef, "flange")
    ad.connect(trq, "flange_a", rspd, "flange_a")
    ad.connect(mass, "flange", tspd, "flange_a")

    ac = AcausalCompiler(ev, ad, verbose=False)
    acausal_system = ac()
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    rspd_idx = acausal_system.outsym_to_portid[rspd.get_sym_by_port_name("w_rel")]
    tspd_idx = acausal_system.outsym_to_portid[tspd.get_sym_by_port_name("v_rel")]
    recorded_signals = {
        "rspd": acausal_system.output_ports[rspd_idx],
        "tspd": acausal_system.output_ports[tspd_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    rspd = results.outputs["rspd"]
    tspd = results.outputs["tspd"]
    if show_plot:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(t, rspd, label="rspd")
        ax1.legend()
        ax1.grid()
        ax3.plot(t, tspd, label="tspd")
        ax3.legend()
        ax3.grid()
        plt.show()

    assert np.allclose(rspd, tspd)


def test_ideal_gear(show_plot=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    trq = rot.TorqueSource(ev, name="trq", tau=1, enable_flange_b=False)
    inertia1 = rot.Inertia(
        ev,
        name="inertia1",
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    gr = rot.IdealGear(ev, r=2.0)
    inertia2 = rot.Inertia(
        ev,
        name="inertia2",
        initial_angle=0.0,
        initial_angle_fixed=True,
    )
    rspd1 = rot.MotionSensor(ev, name="rspd1", enable_flange_b=False)
    rspd2 = rot.MotionSensor(ev, name="rspd2", enable_flange_b=False)
    ad.connect(trq, "flange_a", inertia1, "flange")
    ad.connect(inertia1, "flange", gr, "flange_a")
    ad.connect(inertia1, "flange", rspd1, "flange_a")
    ad.connect(gr, "flange_b", inertia2, "flange")
    ad.connect(inertia2, "flange", rspd2, "flange_a")

    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    rspd1_idx = acausal_system.outsym_to_portid[rspd1.get_sym_by_port_name("w_rel")]
    rspd2_idx = acausal_system.outsym_to_portid[rspd2.get_sym_by_port_name("w_rel")]
    recorded_signals = {
        "rspd1": acausal_system.output_ports[rspd1_idx],
        "rspd2": acausal_system.output_ports[rspd2_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    rspd1 = results.outputs["rspd1"]
    rspd2 = results.outputs["rspd2"]
    if show_plot:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(t, rspd1, label="rspd1")
        ax1.legend()
        ax1.grid()
        ax3.plot(t, rspd2, label="rspd2")
        ax3.legend()
        ax3.grid()
        plt.show()

    assert np.allclose(rspd1, rspd2 * 2)


def test_ideal_planetary(show_plot=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    trq = rot.TorqueSource(ev, name="trq", tau=1, enable_flange_b=False)
    inertia1 = rot.Inertia(
        ev,
        name="inertia1",
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    inertia2 = rot.Inertia(
        ev,
        name="inertia2",
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    inertia3 = rot.Inertia(
        ev,
        name="inertia3",
        initial_angle=0.0,
        initial_angle_fixed=True,
    )
    planetary = rot.IdealPlanetary(ev, r=2.0)

    rspd1 = rot.MotionSensor(ev, name="rspd1", enable_flange_b=False)
    rspd2 = rot.MotionSensor(ev, name="rspd2", enable_flange_b=False)
    rspd3 = rot.MotionSensor(ev, name="rspd3", enable_flange_b=False)

    # input at carrier
    ad.connect(trq, "flange_a", inertia1, "flange")
    ad.connect(inertia1, "flange", planetary, "carrier")
    ad.connect(inertia1, "flange", rspd1, "flange_a")

    # output at sun
    ad.connect(planetary, "sun", inertia2, "flange")
    ad.connect(inertia2, "flange", rspd2, "flange_a")

    # output at ring
    ad.connect(planetary, "ring", inertia3, "flange")
    ad.connect(inertia3, "flange", rspd3, "flange_a")

    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    rspd1_idx = acausal_system.outsym_to_portid[rspd1.get_sym_by_port_name("w_rel")]
    rspd2_idx = acausal_system.outsym_to_portid[rspd2.get_sym_by_port_name("w_rel")]
    rspd3_idx = acausal_system.outsym_to_portid[rspd3.get_sym_by_port_name("w_rel")]
    recorded_signals = {
        "rspd1": acausal_system.output_ports[rspd1_idx],
        "rspd2": acausal_system.output_ports[rspd2_idx],
        "rspd3": acausal_system.output_ports[rspd3_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    rspd1 = results.outputs["rspd1"]
    rspd2 = results.outputs["rspd2"]
    rspd3 = results.outputs["rspd3"]
    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 3))
        ax1.plot(t, rspd1, label="rspd1")
        ax1.legend()
        ax1.grid()
        ax2.plot(t, rspd2, label="rspd2")
        ax2.legend()
        ax2.grid()
        ax3.plot(t, rspd3, label="rspd3")
        ax3.legend()
        ax3.grid()
        plt.show()

    # print(f"{rspd1[-1]=}")
    # print(f"{rspd2[-1]=}")
    # print(f"{rspd3[-1]=}")
    # print(f"{rspd1[-1]/rspd2[-1]}")
    # print(f"{rspd3[-1]/rspd2[-1]}")

    assert np.allclose(rspd1, rspd2 * 1.666666666666667, atol=0.01, rtol=0.001)
    assert np.allclose(rspd3, rspd2 * 2.0, atol=0.01, rtol=0.001)


def test_gear(show_plot=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    trq = rot.TorqueSource(ev, name="trq", tau=1, enable_flange_b=False)
    inertia1 = rot.Inertia(
        ev,
        name="inertia1",
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    gr = rot.Gear(ev, r=1.0)
    inertia2 = rot.Inertia(
        ev,
        name="inertia2",
        initial_angle=0.0,
        initial_angle_fixed=True,
    )
    rspd1 = rot.MotionSensor(ev, name="rspd1", enable_flange_b=False)
    rspd2 = rot.MotionSensor(ev, name="rspd2", enable_flange_b=False)
    rtrq1 = rot.TorqueSensor(ev, "rtrq1")
    rtrq2 = rot.TorqueSensor(ev, "rtrq2")
    ad.connect(trq, "flange_a", inertia1, "flange")
    ad.connect(inertia1, "flange", rtrq1, "flange_a")
    ad.connect(rtrq1, "flange_b", gr, "flange_a")
    ad.connect(inertia1, "flange", rspd1, "flange_a")
    ad.connect(gr, "flange_b", rtrq2, "flange_a")
    ad.connect(rtrq2, "flange_b", inertia2, "flange")
    ad.connect(inertia2, "flange", rspd2, "flange_a")

    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    rspd1_idx = acausal_system.outsym_to_portid[rspd1.get_sym_by_port_name("w_rel")]
    rspd2_idx = acausal_system.outsym_to_portid[rspd2.get_sym_by_port_name("w_rel")]
    rtrq1_idx = acausal_system.outsym_to_portid[rtrq1.get_sym_by_port_name("tau")]
    rtrq2_idx = acausal_system.outsym_to_portid[rtrq2.get_sym_by_port_name("tau")]
    recorded_signals = {
        "rspd1": acausal_system.output_ports[rspd1_idx],
        "rspd2": acausal_system.output_ports[rspd2_idx],
        "rtrq1": acausal_system.output_ports[rtrq1_idx],
        "rtrq2": acausal_system.output_ports[rtrq2_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    rspd1 = results.outputs["rspd1"]
    rspd2 = results.outputs["rspd2"]
    rtrq1 = results.outputs["rtrq1"]
    rtrq2 = results.outputs["rtrq2"]

    print(f"{rtrq1=}")
    print(f"{rtrq2=}")
    print(f"{rspd1=}")
    print(f"{rspd2=}")

    if show_plot:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(t, rspd1, label="rspd1")
        ax1.plot(t, rspd2, label="rspd2")
        ax1.legend()
        ax1.grid()
        ax3.plot(t, rtrq1, label="rtrq1")
        ax3.plot(t, rtrq2, label="rtrq2")
        ax3.legend()
        ax3.grid()
        plt.show()

    assert np.allclose(rspd1, rspd2, rtol=1e-3, atol=0.001)
    # the first X samples don't meet the condition
    assert np.allclose(rtrq2[5:], rtrq1[5:] * 0.98, rtol=1e-4, atol=0.02)


def test_friction(show_plot=False, f_sinusoidal=False):
    # make two systems, one without viscous friction, and one with.
    # can be confirgured to apply sinusoidal force, but no pass conditions for this,
    # only for manual inspection of results
    def make_sys(name, C=None, f_sinusoidal=False):
        # force-mass-friction-force_sensor-ref
        ev = EqnEnv()
        ad = AcausalDiagram(name=name)
        if f_sinusoidal:
            f1 = rot.TorqueSource(
                ev, name="t1", enable_flange_b=False, enable_torque_port=True
            )
        else:
            f1 = rot.TorqueSource(ev, name="t1", tau=10.0, enable_flange_b=False)
        m1 = rot.Inertia(
            ev,
            name="m1",
            I=1.0,
            initial_angle=0.0,
            initial_angle_fixed=True,
            initial_velocity=0.0,
            initial_velocity_fixed=True,
        )
        friction1 = rot.Friction(ev, name="friction1", C=C)
        r1 = rot.FixedAngle(ev, name="r1", initial_angle=0.0)
        spdsnsr = rot.MotionSensor(
            ev,
            name="spdsnsr",
            enable_flange_b=False,
            enable_angle_port=True,
            enable_acceleration_port=True,
        )
        frcsnsr = rot.TorqueSensor(ev, name="frcsnsr")
        ad.connect(f1, "flange_a", m1, "flange")
        ad.connect(m1, "flange", friction1, "flange_a")
        ad.connect(friction1, "flange_b", frcsnsr, "flange_a")
        ad.connect(frcsnsr, "flange_b", r1, "flange")
        ad.connect(m1, "flange", spdsnsr, "flange_a")
        ac = AcausalCompiler(ev, ad, verbose=True)
        sys = ac(name=name)

        return sys, spdsnsr, frcsnsr

    coul_sys, css, cfs = make_sys("coul_sys", f_sinusoidal=f_sinusoidal)
    visc_sys, vss, vfs = make_sys("visc_sys", C=1.0, f_sinusoidal=f_sinusoidal)

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    coul_sys = builder.add(coul_sys)
    visc_sys = builder.add(visc_sys)
    if f_sinusoidal:
        fin = builder.add(lib.Sine(frequency=10.0, amplitude=20.0))
        builder.connect(fin.output_ports[0], coul_sys.input_ports[0])
        builder.connect(fin.output_ports[0], visc_sys.input_ports[0])

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run simulation
    acc_idx = coul_sys.outsym_to_portid[css.get_sym_by_port_name("alpha_rel")]
    spd_idx = coul_sys.outsym_to_portid[css.get_sym_by_port_name("w_rel")]
    pos_idx = coul_sys.outsym_to_portid[css.get_sym_by_port_name("ang_rel")]
    frc_idx = coul_sys.outsym_to_portid[cfs.get_sym_by_port_name("tau")]
    acc2_idx = visc_sys.outsym_to_portid[vss.get_sym_by_port_name("alpha_rel")]
    spd2_idx = visc_sys.outsym_to_portid[vss.get_sym_by_port_name("w_rel")]
    pos2_idx = visc_sys.outsym_to_portid[vss.get_sym_by_port_name("ang_rel")]
    frc2_idx = visc_sys.outsym_to_portid[vfs.get_sym_by_port_name("tau")]
    recorded_signals = {
        "acc": coul_sys.output_ports[acc_idx],
        "spd": coul_sys.output_ports[spd_idx],
        "pos": coul_sys.output_ports[pos_idx],
        "frc": coul_sys.output_ports[frc_idx],
        "acc2": visc_sys.output_ports[acc2_idx],
        "spd2": visc_sys.output_ports[spd2_idx],
        "pos2": visc_sys.output_ports[pos2_idx],
        "frc2": visc_sys.output_ports[frc2_idx],
    }
    t0, tf = 0.0, 10.0
    results = collimator.simulate(
        diagram,
        context,
        (t0, tf),
        recorded_signals=recorded_signals,
    )
    t = results.time
    acc = results.outputs["acc"]
    spd = results.outputs["spd"]
    pos = results.outputs["pos"]
    frc = results.outputs["frc"]
    acc2 = results.outputs["acc2"]
    spd2 = results.outputs["spd2"]
    pos2 = results.outputs["pos2"]
    frc2 = results.outputs["frc2"]

    print(f"{frc[-1]=}")
    print(f"{frc2[-1]=}")
    print(f"{spd2[-1]=}")
    print(f"{acc[-1]=}")
    print(f"{acc2[-1]=}")
    if show_plot:
        fig, (ax1, ax3, ax5, ax7) = plt.subplots(4, 1, figsize=(8, 8))
        ax1.plot(t, frc, label="frc")
        ax1.plot(t, frc2, label="frc2")
        ax1.grid()
        ax1.legend()

        ax3.plot(t, spd, label="spd")
        ax3.plot(t, spd2, label="spd2")
        ax3.grid()
        ax3.legend()

        ax5.plot(t, pos, label="pos")
        ax5.plot(t, pos2, label="pos2")
        ax5.grid()
        ax5.legend()

        ax7.plot(t, acc, label="acc")
        ax7.plot(t, acc2, label="acc2")
        ax7.grid()
        ax7.legend()

        plt.show()

    atol = 1e-3
    rtol = 0.0
    assert np.allclose(frc[-1], 1.0, atol=atol, rtol=rtol)
    assert np.allclose(frc2[-1], 10.0, atol=atol, rtol=rtol)
    assert np.allclose(spd2[-1], 9.0, atol=atol, rtol=rtol)
    assert np.allclose(acc[-1], 9.0, atol=atol, rtol=rtol)
    assert np.allclose(acc2[-1], 0.0, atol=atol, rtol=rtol)


if __name__ == "__main__":
    show_plot = True
    # test_rot_oscillator_with_outputs(0.0, show_plot=show_plot)
    # test_rot_oscillator_with_outputs(-1.0, show_plot=show_plot)
    # test_basic_engine(show_plot=True)
    # test_ideal_wheel(show_plot=show_plot)
    # test_ideal_gear(show_plot=show_plot)
    # test_ideal_planetary(show_plot=show_plot)
    # test_gear(show_plot=show_plot)
    test_friction(show_plot=show_plot, f_sinusoidal=True)
