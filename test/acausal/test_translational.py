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

import collimator.logging as logging

logging.set_log_level(logging.DEBUG)


@pytest.mark.parametrize("fixed_ic", [0.0, -1.0])
def test_mech_oscillator(fixed_ic, show_plot=False):
    # basic test of self contained system.
    # mass-spring-force_sensor-wall. spring is initially strecthed.
    # force sensors measures force of spring, which is same as force everywhere else.
    # speed sensor measures speed of mass.

    # make acausal diagram
    ev = EqnEnv()
    ad = AcausalDiagram()
    m1 = trans.Mass(
        ev,
        name="m1",
        M=1.0,
        initial_position=1.0,
        initial_position_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    sp1 = trans.Spring(ev, name="sp1", K=1.0)
    r1 = trans.FixedPosition(ev, name="r1", initial_position=fixed_ic)
    spdsnsr = trans.MotionSensor(
        ev,
        name="spdsnsr",
        enable_flange_b=True,
        enable_position_port=True,
        enable_acceleration_port=True,
    )
    frcsnsr = trans.ForceSensor(ev, name="frcsnsr")
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

    # verify acausal diagram params are in the acausal_system context
    params = context[mech_oscillator.system_id].parameters
    assert params["m1_M"] == 1.0
    assert params["sp1_K"] == 1.0

    # run simulation
    acc_idx = mech_oscillator.outsym_to_portid[spdsnsr.get_sym_by_port_name("a_rel")]
    spd_idx = mech_oscillator.outsym_to_portid[spdsnsr.get_sym_by_port_name("v_rel")]
    pos_idx = mech_oscillator.outsym_to_portid[spdsnsr.get_sym_by_port_name("x_rel")]
    frc_idx = mech_oscillator.outsym_to_portid[frcsnsr.get_sym_by_port_name("f")]
    recorded_signals = {
        "acc": mech_oscillator.output_ports[acc_idx],
        "spd": mech_oscillator.output_ports[spd_idx],
        "pos": mech_oscillator.output_ports[pos_idx],
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
    acc = results.outputs["acc"]
    spd = results.outputs["spd"]
    pos = results.outputs["pos"]
    frc = results.outputs["frc"]

    if fixed_ic == 0.0:
        # this is the equation of motion for ths system
        frc_sol = np.cos(t)
        spd_sol = np.sin(t) * -1
        pos_sol = np.cos(t)
        acc_sol = np.cos(t) * -1
    elif fixed_ic == -1.0:
        frc_sol = np.cos(t) * 2.0
        spd_sol = np.sin(t) * -2.0
        pos_sol = np.cos(t) * 2.0
        acc_sol = np.cos(t) * -2.0
    else:
        raise ValueError(
            f"test test_mech_oscillator does not have pass conditions for {fixed_ic=}."
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

        ax5.plot(t, pos_sol, label="pos_sol")
        ax5.plot(t, pos, label="pos")
        ax5.grid()
        ax5.legend()
        ax6.plot(t, pos_sol - pos)
        ax6.grid()

        ax7.plot(t, acc_sol, label="acc_sol")
        ax7.plot(t, acc, label="acc")
        ax7.grid()
        ax7.legend()
        ax8.plot(t, acc_sol - acc)
        ax8.grid()

        plt.show()

    atol = 4e-5
    rtol = 0.0
    assert np.allclose(frc_sol, frc, atol=atol, rtol=rtol)
    assert np.allclose(spd_sol, spd, atol=atol, rtol=rtol)
    assert np.allclose(pos_sol, pos, atol=atol, rtol=rtol)
    assert np.allclose(acc_sol, acc, atol=atol, rtol=rtol)


def test_damped_mech_oscillators_with_Finput(show_plot=False):
    # mass-spring+damper-force_sensor-wall.
    # inertia-spring+damper-torque_sensor-wall.
    # most comments are related to force, but equally apply to torque.
    # force sensors measures force of spring+damper, which is same as force everywhere else.
    # speed sensor measures speed of mass/interia.
    D = 1.0
    K = 1.0
    M = 1.0

    # make translational oscillator
    ev = EqnEnv()
    ad = AcausalDiagram()
    m1 = trans.Mass(ev, name="m1", M=M)
    f1 = trans.ForceSource(ev, name="f1", enable_force_port=True, enable_flange_b=True)
    sp1 = trans.Spring(
        ev,
        name="sp1",
        K=K,
        initial_position_A=0.0,
        initial_position_A_fixed=True,
        initial_velocity_A=0.0,
        initial_velocity_A_fixed=True,
        initial_position_B=0.0,
        initial_position_B_fixed=True,
        initial_velocity_B=0.0,
        initial_velocity_B_fixed=True,
    )
    d1 = trans.Damper(ev, name="d1", D=D)
    r1 = trans.FixedPosition(ev, name="r1")
    spdsnsr = trans.MotionSensor(ev, name="spdsnsr", enable_flange_b=True)
    frcsnsr = trans.ForceSensor(ev, name="frcsnsr")
    ad.connect(m1, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", frcsnsr, "flange_a")
    ad.connect(frcsnsr, "flange_b", r1, "flange")
    ad.connect(m1, "flange", spdsnsr, "flange_a")
    ad.connect(r1, "flange", spdsnsr, "flange_b")
    ad.connect(sp1, "flange_a", d1, "flange_a")
    ad.connect(sp1, "flange_b", d1, "flange_b")
    ad.connect(m1, "flange", f1, "flange_a")
    ad.connect(r1, "flange", f1, "flange_b")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    trans_oscillator = ac(name="trans_oscillator")

    # make rotataional oscillator
    ev = EqnEnv()
    adr = AcausalDiagram()
    i1 = rot.Inertia(ev, name="i1", I=M)
    t1 = rot.TorqueSource(ev, name="t1", enable_torque_port=True, enable_flange_b=True)
    sp1r = rot.Spring(
        ev,
        name="sp1r",
        K=K,
        initial_angle_A=0.0,
        initial_velocity_A=0.0,
        initial_angle_B=0.0,
        initial_velocity_B=0.0,
    )
    d1r = rot.Damper(ev, name="d1r", D=D)
    r1r = rot.FixedAngle(ev, name="r1r")
    spdsnsrr = rot.MotionSensor(ev, name="spdsnsrr", enable_flange_b=True)
    trqsnsr = rot.TorqueSensor(ev, name="trqsnsr")
    adr.connect(i1, "flange", sp1r, "flange_a")
    adr.connect(sp1r, "flange_b", trqsnsr, "flange_a")
    adr.connect(trqsnsr, "flange_b", r1r, "flange")
    adr.connect(i1, "flange", spdsnsrr, "flange_a")
    adr.connect(r1r, "flange", spdsnsrr, "flange_b")
    adr.connect(sp1r, "flange_a", d1r, "flange_a")
    adr.connect(sp1r, "flange_b", d1r, "flange_b")
    adr.connect(i1, "flange", t1, "flange_a")
    adr.connect(r1r, "flange", t1, "flange_b")

    # compile to acausal system
    acr = AcausalCompiler(ev, adr, verbose=True)
    rot_oscillator = acr(name="rot_oscillator")

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    trans_oscillator = builder.add(trans_oscillator)
    rot_oscillator = builder.add(rot_oscillator)
    sw = builder.add(
        lib.Sine(name="sw", amplitude=0.5, frequency=10, bias=1, phase=np.pi / 2)
    )
    # derivation of StateSpace equivalent of the acausal system
    # x = [v,p]
    # xdot = [vdot,pdot]
    # u = [f(t)]
    # vdot = f(t)/M - D/M*v - K/M*p = [D/M,K/M]*x + [1]*u
    # pdot = v = [1,0]*x + [0]*u
    # fout = K*p + D*v = [D,K]*x + [0]*u
    # spd = v = [1,0]*x + [0]*u
    A = np.array([[-D / M, -K / M], [1, 0]])
    B = np.array([1, 0])
    C = np.array([[D, K], [1, 0]])
    D = np.array([0, 0])

    lti = builder.add(lib.LTISystem(A=A, B=B, C=C, D=D))
    builder.connect(sw.output_ports[0], trans_oscillator.input_ports[0])
    builder.connect(sw.output_ports[0], rot_oscillator.input_ports[0])
    builder.connect(sw.output_ports[0], lti.input_ports[0])

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run simulation
    spd_idx = trans_oscillator.outsym_to_portid[spdsnsr.get_sym_by_port_name("v_rel")]
    frc_idx = trans_oscillator.outsym_to_portid[frcsnsr.get_sym_by_port_name("f")]
    spdr_idx = rot_oscillator.outsym_to_portid[spdsnsrr.get_sym_by_port_name("w_rel")]
    trq_idx = rot_oscillator.outsym_to_portid[trqsnsr.get_sym_by_port_name("tau")]
    recorded_signals = {
        "spd": trans_oscillator.output_ports[spd_idx],
        "frc": trans_oscillator.output_ports[frc_idx],
        "spdr": rot_oscillator.output_ports[spdr_idx],
        "trq": rot_oscillator.output_ports[trq_idx],
        "sw": sw.output_ports[0],
        "lti": lti.output_ports[0],
    }
    t0, tf = 0.0, 10.0
    results = collimator.simulate(
        diagram,
        context,
        (t0, tf),
        recorded_signals=recorded_signals,
    )
    t = results.time
    spd = results.outputs["spd"]
    frc = results.outputs["frc"]
    spdr = results.outputs["spdr"]
    trq = results.outputs["trq"]
    sw = results.outputs["sw"]
    lti_frc = results.outputs["lti"][:, 0]
    lti_spd = results.outputs["lti"][:, 1]

    if show_plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))
        ax1.plot(t, sw, label="sw")
        ax1.plot(t, frc, label="frc")
        ax1.plot(t, trq, label="trq")
        ax1.plot(t, lti_frc, label="lti_frc")
        ax1.grid()
        ax1.legend()
        ax2.plot(t, lti_frc - frc, label="force error")
        ax2.plot(t, lti_frc - trq, label="torque error")
        ax2.grid()
        ax2.legend()

        ax3.plot(t, spd, label="spd")
        ax3.plot(t, spdr, label="spdr")
        ax3.plot(t, lti_spd, label="lti_spd")
        ax3.grid()
        ax3.legend()
        ax4.plot(t, lti_spd - spd, label="speed error")
        ax4.plot(t, lti_spd - spdr, label="speedr error")
        ax4.grid()
        ax4.legend()

        plt.show()

    assert np.allclose(lti_frc, frc)
    assert np.allclose(lti_spd, spd)


def test_many_springs_and_masses(show_plot=False):
    # self contained system, lots of springs and masses.
    # just to see if many components can break the simulator.

    # make acausal diagram
    ev = EqnEnv()
    # ad = AcausalDiagram()
    # m1 = trans.Mass(ev, name="m1", M=1.0, initial_position=1.0, initial_velocity=0.0)
    # sp1 = trans.Spring(ev, name="sp1", K=1.0)
    # r1 = trans.FixedPosition(ev, name="r1")
    # ad.connect(sp1, "flange_a", m1, "flange")
    # ad.connect(sp1, "flange_b", r1, "flange")

    mdl = AcausalDiagram()
    # instantiate components
    m1 = trans.Mass(
        ev,
        name="m1",
        initial_position=1.0,
        initial_position_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    m2 = trans.Mass(
        ev,
        name="m2",
        initial_position=-1.0,
        initial_position_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    m3 = trans.Mass(
        ev,
        name="m3",
        initial_position=0.0,
        initial_position_fixed=True,
        initial_velocity=1.0,
        initial_velocity_fixed=True,
    )
    m4 = trans.Mass(ev, name="m4")
    m5 = trans.Mass(ev, name="m5")
    f1 = trans.ForceSource(ev, name="f1", f=1.0, enable_flange_b=False)
    f2 = trans.ForceSource(ev, name="f2", f=-2.0, enable_flange_b=False)
    sp1 = trans.Spring(ev, name="sp1")
    sp2 = trans.Spring(ev, name="sp2")
    sp3 = trans.Spring(ev, name="sp3")

    # connect components. this adds components to the system as well.
    # node 0
    mdl.connect(m1, "flange", f1, "flange_a")
    mdl.connect(sp1, "flange_a", m1, "flange")

    # node 2
    mdl.connect(m2, "flange", f2, "flange_a")
    mdl.connect(sp1, "flange_b", m2, "flange")
    mdl.connect(sp2, "flange_a", sp1, "flange_b")

    # node 3
    mdl.connect(sp2, "flange_b", m3, "flange")
    mdl.connect(sp2, "flange_b", m4, "flange")
    mdl.connect(m5, "flange", m4, "flange")
    mdl.connect(sp3, "flange_a", m3, "flange")
    mdl.connect(sp3, "flange_b", m5, "flange")

    # compile to acausal system
    ac = AcausalCompiler(ev, mdl)
    mech_oscillator = ac()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    mech_oscillator = builder.add(mech_oscillator)

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    mech_oscillator_ctx = context[mech_oscillator.system_id]

    # run simulation
    recorded_signals = {"x": mech_oscillator.output_ports[0]}
    t0, tf = 0.0, 10.0
    results = collimator.simulate(
        mech_oscillator,
        context,
        (t0, tf),
        recorded_signals=recorded_signals,
    )
    t = results.time
    x = results.outputs["x"]  # x0=pos, x1=vel, x2=x3=algebraic

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        for i in range(len(mech_oscillator_ctx.state.continuous_state)):
            label = "x" + str(i)
            ax.plot(t, x[:, i], label=label)
        ax.legend()
        plt.show()

    # TODO: implement pass conditions.


def test_friction(show_plot=False, f_sinusoidal=False):
    # make two systems, one without viscous friction, and one with.
    # can be confirgured to apply sinusoidal force, but no pass conditions for this,
    # only for manual inspection of results
    def make_sys(name, C=None, f_sinusoidal=False):
        # force-mass-friction-force_sensor-ref
        ev = EqnEnv()
        ad = AcausalDiagram(name=name)
        if f_sinusoidal:
            f1 = trans.ForceSource(
                ev, name="f1", enable_flange_b=False, enable_force_port=True
            )
        else:
            f1 = trans.ForceSource(ev, name="f1", f=10.0, enable_flange_b=False)
        m1 = trans.Mass(
            ev,
            name="m1",
            M=1.0,
            initial_position=0.0,
            initial_position_fixed=True,
            initial_velocity=0.0,
            initial_velocity_fixed=True,
        )
        friction1 = trans.Friction(ev, name="friction1", C=C)
        r1 = trans.FixedPosition(ev, name="r1", initial_position=0.0)
        spdsnsr = trans.MotionSensor(
            ev,
            name="spdsnsr",
            enable_flange_b=False,
            enable_position_port=True,
            enable_acceleration_port=True,
        )
        frcsnsr = trans.ForceSensor(ev, name="frcsnsr")
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
    acc_idx = coul_sys.outsym_to_portid[css.get_sym_by_port_name("a_rel")]
    spd_idx = coul_sys.outsym_to_portid[css.get_sym_by_port_name("v_rel")]
    pos_idx = coul_sys.outsym_to_portid[css.get_sym_by_port_name("x_rel")]
    frc_idx = coul_sys.outsym_to_portid[cfs.get_sym_by_port_name("f")]
    acc2_idx = visc_sys.outsym_to_portid[vss.get_sym_by_port_name("a_rel")]
    spd2_idx = visc_sys.outsym_to_portid[vss.get_sym_by_port_name("v_rel")]
    pos2_idx = visc_sys.outsym_to_portid[vss.get_sym_by_port_name("x_rel")]
    frc2_idx = visc_sys.outsym_to_portid[vfs.get_sym_by_port_name("f")]
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
    # test_mech_oscillator(0.0, show_plot=show_plot)
    # test_mech_oscillator(-1.0, show_plot=show_plot)
    # test_mech_oscillator_with_outputs(show_plot=show_plot)
    # test_damped_mech_oscillators_with_Finput(show_plot=show_plot)
    # test_many_springs_and_masses(show_plot=show_plot)
    test_friction(show_plot=show_plot, f_sinusoidal=True)

    # @am. ill come back to this later.
    # commented out to please ruff
    # @pytest.mark.xfail(reason="num eqns and num vars mismatch")
    # def test_damped_mech_oscillators_with_Sinput(show_plot=False):
    #     # FIXME: this test fails because we do not correctly handle the case
    #     # where one of the system states thta is normally integrated, is instead
    #     # driven by an input signal.

    #     # mass-spring+damper-force_sensor-wall.
    #     # inertia-spring+damper-torque_sensor-wall.
    #     # force sensors measures force of spring+damper, which is same as force everywhere else.
    #     # speed sensor measures speed of mass.
    #     D = 1.0
    #     K = 1.0
    #     M = 1.0

    #     # make translational oscillator
    #     ev = EqnEnv()
    #     ad = AcausalDiagram()
    #     m1 = trans.Mass(ev, name="m1", M=M)
    #     ss1 = trans.SpeedSource(
    #         ev, name="ss1", enable_speed_port=True, enable_flange_b=True
    #     )
    #     sp1 = trans.Spring(
    #         ev,
    #         name="sp1",
    #         K=K,
    #         initial_position_A=0.0,
    #         initial_velocity_A=0.0,
    #         initial_position_B=0.0,
    #         initial_velocity_B=0.0,
    #     )
    #     d1 = trans.Damper(ev, name="d1", D=D)
    #     r1 = trans.FixedPosition(ev, name="r1")
    #     spdsnsr = trans.MotionSensor(ev, name="spdsnsr", enable_flange_b=True)
    #     frcsnsr = trans.ForceSensor(ev, name="frcsnsr")
    #     ad.connect(m1, "flange", sp1, "flange_a")
    #     ad.connect(sp1, "flange_b", frcsnsr, "flange_b")
    #     ad.connect(frcsnsr, "flange_a", r1, "flange")
    #     ad.connect(m1, "flange", spdsnsr, "flange_b")
    #     ad.connect(r1, "flange", spdsnsr, "flange_a")
    #     ad.connect(sp1, "flange_a", d1, "flange_a")
    #     ad.connect(sp1, "flange_b", d1, "flange_b")
    #     ad.connect(m1, "flange", ss1, "flange_a")
    #     ad.connect(r1, "flange", ss1, "flange_b")

    #     # compile to acausal system
    #     ac = AcausalCompiler(ev, ad, verbose=True)
    #     trans_oscillator = ac(name="trans_oscillator")

    #     # make wildcat diagram
    #     builder = collimator.DiagramBuilder()
    #     trans_oscillator = builder.add(trans_oscillator)
    #     sw = builder.add(
    #         lib.Sine(name="sw", amplitude=0.5, frequency=10, bias=1, phase=np.pi / 2)
    #     )
    #     # derivation of StateSpace equivalent of the acausal system
    #     # x = [p]
    #     # xdot = [pdot]
    #     # u = [v(t)]
    #     # pdot = v = [0]*x + [1]*u
    #     # fout = K*p + D*v = [K]*x + [D]*u
    #     # spd = v = [0]*x + [1]*u
    #     A = np.array([0])
    #     B = np.array([1])
    #     C = np.array([K])
    #     D = np.array([1])

    #     lti = builder.add(lib.LTISystem(A=A, B=B, C=C, D=D))
    #     builder.connect(sw.output_ports[0], trans_oscillator.input_ports[0])
    #     builder.connect(sw.output_ports[0], lti.input_ports[0])

    #     # 'compile' wildcat diagram
    #     diagram = builder.build()
    #     context = diagram.create_context(check_types=True)

    #     # run simulation
    #     spd_idx = trans_oscillator.outsym_to_portid[spdsnsr.get_sym_by_port_name("v_rel")]
    #     frc_idx = trans_oscillator.outsym_to_portid[frcsnsr.get_sym_by_port_name("f")]
    #     recorded_signals = {
    #         "spd": trans_oscillator.output_ports[spd_idx],
    #         "frc": trans_oscillator.output_ports[frc_idx],
    #         "sw": sw.output_ports[0],
    #         "lti": lti.output_ports[0],
    #     }
    #     t0, tf = 0.0, 10.0
    #     results = collimator.simulate(
    #         diagram,
    #         context,
    #         (t0, tf),
    #         recorded_signals=recorded_signals,
    #     )
    #     t = results.time
    #     spd = results.outputs["spd"]
    #     frc = results.outputs["frc"]
    #     sw = results.outputs["sw"]
    #     lti_frc = results.outputs["lti"][:, 0]
    #     lti_spd = results.outputs["lti"][:, 1]

    #     if show_plot:
    #         fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))
    #         ax1.plot(t, sw, label="sw")
    #         ax1.plot(t, frc, label="frc")
    #         ax1.plot(t, lti_frc, label="lti_frc")
    #         ax1.grid()
    #         ax1.legend()
    #         ax2.plot(t, lti_frc - frc, label="force error")
    #         ax2.grid()
    #         ax2.legend()

    #         ax3.plot(t, spd, label="spd")
    #         ax3.plot(t, lti_spd, label="lti_spd")
    #         ax3.grid()
    #         ax3.legend()
    #         ax4.plot(t, lti_spd - spd, label="speed error")
    #         ax4.grid()
    #         ax4.legend()

    #         plt.show()

    #     assert np.allclose(lti_frc, frc)
    #     assert np.allclose(lti_spd, spd)

    # @pytest.mark.xfail(reason="num eqns and num vars mismatch")
    # def test_damped_mech_oscillators_with_Sinput2(show_plot=False):
    #     # FIXME: this test fails because we do not correctly handle the case
    #     # where the reference point has a none constant constraint.

    #     # mass-spring+damper-force_sensor-speed_source.
    #     # inertia-spring+damper-torque_sensor-wall.
    #     # force sensors measures force of spring+damper, which is same as force everywhere else.
    #     # speed sensor measures speed of mass.
    #     D = 1.0
    #     K = 1.0
    #     M = 1.0

    #     # make translational oscillator
    #     ev = EqnEnv()
    #     ad = AcausalDiagram()
    #     m1 = trans.Mass(ev, name="m1", M=M)
    #     ss1 = trans.SpeedSource(ev, name="ss1", enable_speed_port=True)
    #     sp1 = trans.Spring(
    #         ev,
    #         name="sp1",
    #         K=K,
    #         initial_position_A=0.0,
    #         initial_velocity_A=0.0,
    #         initial_position_B=0.0,
    #         initial_velocity_B=0.0,
    #     )
    #     d1 = trans.Damper(ev, name="d1", D=D)
    #     spdsnsr = trans.MotionSensor(ev, name="spdsnsr", enable_flange_b=True)
    #     frcsnsr = trans.ForceSensor(ev, name="frcsnsr")
    #     ad.connect(m1, "flange", sp1, "flange_a")
    #     ad.connect(sp1, "flange_b", frcsnsr, "flange_b")
    #     ad.connect(frcsnsr, "flange_a", ss1, "flange")
    #     ad.connect(m1, "flange", spdsnsr, "flange_b")
    #     ad.connect(ss1, "flange", spdsnsr, "flange_a")
    #     ad.connect(sp1, "flange_a", d1, "flange_a")
    #     ad.connect(sp1, "flange_b", d1, "flange_b")

    #     # compile to acausal system
    #     ac = AcausalCompiler(ev, ad, verbose=True)
    #     trans_oscillator = ac(name="trans_oscillator")

    #     # make wildcat diagram
    #     builder = collimator.DiagramBuilder()
    #     trans_oscillator = builder.add(trans_oscillator)
    #     sw = builder.add(
    #         lib.Sine(name="sw", amplitude=0.5, frequency=10, bias=1, phase=np.pi / 2)
    #     )
    #     builder.connect(sw.output_ports[0], trans_oscillator.input_ports[0])

    #     # 'compile' wildcat diagram
    #     diagram = builder.build()
    #     context = diagram.create_context(check_types=True)

    #     # run simulation
    #     spd_idx = trans_oscillator.outsym_to_portid[spdsnsr.get_sym_by_port_name("v_rel")]
    #     frc_idx = trans_oscillator.outsym_to_portid[frcsnsr.get_sym_by_port_name("f")]
    #     recorded_signals = {
    #         "spd": trans_oscillator.output_ports[spd_idx],
    #         "frc": trans_oscillator.output_ports[frc_idx],
    #         "sw": sw.output_ports[0],
    #     }
    #     t0, tf = 0.0, 10.0
    #     results = collimator.simulate(
    #         diagram,
    #         context,
    #         (t0, tf),
    #         recorded_signals=recorded_signals,
    #     )
    #     t = results.time
    #     spd = results.outputs["spd"]
    #     frc = results.outputs["frc"]
    #     sw = results.outputs["sw"]

    #     if show_plot:
    #         fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 8))
    #         ax1.plot(t, sw, label="sw")
    #         ax1.plot(t, frc, label="frc")
    #         ax1.grid()
    #         ax1.legend()

    #         ax3.plot(t, spd, label="spd")
    #         ax3.grid()
    #         ax3.legend()

    #         plt.show()
