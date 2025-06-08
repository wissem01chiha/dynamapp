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

# acausal imports
from collimator.experimental import AcausalCompiler, AcausalDiagram, EqnEnv
from collimator.experimental import thermal as ht

# collimator imports
import collimator
from collimator import library as lib
import collimator.logging as logging

logging.set_log_level(logging.DEBUG)


def test_basic_thermal(show_plot=False):
    # basic test of self contained system, the output is the state of the system.
    # const_temperature-insulator-thermal_mass.
    # in the simulation, the thermal mass heats up until its temperature matches that of
    # the source.
    temperature = 300
    thermal_mass_temp = 250

    # make acausal diagram
    ev = EqnEnv()
    ad = AcausalDiagram()
    t1 = ht.TemperatureSource(ev, name="t1", temperature=temperature)
    r1 = ht.ThermalInsulator(ev, name="r1", R=1.0)
    c1 = ht.HeatCapacitor(
        ev,
        name="c1",
        C=1.0,
        initial_temperature=thermal_mass_temp,
        initial_temperature_fixed=True,
    )
    ad.connect(t1, "port", r1, "port_a")
    ad.connect(r1, "port_b", c1, "port")
    # compile to acausal system
    ac = AcausalCompiler(ev, ad)
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
    print(params)
    assert params["t1_temperature"] == 300.0
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

    def rc_filter(t, v=1, r=1, c=1, offset=0):
        return v * (1 - np.exp(-t / (r * c))) + offset

    cv_sol = rc_filter(t, v=temperature - thermal_mass_temp, offset=thermal_mass_temp)

    atol = 0.0002
    rtol = 0

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        for i in range(len(lpf_ctx.state.continuous_state)):
            label = "x" + str(i)
            ax.plot(t, x[:, i], label=label)
        ax.plot(t, cv_sol, label="cv_sol")
        ax.legend()
        ax.grid()
        plt.show()

    assert np.allclose(cv_sol, x[:, 0], atol=atol, rtol=rtol)


def test_basic_thermal_with_IO(show_plot=False):
    # heatFlowSource-mass0-tempSensor0-
    #                               |-r1-mass2-tempSensor2
    #                               |-gr1-mass2-tempSensor2
    #                               |-gr2-mass3-tempSensor3
    thermal_mass_temp = 250
    Gr_const = 0.01

    # make acausal diagram
    ev = EqnEnv()
    ad2 = AcausalDiagram()
    hf1 = ht.HeatflowSource(ev, name="hf", Q_flow=-100.0, enable_port_b=False)
    c0 = ht.HeatCapacitor(
        ev,
        name="c0",
        C=1.0,
        initial_temperature=thermal_mass_temp,
        initial_temperature_fixed=True,
    )
    r1 = ht.ThermalInsulator(ev, name="r1", enable_resistance_port=True)
    gr1 = ht.ThermalRadiation(ev, name="gr1", enable_Gr_port=True)
    gr2 = ht.ThermalRadiation(ev, name="gr2", enable_Gr_port=False, Gr=Gr_const)
    c1 = ht.HeatCapacitor(
        ev,
        name="c1",
        C=1.0,
        initial_temperature=thermal_mass_temp,
        initial_temperature_fixed=True,
    )
    c2 = ht.HeatCapacitor(
        ev,
        name="c2",
        C=1.0,
        initial_temperature=thermal_mass_temp,
        initial_temperature_fixed=True,
    )
    c3 = ht.HeatCapacitor(
        ev,
        name="c3",
        C=1.0,
        initial_temperature=thermal_mass_temp,
        initial_temperature_fixed=True,
    )
    tmpsnsr0 = ht.TemperatureSensor(ev, name="tmpsnsr0", enable_port_b=False)
    tmpsnsr1 = ht.TemperatureSensor(ev, name="tmpsnsr1", enable_port_b=False)
    tmpsnsr2 = ht.TemperatureSensor(ev, name="tmpsnsr2", enable_port_b=False)
    tmpsnsr3 = ht.TemperatureSensor(ev, name="tmpsnsr3", enable_port_b=False)
    ad2.connect(hf1, "port_a", c0, "port")
    ad2.connect(c0, "port", tmpsnsr0, "port_a")
    # variable thermal insulator branch
    ad2.connect(tmpsnsr0, "port_a", r1, "port_a")
    ad2.connect(r1, "port_b", c1, "port")
    ad2.connect(c1, "port", tmpsnsr1, "port_a")
    # constant radiation branch
    ad2.connect(tmpsnsr0, "port_a", gr1, "port_a")
    ad2.connect(gr1, "port_b", c2, "port")
    ad2.connect(c2, "port", tmpsnsr2, "port_a")
    # variable radiation branch
    ad2.connect(tmpsnsr0, "port_a", gr2, "port_a")
    ad2.connect(gr2, "port_b", c3, "port")
    ad2.connect(c3, "port", tmpsnsr3, "port_a")

    # compile to acausal system
    ac2 = AcausalCompiler(ev, ad2)
    hs2 = ac2()

    # make wildcat diagram
    builder = collimator.DiagramBuilder()
    hs2 = builder.add(hs2)
    R_k = builder.add(lib.Constant(1.0))
    Gr_k = builder.add(lib.Constant(Gr_const))
    R_k_idx = hs2.insym_to_portid[r1.get_sym_by_port_name("R")]
    Gr_k_idx = hs2.insym_to_portid[gr1.get_sym_by_port_name("Gr")]
    builder.connect(R_k.output_ports[0], hs2.input_ports[R_k_idx])
    builder.connect(Gr_k.output_ports[0], hs2.input_ports[Gr_k_idx])

    # 'compile' wildcat diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    t0_idx = hs2.outsym_to_portid[tmpsnsr0.get_sym_by_port_name("T_rel")]
    t1_idx = hs2.outsym_to_portid[tmpsnsr1.get_sym_by_port_name("T_rel")]
    t2_idx = hs2.outsym_to_portid[tmpsnsr2.get_sym_by_port_name("T_rel")]
    t3_idx = hs2.outsym_to_portid[tmpsnsr3.get_sym_by_port_name("T_rel")]
    recorded_signals = {
        "t0": hs2.output_ports[t0_idx],
        "t1": hs2.output_ports[t1_idx],
        "t2": hs2.output_ports[t2_idx],
        "t3": hs2.output_ports[t3_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time

    t0 = results.outputs["t0"]
    t1 = results.outputs["t1"]
    t2 = results.outputs["t2"]
    t3 = results.outputs["t3"]

    print(t0[-1])
    print(t1[-1])
    print(t2[-1])
    print(t3[-1])

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(t, t0, label="t0", marker="o")
        ax.plot(t, t1, label="t1")
        ax.plot(t, t2, label="t2")
        ax.plot(t, t3, label="t3")
        ax.legend()
        ax.grid()
        plt.show()

    assert np.abs(t0[-1] - 506.24999023046587) < 1e-4
    assert np.abs(t1[-1] - 481.2500293095652) < 1e-4
    assert np.abs(t2[-1] - 506.2499902299842) < 1e-4
    assert np.abs(t3[-1] - 506.2499902299842) < 1e-4


if __name__ == "__main__":
    show_plot = True
    # test_basic_thermal(show_plot=show_plot)
    test_basic_thermal_with_IO(show_plot=show_plot)
