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

from matplotlib import pyplot as plt
import collimator
from collimator import LeafSystem
from collimator.framework import DependencyTicket
import sympy as sp
from collimator.backend import numpy_api as cnp
import pytest

"""
This is a development sandbox for investigating the issue related to using
sp.Piecewise() or jax.numpy.where as expression in lamdify functions of
AcausalSystems.
see acaual/component_library/sandbox.py:TorqueSwitch for implementation of
analogous block in the acausal framework.
see test_acausal.py:test_torque_switch() for testing that.
"""


class SympyPiecewiseBlock(LeafSystem):
    def __init__(self, name="sympy_block"):
        super().__init__(name=name)
        time = sp.Symbol("t")
        inputs = []
        params = []
        spb_st = sp.Function("spb_st")(time)
        lambda_args = (time, spb_st, *inputs, *params)
        lambda_exprs = [sp.Piecewise((spb_st, spb_st < 5), (-1 / spb_st, spb_st >= 5))]

        sp_rhs = sp.lambdify(
            lambda_args,
            lambda_exprs,
            "jax",
        )

        def _rhs(time, state, *u, **params):
            cstate = state.continuous_state
            param_values = []
            x = cstate
            return cnp.array(sp_rhs(time, x, *u, *param_values))

        self._continuous_state_idx = self.declare_continuous_state(
            default_value=1.0,
            ode=_rhs,
        )

        self._output_port_idx = self.declare_output_port(name="out_0")

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
        )

    def _output(self, _time, state, *_inputs, **params):
        xc = state.continuous_state
        return xc


@pytest.mark.skip(reason="development test")
def test_piecewise_block():
    builder = collimator.DiagramBuilder()
    spb = builder.add(SympyPiecewiseBlock(name="spb"))

    system = builder.build()
    context = system.create_context()

    recorded_signals = {"spb": spb.output_ports[0]}
    options = collimator.SimulatorOptions(max_major_step_length=0.1)
    results = collimator.simulate(
        system,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
        options=options,
    )
    t = results.time
    spb = results.outputs["spb"]

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 3))
    ax1.plot(t, spb, label="spb")
    ax1.legend()
    ax1.grid()
    plt.show()


if __name__ == "__main__":
    test_piecewise_block()
