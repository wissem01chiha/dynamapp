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

import collimator

pytestmark = pytest.mark.minimal


class SimpleDiscreteTimeSystem(collimator.LeafSystem):
    def __init__(self, x0=0.0):
        super().__init__()

        self.declare_discrete_state(default_value=x0)  # One state variable.

        self.declare_output_port(self.output, period=1.0, offset=0.0, name="y")
        self.declare_periodic_update(
            self.update,
            period=1.0,
            offset=0.0,
        )

    # x[n+1] = x^3[n]
    def update(self, time, state, *inputs):
        x = state.discrete_state
        return x**3

    # y[n] = x[n]
    def output(self, time, state, *inputs):
        return state.discrete_state


class TestScalarSystem:
    def test_manual_update(self):
        # Instantiate the System
        model = SimpleDiscreteTimeSystem()

        # One state update event, one output update event
        assert model.periodic_events.num_events == 2

        # # Create a context for this system
        context = model.create_context()
        xd0 = 0.9
        context = context.with_discrete_state(xd0)

        # Check this updated the state properly
        xd = context.discrete_state
        assert xd.dtype == float
        assert xd == xd0

        # Manually call the state update
        event = model.state_update_events.mark_all_active().events[0]
        new_state = event.handle(context)
        xd = new_state.discrete_state
        assert xd == xd0**3

        # Manually call the output update.  Usually this would happen in the opposite
        # order during simulation, but here we're just testing the update functionality
        event = model.cache_update_events.mark_all_active().events[0]
        context = context.with_state(new_state)
        new_state = event.handle(context)
        (y,) = new_state.cache
        assert y == xd


if __name__ == "__main__":
    test = TestScalarSystem()
    test.test_manual_update()
