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

import jax
import jax.numpy as jnp

from collimator.framework.event import (
    IntegerTime,
    Event,
    PeriodicEventData,
    LeafEventCollection,
    DiagramEventCollection,
)

from collimator.simulation.simulator import _next_update_time, _next_sample_time

pytestmark = pytest.mark.minimal


class TestNextUpdateTime:
    def test_next_sample_time(self):
        period = 0.1
        offset = 0.05
        event = Event(
            system_id=0,
            event_data=PeriodicEventData(period=0.1, offset=0.05, active=False),
        )

        period_int = IntegerTime.from_decimal(period)
        offset_int = IntegerTime.from_decimal(offset)

        # Negative time
        t1 = _next_sample_time(-period_int, event.event_data)
        assert t1 == offset_int

        # Zero time
        t2 = _next_sample_time(0, event.event_data)
        assert t2 == offset_int

        # Just after the offset - should get the first sample
        t3 = _next_sample_time(IntegerTime.from_decimal(1.1 * period), event.event_data)
        assert jnp.allclose(t3, offset_int + period_int)  # offset + period

    def test_empty_leaf_collection(self):
        events = LeafEventCollection()
        t0 = 0.0
        t1, active = _next_update_time(events, t0)
        assert t1 == IntegerTime.max_int_time
        assert not active.has_events

    def test_empty_diagram_collection(self):
        leaf_events = LeafEventCollection()
        events = DiagramEventCollection({0: leaf_events})
        t0 = 0.0
        t1, active = _next_update_time(events, t0)
        assert t1 == IntegerTime.max_int_time
        assert not active.has_events

    def test_composite_leaf_collection(self):
        # Dummy "contexts" for testing
        ctx0 = jnp.array(0)
        ctx1 = jnp.array(1)
        ctx2 = jnp.array(2)
        ctx3 = jnp.array(3)

        event1 = Event(
            system_id=0,
            event_data=PeriodicEventData(period=0.1, offset=0.0, active=False),
            callback=lambda context: ctx1,
        )
        event2 = Event(
            system_id=0,
            event_data=PeriodicEventData(period=0.2, offset=0.0, active=False),
            callback=lambda context: ctx2,
        )
        event3 = Event(
            system_id=0,
            event_data=PeriodicEventData(period=0.03, offset=0.0, active=False),
            callback=lambda context: ctx3,
        )

        events = LeafEventCollection([event1, event2, event3])

        # What gets returned is a new event collection with ActiveEventData in
        # place of PeriodicEventData, so to test we need to extract the
        # `event_data.active` attribute
        def _extract_data(events):
            return jax.tree_util.tree_map(
                lambda x: x.event_data.active,
                events,
                is_leaf=lambda x: isinstance(x, Event),
            )

        t0 = IntegerTime.from_decimal(0.01)
        t1, active = _next_update_time(events, t0)
        active = _extract_data(active)
        assert t1 == IntegerTime.from_decimal(0.03)
        assert active.events == [False, False, True]

        t0 = IntegerTime.from_decimal(0.09)
        t1, active = _next_update_time(events, t0)
        active = _extract_data(active)
        assert t1 == IntegerTime.from_decimal(0.1)
        assert active.events == [True, False, False]

        t0 = IntegerTime.from_decimal(0.1)
        t1, active = _next_update_time(events, t0)
        active = _extract_data(active)
        assert t1 == IntegerTime.from_decimal(0.12)
        assert active.events == [False, False, True]

        t0 = IntegerTime.from_decimal(0.195)
        t1, active = _next_update_time(events, t0)

        # Check that the callbacks are preserved after the mapping
        assert active.events[0].handle(ctx0) == ctx1  # Active
        assert active.events[1].handle(ctx0) == ctx2  # Active
        assert active.events[2].handle(ctx0) == ctx0  # Inactive

        # Multiple events active at the same time
        active = _extract_data(active)
        assert t1 == IntegerTime.from_decimal(0.2)
        assert active.events == [True, True, False]
