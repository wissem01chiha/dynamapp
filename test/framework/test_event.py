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

from collimator.framework.event import (
    LeafEventCollection,
    DiagramEventCollection,
)

pytestmark = pytest.mark.minimal


@pytest.fixture
def empty_collection():
    return LeafEventCollection()


@pytest.fixture
def sample_leaf_collection():
    dummy_events = ("event1", "event2")
    return LeafEventCollection(dummy_events)


def test_leaf_collection(empty_collection, sample_leaf_collection):
    assert empty_collection.num_events == 0
    assert len(empty_collection.events) == 0
    assert not empty_collection.has_events

    assert sample_leaf_collection.num_events == 2
    assert len(sample_leaf_collection.events) == 2
    assert sample_leaf_collection.has_events

    # Check that the iterator works
    for i, event in enumerate(sample_leaf_collection):
        assert event == sample_leaf_collection.events[i]


def test_diagram_collection(sample_leaf_collection):
    empty_collection = DiagramEventCollection(subevent_collection={})
    assert empty_collection.num_events == 0
    assert len(empty_collection.events) == 0
    assert not empty_collection.has_events
    assert empty_collection.num_subevents == 0

    collection = DiagramEventCollection(subevent_collection={0: sample_leaf_collection})
    assert collection.num_events == sample_leaf_collection.num_events
    assert collection.events == sample_leaf_collection.events
    assert collection.has_events
    assert collection.num_subevents == 1

    # Check that the iterator works
    for i, event in enumerate(collection):
        assert event == collection.events[i]
