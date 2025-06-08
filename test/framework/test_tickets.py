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

from collimator.framework import DependencyTicket, next_dependency_ticket


def test_empty():
    ticket = DependencyTicket.nothing
    assert ticket == 0


def test_increments():
    ticket1 = next_dependency_ticket()
    assert ticket1 == DependencyTicket._next_available

    ticket2 = next_dependency_ticket()
    assert ticket2 == ticket1 + 1

    # Test inequality
    assert ticket1 != ticket2
    assert ticket1 < ticket2
    assert ticket2 > ticket1
