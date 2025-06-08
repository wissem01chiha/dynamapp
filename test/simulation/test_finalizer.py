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

from collimator.logging import logger


class BlockWithFinalizer(collimator.LeafSystem):
    def __init__(self, ref: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("BlockWithFinalizer.__init__")
        self._ref = ref

    def post_simulation_finalize(self) -> None:
        logger.debug("BlockWithFinalizer.post_simulation_finalize")
        self._ref["finalized"] = True
        return super().post_simulation_finalize()


def build_diagram_with_finalizer(name):
    builder = collimator.DiagramBuilder()
    ref = {"finalized": False}
    builder.add(BlockWithFinalizer(ref))
    return builder.build(name), ref


def build_diagram_with_group_and_finalizer():
    builder = collimator.DiagramBuilder()

    grp1, ref1 = build_diagram_with_finalizer("ref1_grp")
    grp2, ref2 = build_diagram_with_finalizer("ref2_grp")
    builder.add(grp1)
    builder.add(grp2)

    ref3 = {"finalized": False}
    builder.add(BlockWithFinalizer(ref3))

    return builder.build("root"), ref1, ref2, ref3


@pytest.mark.minimal
@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_finalizer(backend: str):
    collimator.set_backend(backend)

    diagram, ref1, ref2, ref3 = build_diagram_with_group_and_finalizer()
    assert not ref1["finalized"]
    assert not ref2["finalized"]
    assert not ref3["finalized"]

    context = diagram.create_context()
    collimator.simulate(diagram, context, (0.0, 10.0))

    assert ref1["finalized"], "Finalizer 1 was not called"
    assert ref2["finalized"], "Finalizer 2 was not called"
    assert ref3["finalized"], "Finalizer 3 was not called"
