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

import os

import pytest

import collimator
import collimator.testing as test
from collimator.library import MLP

pytestmark = pytest.mark.minimal


def test_mlp(request):
    # create workdir
    test_paths = test.get_paths(request)
    workdir = test_paths["workdir"]
    print(f"workdir: {workdir}")

    # create dummy NN to placehold for 'pretrtained model'
    nn_config = {
        "in_size": 2,
        "out_size": 2,
        "width_size": 2,
        "depth": 2,
        "seed": 0,
    }
    pretrained = MLP(**nn_config, name="pretrained")

    # save NN
    pretrained.serialize(f"{workdir}/pretrained.eqx")

    # copy model to workdir
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model.json"))
    os.system(f"ln -sf {model_path} {workdir}/")

    # FIXME collimator.load_model should know to look for files in workdir
    curdir = os.getcwd()
    try:
        os.chdir(workdir)
        # load/simulate collimator model which references the above saved NN
        model = collimator.load_model(".")
        model.simulate()

    finally:
        os.chdir(curdir)
