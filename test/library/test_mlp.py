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

import jax
import jax.numpy as jnp
import os
import pathlib
import shutil
import equinox as eqx
import collimator
from collimator.library import Constant, MLP
import pytest


@pytest.mark.parametrize(
    "seed, in_size, out_size, width_size, depth",
    [
        (40, 4, 2, 4, 1),
        (41, 16, 16, 16, 2),
        (42, 64, 128, 64, 3),
    ],
)
def test_mlp_against_native_equinox(seed, in_size, out_size, width_size, depth):
    collimator.set_backend("jax")
    key = jax.random.PRNGKey(seed)
    mlp_config = {
        "key": key,
        "in_size": in_size,
        "out_size": out_size,
        "width_size": width_size,
        "depth": depth,
        "activation": jax.nn.relu,
    }

    # Generate a random input
    key, subkey = jax.random.split(key)
    input = jax.random.normal(subkey, (in_size,))

    # Equinox model
    model = eqx.nn.MLP(**mlp_config)

    # Collimator model
    col_mlp_config = mlp_config.copy()  # Copy the original equinox config
    col_mlp_config["activation_str"] = col_mlp_config.pop(
        "activation"
    )  # Rename 'activation' to 'activation_str'
    col_mlp_config["activation_str"] = "relu"  # Set activation string

    col_mlp_config["seed"] = col_mlp_config.pop("key")  # Rename 'key' to 'seed'
    col_mlp_config["seed"] = seed  # Set seed

    builder = collimator.DiagramBuilder()
    col_mlp_block = builder.add(MLP(**col_mlp_config, name="mlp"))
    input_block = builder.add(Constant(input, name="input"))
    builder.connect(input_block.output_ports[0], col_mlp_block.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()

    result = col_mlp_block.output_ports[0].eval(ctx)

    # Check if the output of the collimator model
    # is the same as the output of the equinox model
    assert jnp.allclose(result, model(input))


def test_mlp_static():
    collimator.set_backend("jax")
    builder = collimator.DiagramBuilder()
    mlp_config = {
        "in_size": 3,
        "out_size": 2,
        "width_size": 4,
        "depth": 2,
        "seed": 0,
    }
    val = jnp.arange(mlp_config["in_size"])

    constant = builder.add(Constant(val, name="constant"))
    mlp = builder.add(MLP(**mlp_config, name="mlp"))

    builder.connect(constant.output_ports[0], mlp.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    recorded_signals = {"mlp.y": mlp.output_ports[0]}
    results = collimator.simulate(
        diagram, context, (0.0, 1.0), recorded_signals=recorded_signals
    )
    assert jnp.allclose(results.outputs["mlp.y"], diagram["mlp"].mlp(val))


def test_mlp_serialize():
    collimator.set_backend("jax")
    builder = collimator.DiagramBuilder()
    mlp_config = {
        "in_size": 3,
        "out_size": 2,
        "width_size": 4,
        "depth": 2,
    }
    val = jnp.arange(mlp_config["in_size"])

    constant = builder.add(Constant(val, name="constant"))
    mlp = builder.add(MLP(**mlp_config, seed=0, name="mlp"))

    # make a directory for saving the serialized model
    cwd = os.getcwd()
    workdir = os.path.join(cwd, "test/workdir/test_mlp_serialize/")
    workdir_path = pathlib.Path(workdir)
    if workdir_path.exists():
        shutil.rmtree(workdir_path)
    os.makedirs(workdir_path, mode=0o777, exist_ok=True)

    # test whole model serialization
    file_name = os.path.join(workdir, "test_mlp_serialize.eqx")
    mlp.serialize(file_name)
    mlp2 = builder.add(MLP(**mlp_config, file_name=file_name, name="mlp2"))

    builder.connect(constant.output_ports[0], mlp.input_ports[0])
    builder.connect(constant.output_ports[0], mlp2.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    recorded_signals = {"mlp.y": mlp.output_ports[0], "mlp2.y": mlp2.output_ports[0]}
    results = collimator.simulate(
        diagram, context, (0.0, 1.0), recorded_signals=recorded_signals
    )
    assert jnp.allclose(results.outputs["mlp.y"], results.outputs["mlp2.y"])


if __name__ == "__main__":
    # test_mlp_against_native_equinox(40, 4, 2, 4, 1)
    # test_mlp_static()
    test_mlp_serialize()
    # test_pretrained(show_plot=True)
