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
from collimator.framework import (
    CollimatorError,
    StaticError,
    ShapeMismatchError,
    BlockParameterError,
)

this_dir = os.path.dirname(__file__)
output_dir = "test/workdir/errors"


def test_add_different_length_vectors():
    collimator.set_backend("jax")  # For TypeError, otherwise ValueError with numpy
    with pytest.raises(StaticError) as exc:
        collimator.cli.run(
            this_dir, "model_add_different_length_vectors.json", output_dir
        )
    cause = exc.value.__cause__
    assert isinstance(cause, TypeError)
    assert "got incompatible shapes for broadcasting" in cause.args[0]


def test_string_gain():
    with pytest.raises(CollimatorError) as e:
        collimator.cli.run(this_dir, "model_string_gain.json", output_dir)
    assert e.value.caused_by(TypeError)


def test_name_error():
    with pytest.raises(BlockParameterError) as exc:
        collimator.cli.run(this_dir, "model_name_error.json", output_dir)
    cause = exc.value.__cause__
    assert isinstance(cause, NameError)
    assert "name 'undefined_name' is not defined" in cause.args[0]


def test_syntax_error():
    with pytest.raises(BlockParameterError) as exc:
        collimator.cli.run(this_dir, "model_syntax_error.json", output_dir)
    cause = exc.value.__cause__
    assert isinstance(cause, SyntaxError)
    # 3.9: assert "EOL while scanning string literal" in cause.args[0]
    # 3.10: assert "unterminated string literal" in cause.args[0]


def test_integrator_state_wrong_shape():
    with pytest.raises(ShapeMismatchError):
        collimator.cli.run(
            this_dir, "model_integrator_state_wrong_shape.json", output_dir
        )
