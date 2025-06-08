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

"""Test for collimator.framework.parameter."""

import pytest

import jax.numpy as jnp
from jax.tree_util import register_pytree_node
import numpy as np

from collimator.framework import Parameter
from collimator.framework.error import ParameterError


pytest.mark.minimal

one = Parameter(name="one", value=1)
two = Parameter(name="two", value=2)
unnamed_param = Parameter(value=42)

# expression string -> (actual expression, (expected value, expected string))
expr = {
    "1": (1, (1, "1")),
    "np.array([1])": (np.array([1]), (np.array([1]), "np.array([1])")),
    "np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])": (
        np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        (
            np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
            "np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])",
        ),
    ),
    "np.array([[1, 2], [3, 4]])": (
        np.array([[1, 2], [3, 4]]),
        (
            np.array([[1, 2], [3, 4]]),
            "np.array([[1, 2], [3, 4]])",
        ),
    ),
    "np.array([1.0, 2.0, 3.0], dtype=np.float32)": (
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "np.array([1.0, 2.0, 3.0], dtype=np.float32)",
        ),
    ),
    "True": (True, (True, "True")),
    "(1,2)": ((1, 2), ((1, 2), "(1, 2)")),
    "one": (one, (1, "one")),
    "1 + one": (1 + one, (2, "1 + one")),
    "one + 1": (one + 1, (2, "one + 1")),
    "1 - one": (1 - one, (0, "1 - one")),
    "one - 1": (one - 1, (0, "one - 1")),
    "one * 2": (one * 2, (2, "one * 2")),
    "2 * one": (2 * one, (2, "2 * one")),
    "2 / one": (2 / one, (2, "2 / one")),
    "one / 2": (one / 2, (0.5, "one / 2")),
    "one ** 2": (one**2, (1, "one ** 2")),
    "2 ** one": (2**one, (2, "2 ** one")),
    "-one": (-one, (-1, "-one")),
    "+one": (+one, (1, "+one")),
    "abs(-one)": (abs(-one), (1, "abs(-one)")),
    "one / two": (one / two, (0.5, "one / two")),
    "1 + (one * (-two / one))": (
        1 + (one * (-two / one)),
        (-1, "1 + (one * ((-two) / one))"),
    ),
    "one < two": (one < two, (True, "one < two")),
    "one <= two": (one <= two, (True, "one <= two")),
    "one == two": (one == two, (False, "one == two")),
    "one != two": (one != two, (True, "one != two")),
    "one >= two": (one >= two, (False, "one >= two")),
    "one > two": (one > two, (False, "one > two")),
    "one < one": (one < one, (False, "one < one")),
    "one <= one": (one <= one, (True, "one <= one")),
    "one == one": (one == one, (True, "one == one")),
    "one != one": (one != one, (False, "one != one")),
    "one >= one": (one >= one, (True, "one >= one")),
    "one > one": (one > one, (False, "one > one")),
    "one < 1": (one < 1, (False, "one < 1")),
    "one <= 1": (one <= 1, (True, "one <= 1")),
    "one == 1": (one == 1, (True, "one == 1")),
    "one != 1": (one != 1, (False, "one != 1")),
    "one >= 1": (one >= 1, (True, "one >= 1")),
    "one > 1": (one > 1, (False, "one > 1")),
    "unnamed_param": (unnamed_param, (42, "42")),
    # FIXME: the string representation is logically correct but inverted
    # implementing __rlt__, __rle__, __req__, etc. does not fix the issue
    "1 < one": (1 < one, (False, "one > 1")),
    "1 <= one": (1 <= one, (True, "one >= 1")),
    "1 == one": (1 == one, (True, "one == 1")),
    "1 != one": (1 != one, (False, "one != 1")),
    "1 >= one": (1 >= one, (True, "one <= 1")),
    "1 > one": (1 > one, (False, "one < 1")),
    "np.array([one])": (np.array([one]), (np.array([1]), "np.array([one])")),
    "np.float32(2)": (np.float32(2), (np.float32(2), "np.float32(2.0)")),
    # FIXME: str representation should be "np.float32(one)" - currently
    # parameters passed to numpy functions are not optimizable because they
    # are resolved to their values when evaluated in model_interface.py.
    # "np.float32(one)": (np.float32(one), (1.0, "np.float32(one)")),
    "[one, two]": ([one, two], ([1, 2], "[one, two]")),
    "[[one], [two]]": ([[one], [two]], ([[1], [2]], "[[one], [two]]")),
    "(one, two)": ((one, two), ((1, 2), "(one, two)")),
    "((one, two), (one, two))": (
        ((one, two), (one, two)),
        (((1, 2), (1, 2)), "((one, two), (one, two))"),
    ),
}


@pytest.mark.parametrize(
    "param_value, expected",
    expr.values(),
    ids=expr.keys(),
)
def test_parameter_compute(param_value, expected):
    expected_value, expected_string = expected
    p = Parameter(value=param_value)
    assert not isinstance(p.get(), Parameter)
    # assert p.get() == expected_value
    np.testing.assert_equal(p.get(), expected_value)
    assert str(p) == expected_string


def test_parameter_cache_invalidation():
    p1 = Parameter(value=1)
    p2 = Parameter(value=2)
    new_param = p1 + p2

    assert new_param.get() == 3

    p1.set(2 * p2)
    assert new_param.get() == 6

    p1.set(11)
    assert new_param.get() == 13

    # FIXME: again, passing a parameter to a numpy function breaks the
    # traceability of the parameter
    # new_param = Parameter(value=np.array(p2))
    # p2.set(123)
    # assert new_param.get() == 123

    new_param = Parameter(value=[[p2]])
    assert new_param.get() == [[2]]
    p2.set(123)
    assert new_param.get() == [[123]]


def test_raise_on_invalid_value():
    with pytest.raises(ParameterError) as exc_info:
        Parameter(value=one + "1").get()

    assert "Invalid value in parameter list" in str(exc_info.value)


def test_pytree_parameter():
    class CustomPytree:
        def __init__(self, a):
            self.a = jnp.asarray(a)

    register_pytree_node(
        CustomPytree,
        lambda tree: ((tree.a,), None),
        lambda _, args: CustomPytree(*args),
    )

    ptree = Parameter(value=CustomPytree([1, 2, 3]))

    np.testing.assert_array_equal(ptree.get().a, jnp.array([1, 2, 3]))
    assert str(ptree) == "np.array([1, 2, 3])"


def test_array_parameter():
    # There was a bug where Parameter.get() and Parameter.__str__() with array value
    # would blow up by converting every single element into its own object, like
    # array([[1,2],[3,4]]) became array([array([1,2]), array([3,4])]). With large
    # enough arrays, everything would blow up.

    def _validate_str(p, locals_=None):
        # Validate that the stringified version is a valid Python expression,
        # unlike np.ndarray's native str/repr.
        evaluated = eval(str(p), {"np": np, "jnp": jnp}, locals_)
        np.testing.assert_array_equal(evaluated, p.get())

    # Large matrix. Check serialization and immutability in read-only actions.
    # Unfortunately this test does not validate the fix, because numpy is smart
    # enough to convert array([array([1,2]), array([3,4])]) into array([[1,2],[3,4]]).
    matrix = np.random.rand(100, 100)
    p = Parameter(value=matrix)
    _validate_str(p)
    assert p.value is matrix

    # Same but with jnp
    matrix = jnp.array(matrix)
    p = Parameter(value=matrix)
    _validate_str(p)
    assert p.value is matrix


def test_parameter_from_python_expr():
    # Not entirely sure what is the best form to serialize to. We have 2 options:
    # 1. Serialize back as string expression, like 'jnp.float32(a)'
    # 2. Serialize back as an actual expression, like jnp.float32(a)
    # Here, we test the 2nd approach.

    # See test_batch_simulation.py
    a = Parameter(name="a", value=1.0)
    b = Parameter(
        value="jnp.float32(a)",
        is_python_expr=True,
        py_namespace={"jnp": jnp, "a": a},
    )
    assert str(b) == "jnp.float32(a)"
    assert b.get() == 1.0


def test_parameter_string_values():
    b = Parameter(value="sup' yo!")
    assert b.get() == "sup' yo!"
    assert str(b) == "sup' yo!"  # Subject to change. See comments in __str__()
    value_expr, is_string = b.value_as_api_param(allow_string_literal=True)
    assert value_expr == "sup' yo!"
    assert is_string
    value_expr, is_string = b.value_as_api_param(allow_string_literal=False)
    assert value_expr == repr("sup' yo!")
    assert not is_string
    assert eval(value_expr) == "sup' yo!"

    c = Parameter(name="c", value="sup' yo!")
    value_expr, is_string = c.value_as_api_param()
    assert value_expr == "c"
    assert not is_string
    value_expr, is_string = c.value_as_api_param(allow_param_name=False)
    assert value_expr == "sup' yo!"
    assert is_string

    a = np.array(["a", "bb"])
    b = Parameter(a)
    np.testing.assert_array_equal(b.get(), a)
    assert str(b) == "np.array(['a', 'bb'])"
    value_expr, is_string = b.value_as_api_param()
    assert value_expr == "np.array(['a', 'bb'])"
    assert not is_string
    assert np.array_equal(eval(value_expr), a)

    a = Parameter(name="a", value=42)
    b = Parameter(name="b", value="a string")
    c = Parameter(name="c", value=0.0)
    d = Parameter(name="d", value=np.array([[a, b], [2, c]]))
    value_expr, is_string = d.value_as_api_param(
        allow_param_name=False, allow_string_literal=False
    )
    assert value_expr == "np.array([[a, b], [2, c]])"


def test_parameter_list_as_array():
    p = Parameter(value=[1, 2], as_array=True)
    assert isinstance(p.get(), jnp.ndarray)
    np.testing.assert_array_equal(p.get(), np.array([1, 2]))
