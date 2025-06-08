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

"""
This module defines a system for managing and manipulating parameters in wildcat.

Key Components:

1. **Parameter Class**: Represents a parameter with a value that can be an expression,
another parameter, or a variety of data types such as arrays, numbers, or strings.
It supports simple operations like addition, subtraction, multiplication, etc.
More complex expressions can be represented as python expressions, which are evaluated.
For example, a parameter value can be `np.eye(p)` where `p` is a parameter.

2. **ParameterCache Class**: Manages the cache and dependencies of `Parameter` objects.
It ensures that when a parameter's value changes, all dependent parameters are
invalidated and recalculated as needed.

SystemBase objects use Parameter objects to store dynamic and static parameters.
In particular, collimator model parameters are represented in the Diagram system
as dynamic parameters.

SystemBase objects do not need to manage Parameter objects beyond their
declaration, as these parameters are resolved before invoking the methods of
the SystemBase object. This is achieved by the `parameters` decorator
in system_base.py, `InitializeParameterResolver` in leaf_system.py and the
context creation logic that resolves all parameters before a simulation run.

Example usage:

```
c = Parameter(value=1.0)
builder = collimator.DiagramBuilder()
constant1 = builder.add(library.Constant(c))
constant2 = builder.add(library.Constant(c + 1))
diagram = builder.build()

context = diagram.create_context()
constant1.output_ports[0].eval(context) # 1.0
constant2.output_ports[0].eval(context) # 2.0

c.set(2.0)
context = diagram.create_context()
constant1.output_ports[0].eval(context) # out = 2.0
constant2.output_ports[0].eval(context) # out = 3.0

c.get() # 2.0
```

"""

import ast
from collections import defaultdict
import dataclasses
import enum
from typing import Union, TYPE_CHECKING

from jax import Array
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

from . import build_recorder
from .error import ParameterError
from ..backend import utils
from ..backend.backend import IS_JAXLITE
from ..backend.typing import ArrayLike, DTypeLike, ShapeLike


if TYPE_CHECKING:
    from ..system.system_base import SystemBase


class ParameterExpr(list):
    pass


class Ops(enum.Enum):
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    FLOORDIV = enum.auto()
    MOD = enum.auto()
    POW = enum.auto()
    NEG = enum.auto()
    POS = enum.auto()
    ABS = enum.auto()
    EQ = enum.auto()
    NE = enum.auto()
    LT = enum.auto()
    LE = enum.auto()
    GT = enum.auto()
    GE = enum.auto()
    MATMUL = enum.auto()


__OPS_FN__ = {
    Ops.ADD: lambda x, y: x + y,
    Ops.SUB: lambda x, y: x - y,
    Ops.MUL: lambda x, y: x * y,
    Ops.DIV: lambda x, y: x / y,
    Ops.FLOORDIV: lambda x, y: x // y,
    Ops.MOD: lambda x, y: x % y,
    Ops.POW: lambda x, y: x**y,
    Ops.NEG: lambda x: -x,
    Ops.POS: lambda x: +x,
    Ops.ABS: abs,
    Ops.EQ: lambda x, y: x == y,
    Ops.NE: lambda x, y: x != y,
    Ops.LT: lambda x, y: x < y,
    Ops.LE: lambda x, y: x <= y,
    Ops.GT: lambda x, y: x > y,
    Ops.GE: lambda x, y: x >= y,
    Ops.MATMUL: lambda x, y: x @ y,
}

__OPS_STR__ = {
    Ops.ADD: "+",
    Ops.SUB: "-",
    Ops.MUL: "*",
    Ops.DIV: "/",
    Ops.FLOORDIV: "//",
    Ops.MOD: "%",
    Ops.POW: "**",
    Ops.NEG: "-",
    Ops.POS: "+",
    Ops.ABS: "abs",
    Ops.EQ: "==",
    Ops.NE: "!=",
    Ops.LT: "<",
    Ops.LE: "<=",
    Ops.GT: ">",
    Ops.GE: ">=",
    Ops.MATMUL: "@",
}


class _VarRecorder(ast.NodeVisitor):
    """Used to record all variables in a Python expression."""

    def __init__(self):
        super().__init__()
        self.vars = set()

    def visit_Name(self, node):
        self.vars.add(node.id)
        return node.id


ArrayLikeTypes = (
    Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_,
    np.number,  # NumPy scalar types
    bool,
    int,
    float,
    complex,  # Python scalar types
)


def resolve_parameters(python_expr: str, env: dict, mode="eval"):
    # look for variables used in the expression
    tree = ast.parse(python_expr, mode=mode)
    var_recorder = _VarRecorder()
    var_recorder.visit(tree.body)

    # record the parameters and resolve them
    parameters = set()
    resolved_params = {}
    for var in var_recorder.vars:
        if var in env:
            if isinstance(env[var], Parameter):
                parameters.add(env[var])
                resolved_params[var] = env[var].get()

    return parameters, resolved_params


def _resolve_array_like(value: ArrayLike) -> ArrayLike:
    if value.ndim == 0:
        return value

    # If jax array, there can not be dependents because all elements
    # are numerical.
    if isinstance(value, Array):
        return value

    # If ndarray, only resolve if dtype is 'object'. Other dtypes
    # can not contain Parameter elements.
    if isinstance(value, np.ndarray) and value.dtype != np.object_:
        return value

    # FIXME: this changes the array from a well-formed ndarray
    # like: array([[1,2], [3,4]]) to an array of arrays
    # like: array([array([1, 2]), array([3, 4])])
    # The latter form becomes extremely inefficient in JAX.
    # In the vast majority of cases though, we have numerical arrays
    # that have already been resolved to their final values.
    vals = []
    for val in value:
        if isinstance(val, Parameter):
            vals.append(ParameterCache.__compute__(val))
        else:
            vals.append(_resolve_array_like(val))

    numeric_types = (int, float, complex, np.number, bool)
    is_numeric = all(isinstance(v, numeric_types) for v in vals)
    if not is_numeric:
        return vals

    return np.array(vals)


def _resolve_array_param_value(param: "Parameter") -> ArrayLike:
    if not isinstance(param.value, (Array, np.ndarray)):
        raise ValueError("param.value must be an Array or ndarray")

    if not ParameterCache.get_dependents(param):
        return param.value

    return _resolve_array_like(param.value)


def _list_to_str(lst: list):
    str_repr = []
    for val in lst:
        if isinstance(val, list):
            str_repr.append(_list_to_str(val))
        else:
            str_repr.append(str(val))
    return f"[{', '.join(str_repr)}]"


def _tuple_to_str(tpl):
    str_repr = []
    for val in tpl:
        if isinstance(val, tuple):
            str_repr.append(_tuple_to_str(val))
        else:
            str_repr.append(str(val))
    if len(str_repr) == 1:
        return f"({str_repr[0]},)"
    return f"({', '.join(str_repr)})"


def _compute_list(tpl, is_tuple):
    new_lst = []
    for val in tpl:
        if isinstance(val, Parameter):
            new_lst.append(val.get())
        elif isinstance(val, list):
            new_lst.append(_compute_list(val, is_tuple=False))
        elif isinstance(val, tuple):
            new_lst.append(_compute_list(val, is_tuple=True))
        else:
            new_lst.append(val)
    if is_tuple:
        return tuple(new_lst)
    return new_lst


def _add_dependents(lst, param):
    for val in lst:
        if isinstance(val, Parameter):
            ParameterCache.add_dependent(val, param)
        elif isinstance(val, (list, tuple)):
            _add_dependents(val, param)


def _str_to_expression(s: str) -> str:
    # repr will add quotes and escape the string so that it can be passed again to
    # eval().
    return repr(s)


def _array_to_str(arr: Array | np.ndarray):
    if arr.ndim == 0:
        return str(arr.item())

    if isinstance(arr, Array):
        # Should we serialize as jnp.array?
        if arr.weak_type:
            return f"np.array({arr.tolist()})"
        if arr.dtype in (np.int64, np.float64):
            return f"np.array({arr.tolist()})"
        return f"np.array({arr.tolist()}, dtype=np.{arr.dtype})"

    if arr.dtype != np.object_:
        if arr.dtype in (np.int64, np.float64):
            return f"np.array({arr.tolist()})"
        if np.issubdtype(arr.dtype, np.str_):
            return f"np.array({arr.tolist()})"
        return f"np.array({arr.tolist()}, dtype=np.{arr.dtype})"

    return f"np.array({_list_to_str(arr.tolist())})"


def _value_as_str(value) -> str:
    if isinstance(value, ArrayLikeTypes):
        if isinstance(value, (Array, np.ndarray)):
            return _array_to_str(value)
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, np.number):
            dtype = value.dtype
            return f"np.{dtype}({value.item()})"
        return str(value)

    if isinstance(value, ParameterExpr):
        i = 0
        str_repr = []
        while i < len(value):
            val = value[i]
            if isinstance(val, Parameter):
                val_str = val.name if val.name is not None else str(val)
                if isinstance(val.value, ParameterExpr):
                    val_str = f"({val_str})"
                str_repr.append(val_str)
            elif isinstance(val, Ops):
                if val in (Ops.NEG, Ops.POS, Ops.ABS):
                    if i + 1 >= len(value):
                        raise ValueError()
                    next_val = value[i + 1]
                    if val is Ops.ABS:
                        str_repr.append(f"abs({next_val})")
                    elif val is Ops.NEG:
                        str_repr.append(f"-{next_val}")
                    elif val is Ops.POS:
                        str_repr.append(f"+{next_val}")
                    i += 1
                else:
                    str_repr.append(__OPS_STR__[val])
            elif isinstance(val, (Array, np.ndarray)):
                str_repr.append(f"np.array({val.tolist()})")
            else:
                str_repr.append(str(val))
            i += 1

        t = " ".join(str_repr)
        return t

    if isinstance(value, list):
        return _list_to_str(value)

    if isinstance(value, tuple):
        return _tuple_to_str(value)

    if isinstance(value, str):
        return _str_to_expression(value)

    if value is None:
        return ""

    # FIXME special case for jaxlite should not be necessary
    if not IS_JAXLITE or not isinstance(value, Parameter):
        # if is a Pytree
        try:
            value, _ = ravel_pytree(value)
            return _value_as_str(value)
        except BaseException:
            # Not a pytree
            pass

    return str(value)


class ParameterCache:
    __dependents__: dict["Parameter", set["Parameter"]] = {}
    __cache__: dict["Parameter", ArrayLike] = {}
    __is_dirty__ = defaultdict(lambda: True)

    @classmethod
    def get(cls, param: "Parameter") -> ArrayLike:
        if cls.__is_dirty__[param]:
            cls.__cache__[param] = cls.__compute__(param)
            cls.__is_dirty__[param] = False

        return cls.__cache__[param]

    @classmethod
    def replace(cls, param: "Parameter", value: ArrayLike):
        # TODO: update dependencies of this parameter
        param.value = value
        cls.__invalidate__(param)

    @classmethod
    def remove(cls, param: "Parameter"):
        for dependents in cls.__dependents__.values():
            if param in dependents:
                dependents.remove(param)

        if param in cls.__dependents__:
            del cls.__dependents__[param]
        if param in cls.__cache__:
            del cls.__cache__[param]
        if param in cls.__is_dirty__:
            del cls.__is_dirty__[param]

    @classmethod
    def add_dependent(cls, param: "Parameter", dependent: "Parameter"):
        # Mark 'dependent' as having a dependency on 'param', that is,
        # 'param' is built as an expression that involves 'dependent'.
        cls.__dependents__[param].add(dependent)

    @classmethod
    def get_dependents(cls, param: "Parameter"):
        return cls.__dependents__[param]

    @classmethod
    def static_dependents(cls, param: "Parameter"):
        dependents = set()
        for dependent in cls.__dependents__[param]:
            if dependent.is_static:
                dependents.add(dependent)
            dependents |= cls.static_dependents(dependent)
        return dependents

    @classmethod
    def __invalidate__(cls, param: "Parameter"):
        cls.__cache__[param] = None
        cls.__is_dirty__[param] = True
        for dependent in cls.__dependents__[param]:
            cls.__invalidate__(dependent)

    @classmethod
    def __compute__(cls, param: "Parameter"):
        if isinstance(param.value, ParameterExpr):
            acc = None
            right_value = None
            op = None
            i = 0

            while i < len(param.value):
                val = param.value[i]

                if isinstance(val, Parameter):
                    right_value = val.get()
                elif isinstance(val, ArrayLikeTypes):
                    right_value = val
                elif isinstance(val, Ops):
                    if val in (Ops.NEG, Ops.POS, Ops.ABS):
                        if i + 1 >= len(param.value):
                            raise ParameterError(
                                param, message="Invalid parameter value"
                            )
                        if isinstance(param.value[i + 1], Parameter):
                            right_value = __OPS_FN__[val](param.value[i + 1].get())
                        elif isinstance(param.value[i + 1], ArrayLikeTypes):
                            right_value = __OPS_FN__[val](param.value[i + 1])
                        else:
                            raise ParameterError(
                                param,
                                message=f"Invalid value in parameter list: {param.value[i + 1]} of type {type(param.value[i + 1])}",
                            )
                        i += 1
                    else:
                        op = val
                else:
                    raise ParameterError(
                        param,
                        message=f"Invalid value in parameter list: {val} of type {type(val)}",
                    )

                if acc is not None and right_value is not None and op is not None:
                    acc = __OPS_FN__[op](acc, right_value)
                    op = None
                    right_value = None
                elif right_value is not None:
                    acc = right_value
                    right_value = None
                i += 1

            if acc is not None:
                return acc
            if right_value is not None:
                return right_value
            raise ParameterError(param, message="Invalid parameter value")

        if isinstance(param.value, Parameter):
            return cls.__compute__(param.value)

        if isinstance(param.value, tuple):
            t = _compute_list(param.value, is_tuple=True)
            return t

        if isinstance(param.value, list):
            t = _compute_list(param.value, is_tuple=False)
            return t

        if isinstance(param.value, np.ndarray):
            vals = _resolve_array_param_value(param)
            return np.array(vals, dtype=param.value.dtype)

        if isinstance(param.value, Array):
            vals = _resolve_array_param_value(param)
            if param.value.weak_type:
                return jnp.array(vals)
            return jnp.array(vals, dtype=param.value.dtype)

        if isinstance(param.value, np.number):
            if isinstance(param.value.item(), Parameter):
                return type(param.value)(cls.__compute__(param.value.item()))
            return param.value

        if isinstance(param.value, str) and param.is_python_expr:
            _, resolved_parameters = resolve_parameters(param.value, param.py_namespace)
            return eval(
                param.value,
                param.py_namespace,
                {**param.py_namespace, **resolved_parameters},
            )

        return param.value


def _op(op: Ops, left, right):
    param = Parameter(
        value=ParameterExpr([left, op, right]),
    )
    if isinstance(left, Parameter):
        ParameterCache.add_dependent(left, param)
    if isinstance(right, Parameter):
        ParameterCache.add_dependent(right, param)
    return param


def _record_parameter_creation(parameter):
    args = {}
    for field_info in dataclasses.fields(parameter):
        field_name = field_info.name
        field_value = getattr(parameter, field_name)
        default_value = field_info.default
        if field_name != "value" and field_value != default_value:
            args[field_name] = field_value
    args["value"] = _value_as_str(parameter.value)
    build_recorder.create_parameter(args)


@dataclasses.dataclass
class Parameter:
    value: Union[ParameterExpr, "Parameter", ArrayLike, str, tuple]

    # shape & dtype are set at init time when constructing the parameter,
    # they are not necessarily the actual value's shape and dtype
    dtype: DTypeLike = None
    shape: ShapeLike = None
    as_array: bool = False

    # name is used by reference submodels, model parameters and init script
    # variables so that they can be referred to in other fields
    # (we need this for serialization).
    name: str = None

    # For complex parameter values, we can specify a Python expression as string
    # This is useful for expressions like "np.eye(p)" where p is a parameter.
    is_python_expr: bool = False
    py_namespace: dict = None

    is_static: bool = False  # TODO: staticness should be propagated to dependents
    system: "SystemBase" = None

    def get(self):
        value = ParameterCache.get(self)
        if self.as_array:
            value = utils.make_array(value, self.dtype, self.shape)
        return value

    def set(self, value: Union["Parameter", ArrayLike, str, tuple]):
        ParameterCache.replace(self, value)

    def static_dependents(self):
        return ParameterCache.static_dependents(self)

    def __post_init__(self):
        ParameterCache.__dependents__[self] = set()

        if isinstance(self.value, Parameter):
            ParameterCache.add_dependent(self.value, self)
        if isinstance(self.value, ParameterExpr):
            for val in self.value:
                if isinstance(val, Parameter):
                    ParameterCache.add_dependent(val, self)
        if isinstance(self.value, (list, tuple)):
            _add_dependents(self.value, self)

        _record_parameter_creation(self)

    def __add__(self, other):
        return _op(Ops.ADD, self, other)

    def __radd__(self, other):
        return _op(Ops.ADD, other, self)

    def __sub__(self, other):
        return _op(Ops.SUB, self, other)

    def __rsub__(self, other):
        return _op(Ops.SUB, other, self)

    def __mul__(self, other):
        return _op(Ops.MUL, self, other)

    def __rmul__(self, other):
        return _op(Ops.MUL, other, self)

    def __truediv__(self, other):
        return _op(Ops.DIV, self, other)

    def __rtruediv__(self, other):
        return _op(Ops.DIV, other, self)

    def __floordiv__(self, other):
        return _op(Ops.FLOORDIV, self, other)

    def __rfloordiv__(self, other):
        return _op(Ops.FLOORDIV, other, self)

    def __mod__(self, other):
        return _op(Ops.MOD, self, other)

    def __rmod__(self, other):
        return _op(Ops.MOD, other, self)

    def __pow__(self, other):
        return _op(Ops.POW, self, other)

    def __rpow__(self, other):
        return _op(Ops.POW, other, self)

    def __neg__(self):
        p = Parameter(value=ParameterExpr([Ops.NEG, self]))
        ParameterCache.add_dependent(self, p)
        return p

    def __pos__(self):
        p = Parameter(value=ParameterExpr([Ops.POS, self]))
        ParameterCache.add_dependent(self, p)
        return p

    def __abs__(self):
        p = Parameter(value=ParameterExpr([Ops.ABS, self]))
        ParameterCache.add_dependent(self, p)
        return p

    def __eq__(self, other):
        return _op(Ops.EQ, self, other)

    def __ne__(self, other):
        return _op(Ops.NE, self, other)

    def __lt__(self, other):
        return _op(Ops.LT, self, other)

    def __le__(self, other):
        return _op(Ops.LE, self, other)

    def __gt__(self, other):
        return _op(Ops.GT, self, other)

    def __ge__(self, other):
        return _op(Ops.GE, self, other)

    def __del__(self):
        ParameterCache.remove(self)

    def __hash__(self):
        return id(self)

    def __str__(self):
        # Calling str() on a Parameter object is confusing. What's the intent?
        # 1. Serializing to a valid Python expression?
        # 2. Is it for logs? For debugging?
        # 3. Is it part of building a wider expression (like a list of parameters)?
        # 4. Evaluating the actual value of a string parameter?
        # Here, we support 2 & 4. We'll likely have to change this when we want support
        # for non-literal string parameters in the UI.

        expr, _ = self.value_as_api_param(
            allow_param_name=True,
            allow_string_literal=True,
        )
        return expr

    def __matmul__(self, other):
        return _op(Ops.MATMUL, self, other)

    def __int__(self):
        if self.dtype is not None:
            return self.dtype(self.get())
        return int(self.get())

    def __float__(self):
        if self.dtype is not None:
            return self.dtype(self.get())
        return float(self.get())

    # FIXME: this is not working as expected - it will break some tests
    # def __bool__(self):
    #     return bool(self.get())

    def __complex__(self):
        return complex(self.get())

    def value_as_api_param(
        self, allow_param_name=True, allow_string_literal=True
    ) -> tuple[str, bool]:
        """Returns an API-compatible expression[1] that defines this parameter

        What we return depends on the caller's context, since it depends on
        whether we are serializing for a model, submodel or block parameter.

        The boolean is the value of 'is_string' (means "string literal" or
        "do not call eval").

        [1] The returned string can be serialized to JSON, but it is not an
            already escaped JSON string!

        Args:
            allow_param_name: Set to false for (sub)model parameters. Optional.
                If true, and the value is defined by a name, just the name will
                be returned.
            allow_string_literal: Set to false for (sub)model parameters. Optional.
                If true, and the value is a string, then the string will be
                returned and 'is_string' will be returned as True.
        """
        if self.name is not None and allow_param_name:
            return self.name, False

        if self.is_python_expr and isinstance(self.value, str):
            return self.value, False

        if allow_string_literal and isinstance(self.value, str):
            return self.value, True

        return _value_as_str(self.value), False

    def __repr__(self):
        ex, _ = self.value_as_api_param(allow_string_literal=False)
        if len(ex) > 100:
            ex = ex[:50] + "..." + ex[-50:]

        return (
            f"Parameter(name={self.name}, value={ex}, "
            f"value_type={type(self.value).__name__}, "
            f"is_static={self.is_static}, "
            f"is_python_expr={self.is_python_expr}, "
            f"id={self.__hash__()}"
            ")"
        )
