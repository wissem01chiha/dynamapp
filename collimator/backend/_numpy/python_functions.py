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

from typing import TYPE_CHECKING
import numpy as np
from jax.tree_util import tree_map
import warnings
from ...lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    from scipy.interpolate import LinearNDInterpolator
else:
    scipy = LazyLoader("scipy", globals(), "scipy")
    LinearNDInterpolator = LazyModuleAccessor(scipy, "interpolate.LinearNDInterpolator")

__all__ = ["cond", "scan", "while_loop", "callback", "jit", "astype", "interp2d"]


def cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def scan(f, init, xs, length=None):
    warnings.warn("Using scan with numpy backend. This can be extremely slow.")

    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    # pylint: disable=no-value-for-parameter
    stacked_ys = tree_map(lambda *ys: np.stack(ys), *ys)
    return carry, stacked_ys


def while_loop(cond_fun, body_fun, init):
    carry = init
    while cond_fun(carry):
        carry = body_fun(carry)
    return carry


def fori_loop(start, stop, body_fun, init):
    carry = init
    for i in range(start, stop):
        carry = body_fun(i, carry)
    return carry


def callback(callback, result_shape_dtypes, *args, **kwargs):
    return callback(*args, **kwargs)


# Dummy placeholder for JIT compilation
def jit(fun, *args, **kwargs):
    return fun


def astype(x, dtype, copy=True):
    return x.astype(dtype)


# this should work, but it fails in testing. strange because
# it's the same code used in testing to verify numerical results.
def interp2d_scipy_wrapper(xp, yp, zp, inputs, fill_value=None):
    x, y = np.meshgrid(xp, xp, indexing="ij")
    xy = np.vstack([x.flatten(), y.flatten()]).T
    f = scipy.interpolate.LinearNDInterpolator(xy, zp.flatten())

    x = np.array(inputs[0])
    y = np.array(inputs[1])

    x = np.clip(x, xp[0], xp[-1])
    y = np.clip(y, yp[0], yp[-1])

    return f(x, y)


# same fucntion as _jax.python_functions.interp2d. see comment there for more details.
def interp2d(xp, yp, zp, x, y, fill_value=None):
    """
    Bilinear interpolation on a grid.

    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`

    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.

    https://github.com/adam-coogan/jaxinterp2d/blob/master/src/jaxinterp2d/__init__.py
    """

    x = np.array(x)
    y = np.array(y)

    x = np.clip(x, xp[0], xp[-1])
    y = np.clip(y, yp[0], yp[-1])

    ix = np.clip(np.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = np.clip(np.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = np.logical_or(
            x < xp[0],
            np.logical_or(x > xp[-1], np.logical_or(y < yp[0], y > yp[-1])),
        )
        z = np.where(oob, fill_value, z)

    return z
