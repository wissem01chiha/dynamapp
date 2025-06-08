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

from ..lazy_loader import LazyLoader

torch = LazyLoader("torch", globals(), "torch")

__all__ = ["torch_functions", "torch_constants"]


def cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def zeros_like(x):
    return torch.zeros(*x.shape, dtype=x.dtype)


def torch_functions():
    return (
        {
            "asarray": torch.as_tensor,
            "array": torch.tensor,
            "zeros_like": zeros_like,
            "cond": cond,
        }
        if torch is not None
        else {}
    )


torch_constants = (
    {
        "lib": torch,
    }
    if torch is not None
    else {}
)
