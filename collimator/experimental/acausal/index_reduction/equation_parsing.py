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

import re
from typing import Optional, TYPE_CHECKING

from collimator.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")


def preprocess_derivatives(expr_str):
    # Define a regex pattern to match common derivative notations like dx(t)/dt
    pattern = re.compile(r"d([a-zA-Z_]\w*)\(t\)/dt")

    def replace_derivative(match):
        func_name = match.group(1)
        return f"Derivative({func_name}(t), t)"

    # Replace all occurrences of the pattern with the correct Derivative syntax
    preprocessed_str = pattern.sub(replace_derivative, expr_str)

    return preprocessed_str


def parse_expression(expr_str):
    preprocessed_str = preprocess_derivatives(expr_str)
    expr = sp.sympify(preprocessed_str)
    return expr


def parse_string_inputs(
    eqs: list[str],
    knowns: dict[str, float],
    ics: dict[str, float],
    ics_weak: Optional[dict[str, float]] = None,
):
    t = sp.symbols("t")  # Reserved symbol for time

    sym_eqs = [parse_expression(eq) for eq in eqs]
    sym_knowns = {parse_expression(k): v for k, v in knowns.items()}
    sym_ics = {parse_expression(k): v for k, v in ics.items()}
    sym_ics_weak = (
        {parse_expression(k): v for k, v in ics_weak.items()} if ics_weak else None
    )

    return t, sym_eqs, sym_knowns, sym_ics, sym_ics_weak
