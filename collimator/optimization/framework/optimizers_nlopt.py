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
Optimizers using the NLopt library.
"""

import jax
import jax.numpy as jnp

from collimator.lazy_loader import LazyLoader

from .base import Optimizer, Optimizable

nlopt = LazyLoader(
    "nlopt",
    globals(),
    "nlopt",
    error_message="nlopt is not installed.",
)

SUPPORTS_BOUNDS = {
    # FIXME: Extend this list
    "lbfgs": lambda: nlopt.LD_LBFGS,
    "slsqp": lambda: nlopt.LD_SLSQP,
    "cobyla": lambda: nlopt.LN_COBYLA,
    "mma": lambda: nlopt.LD_MMA,
    "isres": lambda: nlopt.GN_ISRES,
    "ags": lambda: nlopt.GN_AGS,
    "direct": lambda: nlopt.GN_ORIG_DIRECT,
}

SUPPORTS_CONSTRAINTS = {
    "slsqp": lambda: nlopt.LD_SLSQP,
    "cobyla": lambda: nlopt.LN_COBYLA,
    "mma": lambda: nlopt.LD_MMA,
    "isres": lambda: nlopt.GN_ISRES,
    "ags": lambda: nlopt.GN_AGS,
    "direct": lambda: nlopt.GN_ORIG_DIRECT,
}

ALL_METHODS = {**SUPPORTS_BOUNDS, **SUPPORTS_CONSTRAINTS}


class NLopt(Optimizer):
    """
    Optimizers using the NLopt library.

    Parameters:
        optimizable (Optimizable):
            The optimizable object.
        opt_method (str):
            The optimization method to use.
        ftol_rel (float):
            Relative tolerance on function value.
        ftol_abs (float):
            Absolute tolerance on function value.
        xtol_rel (float):
            Relative tolerance on optimization parameters.
        xtol_abs (float):
            Absolute tolerance on optimization parameters.
        cons_tol (float):
            Tolerance on constraints.
        maxeval (int):
            Maximum number of function evaluations.
        maxtime (float):
            Maximum time in seconds.
    """

    def __init__(
        self,
        optimizable: Optimizable,
        opt_method: str,
        ftol_rel=1e-06,
        ftol_abs=1e-06,
        xtol_rel=1e-06,
        xtol_abs=1e-06,
        cons_tol=1e-06,
        maxeval=500,
        maxtime=0,
    ):
        self.optimizable = optimizable
        self.opt_method = opt_method
        self.ftol_rel = ftol_rel
        self.ftol_abs = ftol_abs
        self.xtol_rel = xtol_rel
        self.xtol_abs = xtol_abs
        self.cons_tol = cons_tol
        self.maxeval = maxeval
        self.maxtime = maxtime
        self.optimal_params = None

    def optimize(self):
        """Run optimization"""
        params = self.optimizable.params_0_flat
        objective = jax.jit(self.optimizable.objective_flat)
        gradient = jax.jit(jax.grad(objective))

        constraints = jax.jit(self.optimizable.constraints_flat)
        constraints_jac = jax.jit(jax.jacrev(constraints))

        def nlopt_obj(x, grad):
            if grad.size > 0:
                grad[:] = gradient(jnp.array(x))
            return float(objective(jnp.array(x)))

        def nlopt_cons(result, x, grad):
            if grad.size > 0:
                grad[:, :] = -constraints_jac(jnp.array(x))
            result[:] = -constraints(jnp.array(x))

        if (
            self.optimizable.bounds_flat is not None
            and self.opt_method not in SUPPORTS_BOUNDS
        ):
            raise ValueError(
                f"Optimization method nlopt:{self.opt_method} does not support bounds."
            )

        if (
            self.optimizable.has_constraints
            and self.opt_method not in SUPPORTS_CONSTRAINTS
        ):
            raise ValueError(
                f"Optimization method nlopt:{self.opt_method} "
                "does not support constraints."
            )

        if self.opt_method not in ALL_METHODS:
            raise ValueError(
                f"Optimization method nlopt:{self.opt_method} is not supported."
            )

        # Initialize nlopt optimizer
        opt_method = ALL_METHODS[self.opt_method]()
        opt = nlopt.opt(opt_method, len(params))

        # Set the objective function
        opt.set_min_objective(nlopt_obj)

        # Set the constraints
        if self.optimizable.has_constraints:
            num_constraints = self.optimizable.constraints_flat(jnp.array(params)).size
            opt.add_inequality_mconstraint(
                nlopt_cons, [self.cons_tol] * num_constraints
            )

        # Set the bounds
        if self.optimizable.bounds_flat is not None:
            lower_bounds, upper_bounds = zip(*self.optimizable.bounds_flat)
            opt.set_lower_bounds(lower_bounds)
            opt.set_upper_bounds(upper_bounds)

        # Set stopping criteria
        opt.set_ftol_rel(self.ftol_rel)
        opt.set_ftol_abs(self.ftol_abs)
        opt.set_xtol_rel(self.xtol_rel)
        opt.set_xtol_abs(self.xtol_abs)
        opt.set_maxeval(self.maxeval)
        opt.set_maxtime(self.maxtime)

        # Run the optimization
        params = opt.optimize(params)

        self.optimal_params = self.optimizable.unflatten_params(params)
        if self.optimizable.transformation is not None:
            self.optimal_params = self.optimizable.transformation.inverse_transform(
                self.optimal_params
            )
        return self.optimal_params
