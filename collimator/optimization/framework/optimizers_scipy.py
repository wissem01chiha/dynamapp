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
Scipy/JAX-scipy optimizers from `scipy.optimize.minimize` and
`jax.scipy.optimize.minimize`.
"""

from functools import partial
from typing import TYPE_CHECKING
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from collimator.logging import logger
from collimator.optimization.framework.base.metrics import MetricsWriter

from .base import Optimizer, Optimizable

from ...lazy_loader import LazyLoader

if TYPE_CHECKING:
    import jax.scipy.optimize as jaxopt
    import scipy.optimize as sciopt
else:
    jaxopt = LazyLoader("jaxopt", globals(), "jax.scipy.optimize")
    sciopt = LazyLoader("sciopt", globals(), "scipy.optimize")


# From scipy.optimize._minimize.py:
MINIMIZE_METHODS_NEW_CB = [
    "nelder-mead",
    "powell",
    "cg",
    "bfgs",
    "newton-cg",
    "l-bfgs-b",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
    "cobyqa",
]

ACCEPTS_GRAD = [
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
    "trust-constr",
]

SUPPORTS_BOUNDS = [
    "Nelder-Mead",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "Powell",
    "trust-constr",
    "COBYLA",
]

SUPPORTS_CONSTRAINTS = [
    "COBYLA",
    "SLSQP",
    "trust-constr",
]


class Scipy(Optimizer):
    """
    Scipy/JAX-scipy optimizers.

    Parameters:
        optimizable (Optimizable):
            The optimizable object.
        opt_method (str):
            The optimization method to use.
        tol (float):
            Tolerance for termination. For detailed control, use `opt_method_config`.
        opt_method_config (dict):
            Configuration for the optimization method.
        use_autodiff_grad (bool):
            Whether to use autodiff for gradient computation.
        use_jax_scipy (bool):
            Whether to use JAX's version of `optimize.minimize`.
    """

    def __init__(
        self,
        optimizable: Optimizable,
        opt_method,
        tol=None,
        opt_method_config=None,
        use_autodiff_grad=True,
        use_jax_scipy=False,
        metrics_writer: MetricsWriter = None,
    ):
        self.optimizable = optimizable
        self.opt_method = opt_method
        self.tol = tol
        self.opt_method_config = opt_method_config or {}
        self.use_autodiff_grad = use_autodiff_grad
        self.use_jax_scipy = use_jax_scipy
        self.optimal_params = None
        self.metrics_writer = metrics_writer

    def optimize(self):
        """Run optimization"""
        params = self.optimizable.params_0_flat
        objective = jax.jit(self.optimizable.objective_flat)

        if self.use_jax_scipy:
            warnings.warn(
                "`use_jax_scipy` is True. JAX's version of optimize.minimize will be "
                "used. Consequently, `opt_method` will be set of `BFGS` and autodiff "
                "will be used for gradient computation. Constraints and bounds will "
                "be ignored. If you want to use scipy's version of minimize, set "
                " `use_jax_scipy` to False."
            )
            opt_res = jaxopt.minimize(
                objective,
                params,
                method="BFGS",
                tol=self.tol,
                options=self.opt_method_config,
            )
            params = opt_res.x

        else:
            use_jac = False
            if self.opt_method in ACCEPTS_GRAD and self.use_autodiff_grad:
                jac = jax.jit(jax.grad(objective))
                use_jac = True

            # Handle bounds
            bounds = self.optimizable.bounds_flat

            # Jobs from UI would put (-jnp.inf, jnp.inf) as defualt bounds. The user
            # may also have specified bounds this way. Scipy expects `None` to imply
            # unboundedness.
            if bounds is not None:
                bounds = [
                    (
                        None if b[0] == -jnp.inf else b[0],
                        None if b[1] == jnp.inf else b[1],
                    )
                    for b in bounds
                ]

                # Check if all bounds are None, i.e. no bounds at all, and hence
                # algorithms that do not support bounds can be used.
                flattened_bounds = [element for tup in bounds for element in tup]
                all_none = all(element is None for element in flattened_bounds)
                bounds = None if all_none else bounds

            if bounds is not None and self.opt_method not in SUPPORTS_BOUNDS:
                raise ValueError(
                    f"Optimization method scipy:{self.opt_method} "
                    "does not support bounds."
                )

            # Handle constraints
            if (
                self.optimizable.has_constraints
                and self.opt_method not in SUPPORTS_CONSTRAINTS
            ):
                raise ValueError(
                    f"Optimization method scipy:{self.opt_method} "
                    "does not support constraints."
                )

            if self.optimizable.has_constraints:
                constraints = jax.jit(self.optimizable.constraints_flat)
                constraints_jac = jax.jit(jax.jacrev(constraints))
                constraints = sciopt.NonlinearConstraint(
                    constraints, 0.0, jnp.inf, jac=constraints_jac
                )
            else:
                constraints = None

            if self.metrics_writer is not None:
                cb = (
                    self._scipy_callback_new
                    if self.opt_method in MINIMIZE_METHODS_NEW_CB
                    else partial(self._scipy_callback_legacy, objective)
                )
            else:
                cb = None

            opt_res: "sciopt.OptimizeResult" = sciopt.minimize(
                objective,
                params,
                method=self.opt_method,
                jac=jac if use_jac else None,
                bounds=bounds,
                constraints=constraints,
                tol=self.tol,
                options=self.opt_method_config,
                callback=cb,
            )

            params = opt_res.x

            # Show the raw information from scipy. This can help with debugging.
            logger.info("Optimization result:\n%s", opt_res)

            if not opt_res.success:
                logger.warning("Optimization did not converge: %s", opt_res.message)

        self.optimal_params = self.optimizable.unflatten_params(params)
        if self.optimizable.transformation is not None:
            self.optimal_params = self.optimizable.transformation.inverse_transform(
                self.optimal_params
            )
        return self.optimal_params

    # NOTE: if this turns out to be too expensive, we can throttle writes in the
    # MetricsWriter and only compute metrics when we need them.
    def _write_metrics(self, fun, x):
        metrics = {}
        if fun is not None:
            metrics["fun"] = fun
        if x is not None:
            params: dict = self.optimizable.unflatten_params(x)
            for k, v in params.items():
                if np.asarray(v).shape == ():
                    metrics[k] = v
        if len(metrics) > 0:
            self.metrics_writer.write_metrics(**metrics)

    def _scipy_callback_new(self, intermediate_result: "sciopt.OptimizeResult"):
        self._write_metrics(
            intermediate_result.get("fun"), intermediate_result.get("x")
        )

    def _scipy_callback_legacy(self, objective, intermediate_results: np.ndarray):
        fun = objective(intermediate_results)
        self._write_metrics(fun, intermediate_results)
