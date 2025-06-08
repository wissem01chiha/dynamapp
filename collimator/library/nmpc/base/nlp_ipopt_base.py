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

"""Base classes for Ipopt NLP structures"""

from abc import ABC, abstractmethod
from functools import partial

from typing import Tuple

import jax
import jax.numpy as jnp


class IpoptStructureBase(ABC):
    """
    Abstract class for the necessary structure of the `cyipopt` interface to IPOPT.
    Methods `constraints` and `jacobian` are not strictly necessary as the NLP
    problem may not have any constraints.
    """

    @abstractmethod
    def objective(self, optvars):
        """Evaluate the NLP objective function"""
        pass

    @abstractmethod
    def gradient(self, optvars):
        """Evaluate the gradient of the NLP objective function"""
        pass

    def constraints(self, optvars):
        """Evaluate the NLP constraints. Optional override if constraints exist."""
        pass

    def jacobian(self, optvars):
        """
        Evaluate the Jacobian of the NLP constraints.
        Optional override if constraints exist.
        """
        pass

    @abstractmethod
    def hessianstructure(self) -> Tuple:
        """Return the structure of the Hessian matrix"""
        pass

    @abstractmethod
    def hessian(self, optvars, lagrange_multipliers, obj_factor):
        """
        Return the Hessian of the NLP Lagrangian with respect to the optimization
        variables.
        """
        pass


class NMPCProblemStructure(IpoptStructureBase):
    """
    Base class for the NLP problem structure with the Ipopt solver for a nonlinear
    MPC problem.

    Inheriting from `IpoptStructureBase`, this class also provides concrete
    implementations for the following methods with JAX's automatic differentiation
    framework:

        `gradient`,
        `jacobian`,
        `hessianstructure`, and
        `hessian`,

    while the `objective` and `constraints` are abstract methods must be provided
    by the user at initialization.

    Parameters:

    num_optvars : int
        Number of optimization variables in the NLP problem

    objective: Callable
        Function to evaluate the objective function for the NLP problem.
        Sample signature:
            ```
            def objective(self, optvars, t0, x0, x_ref, u_ref):
                Method to evaluate the objective function for the NLP problem.
                Parameters:
                    optvars (Array):
                        Flattened array of optimization variables.
                    t0 (float):
                        Initial/current time (beginning of the MPC step).
                    x0 (Array):
                        Initial/current state of the plant.
                    x_ref (Array):
                        Reference trajectory for the state vector.
                    u_ref (Array):
                        Reference trajectory for the control input vector.

                Returns: float or Array of size (1,)
                    NLP objective function evaluated at `optvars`.
            ```

    constraints: Callable
        Function to evaluate the constraint functions for the NLP problem.
        Sample signature:
            ```
            def constraints(self, optvars, t0, x0, x_ref, u_ref):
                Method to evaluate the objective function for the NLP problem.
                Parameters:
                    optvars (Array):
                        Flattened array of optimization variables.
                    t0 (float):
                        Initial/current time (beginning of the MPC step).
                    x0 (Array):
                        Initial/current state of the plant.
                    x_ref (Array):
                        Reference trajectory for the state vector.
                    u_ref (Array):
                        Reference trajectory for the control input vector.

                Returns: float or Array of size (`num_constraints`,)
                    Constraint functions evaluated at the `optvars`.
            ```
    """

    def __init__(self, num_optvars, objective, constraints=None):
        self.num_optvars = num_optvars
        self.has_constraints = constraints is not None
        self._objective = objective
        if self.has_constraints:
            self._constraints = constraints

    def update_nlp_structure(self, t_curr, x_curr, x_ref, u_ref):
        """
        Utility method to update the initial conditions and reference trajectories
        for the NLP problem. This method is called at the beginning of each
        MPC iteration to update the NLP problem structure.
        """
        self.t0 = t_curr
        self.x0 = x_curr
        self.x_ref = x_ref
        self.u_ref = u_ref

    def objective(self, optvars):
        """Evaluate the NLP objective function"""
        return self._objective(optvars, self.t0, self.x0, self.x_ref, self.u_ref)

    def gradient(self, optvars):
        """Evaluate the gradient of the NLP objective function"""
        return self._gradient(optvars, self.t0, self.x0, self.x_ref, self.u_ref)

    @partial(jax.jit, static_argnames=("self",))
    def _gradient(self, optvars, t0, x0, x_ref, u_ref):
        return jax.grad(self._objective)(optvars, t0, x0, x_ref, u_ref)

    def constraints(self, optvars):
        """Evaluate the NLP constraint functions"""
        if self.has_constraints:
            return self._constraints(optvars, self.t0, self.x0, self.x_ref, self.u_ref)

    def jacobian(self, optvars):
        """
        Evaluate the Jacobian of the constraint functions with respect to the
        optimization variables.
        """
        if self.has_constraints:
            return self._jacobian(optvars, self.t0, self.x0, self.x_ref, self.u_ref)

    @partial(jax.jit, static_argnames=("self",))
    def _jacobian(self, optvars, t0, x0, x_ref, u_ref):
        return jax.jacobian(self._constraints)(optvars, t0, x0, x_ref, u_ref)

    def hessianstructure(self):
        """
        Returns lower triangular row and column indices for a matrix
        of size (num_optvars, num_optvars)
        """
        return jnp.nonzero(jnp.tril(jnp.ones((self.num_optvars, self.num_optvars))))

    def hessian_objective(self, optvars):
        """
        Evaluate the Hessian of the objective function with respect to the optimization
        variables.
        """
        return self._hessian_objective(
            optvars, self.t0, self.x0, self.x_ref, self.u_ref
        )

    @partial(jax.jit, static_argnames=("self",))
    def _hessian_objective(self, optvars, t0, x0, x_ref, u_ref):
        return jax.hessian(self._objective)(optvars, t0, x0, x_ref, u_ref)

    def hessian_constraints(self, optvars):
        """
        Evaluate the Hessian of the constraints functions with respect to the
        optimization variables.
        """
        return self._hessian_constraints(
            optvars, self.t0, self.x0, self.x_ref, self.u_ref
        )

    @partial(jax.jit, static_argnames=("self",))
    def _hessian_constraints(self, optvars, t0, x0, x_ref, u_ref):
        return jax.hessian(self._constraints)(optvars, t0, x0, x_ref, u_ref)

    def hessian(self, optvars, lagrange_multipliers, obj_factor):
        """
        Computes the Hessian of the NLP Lagrangian with respect to the optimization
        variables. The Lagrangian `L` is defined as:

        ```
        L = obj_factor * f_obj + Sum over i of: lagrange_multipliers[i] * constraints[i]
        ```

        where `f_obj` is the objective function of the NLP problem.

        The Hessian is calculated as:

        Hessian = obj_factor * Hessian(f_obj) +
                  Sum over i of: lagrange_multipliers[i] * Hessian(constraints[i])

        Parameters:
        optvars: Array:
            Flattened array of optimization variables at which the Hessian is computed.

        lagrange_multipliers: Array:
            Array of Lagrange multipliers corresponding to the constraints of the NLP.

        obj_factor: float
            Scaling factor for the objective function in the Lagrangian.

        Returns:
        Array: The Hessian matrix of the Lagrangian at the given point (optvars),
               structured according to the non-zero pattern defined by the
               `hessianstructure` method.
        """
        Ho = obj_factor * self.hessian_objective(optvars)
        lm = jnp.array(lagrange_multipliers)
        lm = lm[:, jnp.newaxis, jnp.newaxis]
        Hc = 0.0
        if self.has_constraints:
            Hc = jnp.sum(jnp.multiply(lm, self.hessian_constraints(optvars)), axis=0)
        H = Ho + Hc
        row, col = self.hessianstructure()
        return H[row, col]
