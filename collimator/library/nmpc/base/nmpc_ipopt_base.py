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

"""Base class for nonlinear MPCs utilizing Ipopt"""

from abc import abstractmethod
from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp

from collimator.lazy_loader import LazyLoader
from ....backend.typing import Array
from .nmpc_base import NonlinearMPCBase


_load_error_msg = (
    "CyIPOPT is not installed. NMPC and trajectory optimization blocks that rely "
    "on IPOPT will not be available: DirectShootingNMPC, DirectTranscriptionNMPC, "
    "and HermiteSimpsonNMPC. Install 'pycollimator[nmpc]' for NMPC blocks support."
)

cyipopt = LazyLoader(
    "cyipopt",
    globals(),
    "cyipopt",
    error_message=_load_error_msg,
)


class NonlinearMPCIpopt(NonlinearMPCBase):
    """
    Base class for nonlinear MPCs that utilise Ipopt (provided by `cyipopt`) for the
    solution of nonlinear programming problems (NLPs).

    Parameters:
        dt: float
            Time step in the prediction and control horizonsself.
        nu: int
            Size of the control input vector.
        num_optvars: int
            Number of optimization variables in the NLP.
        nlp_structure_ipopt: instance of `NMPCProblemStructure`
            NLP structure for the Ipopt solver.

    Notes:
        (i) Initialisation requires an appropriate `nlp_structure_ipopt` representing
            the structure of NLP problem as required by `cyipopt`.
        (ii) Inheriting from `NonlinearMPCBase`, this class provides a concrete
            implementation of the `_solve` method, and requires concrete
            implementations of the following properties by derived classes:
                `num_optvars`,
                `num_constraints`,
                `bounds_optvars`, and
                `bounds_constraints`.
    """

    def __init__(
        self,
        dt,
        nu,
        num_optvars,
        nlp_structure_ipopt,
        name=None,
    ):
        super().__init__(
            dt=dt,
            nu=nu,
            nopt=num_optvars,
            name=name,
        )

        self.nlp_structure_ipopt = nlp_structure_ipopt

    def _solve(self, time, state, *inputs):
        """see documentation for the abstract method in the base class"""
        self._callback = partial(jax.pure_callback, self.solve, self._result_template)
        return self._callback(time, state, *inputs)

    def solve(self, time, state, *inputs):
        """
        see documentation for the `_solve` method above
        """
        t_curr = time
        x_curr, x_ref, u_ref = inputs

        # Update the NLP structure
        self.nlp_structure_ipopt.update_nlp_structure(t_curr, x_curr, x_ref, u_ref)

        # FIXME: allow optional warm start
        optvars_guess = jnp.zeros(self.num_optvars)

        lb_optvars, ub_optvars = self.bounds_optvars
        lb_constraints, ub_constraints = self.bounds_constraints
        # Create the problem instance

        nlp = cyipopt.Problem(
            n=self.num_optvars,
            m=self.num_constraints,
            problem_obj=self.nlp_structure_ipopt,
            lb=lb_optvars,
            ub=ub_optvars,
            cl=lb_constraints,
            cu=ub_constraints,
        )

        # Set solver options
        nlp.add_option("mu_strategy", "adaptive")
        nlp.add_option("tol", 1e-7)

        nlp.add_option("print_level", 0)  # Silent

        # Solve the problem
        optvars_sol, info = nlp.solve(optvars_guess)

        if info["status"] != 0:
            status_msg = info["status_msg"]
            raise ValueError(
                f"Error occurred with the solver from 'CyIPOPT' library at {time=}. "
                f"Status message: {status_msg}"
            )

        return jnp.array(optvars_sol)

    def solve_trajectory_optimzation(
        self, t_curr, x_curr, x_ref, u_ref, x_optvars_guess, u_optvars_guess
    ):
        """
        Solve the NLP problem structure specified by `nlp_structure_ipopt`.

        This public method can be used to solve Trajectory Opimization problems, where
        the NLP solution needs to be obtained separately from the MPC loop. Once the
        above `solve` method is able to take initial guesses, the two methods can be
        combined into a single core solution method.

        Parameters:
        t_curr: float
            Current time.
        x_curr: Array
            Current state.
        x_ref: Array
            Reference trajectory for the state vector.
        u_ref: Array
            Reference trajectory for the control input vector.
        x_optvars_guess: Array
            Initial guess of the state trajectory.
        u_optvars_guess: Array
            Initial guess of the control input trajectory.

        Returns: Array
            Solution of the NLP problem.
        """
        x_curr = jnp.array(x_curr)

        # Update the NLP structure
        self.nlp_structure_ipopt.update_nlp_structure(t_curr, x_curr, x_ref, u_ref)

        optvars_guess = jnp.hstack([u_optvars_guess.ravel(), x_optvars_guess.ravel()])

        lb_optvars, ub_optvars = self.bounds_optvars
        lb_constraints, ub_constraints = self.bounds_constraints

        # Create the problem instance
        nlp = cyipopt.Problem(
            n=self.num_optvars,
            m=self.num_constraints,
            problem_obj=self.nlp_structure_ipopt,
            lb=lb_optvars,
            ub=ub_optvars,
            cl=lb_constraints,
            cu=ub_constraints,
        )

        # Set solver options
        nlp.add_option("mu_strategy", "adaptive")
        nlp.add_option("tol", 1e-7)

        nlp.add_option("print_level", 0)  # Silent

        # Solve the problem
        optvars_sol, info = nlp.solve(optvars_guess)

        if info["status"] != 0:
            status_msg = info["status_msg"]
            raise ValueError(
                f"Error occurred with the solver from 'CyIPOPT' library at "
                f" time={t_curr}. Status message: {status_msg}"
            )

        return jnp.array(optvars_sol)

    @property
    @abstractmethod
    def num_optvars(self) -> int:
        """
        Number of optimization variables in the NLP.
        """
        pass

    @property
    @abstractmethod
    def num_constraints(self) -> int:
        """
        Number of constraints in the NLP.
        """
        pass

    @property
    @abstractmethod
    def bounds_optvars(self) -> Tuple[Array, Array]:
        """
        Tuple of lower and upper bounds of the optimization variables in the
        NLP: (lower_bounds, upper_bounds)
        """
        pass

    @property
    @abstractmethod
    def bounds_constraints(self) -> Tuple[Array, Array]:
        """
        Tuple of lower and upper bounds of the constraints in the NLP:
        (lower_bounds_constraints, upper_bounds_constraints)
        """
        pass
