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
PID autotoning (without a measurement filter) with constraints in the frequency domain.

See the following references for more details:

[1] Hast, M., Åström, K.J., Bernhardsson, B. and Boyd, S., 2013, July.
PID design by convex-concave optimization. In 2013 European Control Conference (ECC)
(pp. 4460-4465). IEEE.

[2] Soltesz, K., Grimholt, C. and Skogestad, S., 2017. Simultaneous design of
proportional–integral–derivative controller and measurement filter by optimisation.
IET Control Theory & Applications, 11(3), pp.341-348.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import collimator
from collimator.lazy_loader import LazyLoader, LazyModuleAccessor
from collimator.library import (
    PID,
    Abs,
    Adder,
    Integrator,
    LTISystem,
    Step,
)
from collimator.library.state_estimators.utils import linearize_plant
from collimator.simulation import SimulatorOptions

scipy = LazyLoader("scipy", globals(), "scipy")
NonlinearConstraint = LazyModuleAccessor(scipy, "optimize.NonlinearConstraint")
minimize = LazyModuleAccessor(scipy, "optimize.minimize")

ct = LazyLoader("ct", globals(), "control")
plt = LazyLoader("plt", globals(), "matplotlib.pyplot")

cyipopt = LazyLoader(
    "cyipopt",
    globals(),
    "cyipopt",
    error_message="cyipopt is not installed.",
)

nlopt = LazyLoader(
    "nlopt",
    globals(),
    "nlopt",
    error_message="nlopt is not installed.",
)

SCIPY_METHODS = {
    "scipy-slsqp": "SLSQP",
    "scipy-cobyla": "COBYLA",
    "scipy-trust-constr": "trust-constr",
}

NLOPT_METHODS_LOCAL = {
    "nlopt-slsqp": lambda: nlopt.LD_SLSQP,
    "nlopt-cobyla": lambda: nlopt.LN_COBYLA,
    "nlopt-ld_mma": lambda: nlopt.LD_MMA,
}

NLOPT_METHODS_GLOBAL = {
    "nlopt-isres": lambda: nlopt.GN_ISRES,
    "nlopt-ags": lambda: nlopt.GN_AGS,
    "nlopt-direct": lambda: nlopt.GN_ORIG_DIRECT,
}

NLOPT_METHODS = {**NLOPT_METHODS_LOCAL, **NLOPT_METHODS_GLOBAL}


@dataclass
class OptResults:
    x: np.ndarray | float
    fun: np.ndarray | float
    success: bool
    message: str


def make_closed_loop_pid_system(system, metric, kp=0.1, ki=0.1, kd=0.1, n=100):
    builder = collimator.DiagramBuilder()

    pid = builder.add(PID(kp=kp, ki=ki, kd=kd, n=n, initial_state=0.0, name="pid"))
    plant = builder.add(system)

    err = builder.add(Adder(2, operators="+-", name="err"))
    ref = builder.add(Step(0.0, 1.0, 0.0, name="ref"))

    builder.connect(ref.output_ports[0], err.input_ports[0])
    builder.connect(plant.output_ports[0], err.input_ports[1])

    builder.connect(err.output_ports[0], pid.input_ports[0])
    builder.connect(pid.output_ports[0], plant.input_ports[0])

    integrator = builder.add(Integrator(initial_state=0.0, name="integrator"))
    if metric == "IAE":
        abs = builder.add(Abs(name="abs"))
        builder.connect(err.output_ports[0], abs.input_ports[0])
        builder.connect(abs.output_ports[0], integrator.input_ports[0])
    elif metric == "IE":
        builder.connect(err.output_ports[0], integrator.input_ports[0])
    else:
        raise ValueError("Invalid metric")

    builder.export_output(plant.output_ports[0], name="plant_output")
    diagram = builder.build(name="closed_loop_system")
    return pid, integrator, diagram


class AutoTuner:
    """
    PID autotuning (without a measurement filter) with constraints in the frequency
    domain.

    Supports only SISO systems.

    Supports only continuous-time plants (TODO: extend to discrete-time systems)

    Parameters:
        plant: LeafSystem or a Diagram.
            If plant is not an LTISystem, operating points x_op and u_op must be
            provided for linearization.
        n: int, optional
            Filter coefficient for the continuous-time PID controller
        sim_time: float, optional
            Simulation time for computation of the error metric
        metric: str, optional
            Error metric to be minimized. Options are "IAE" and "IE"
                "IAE": Integral of the absolute error
                "IE": Integral of the error
        x_op: np.ndarray, optional
            Operating point of state vector for linearization
        u_op: np.ndarray, optional
            Operating point of control vector for linearization
        pid_gains_0: list or Array, optional
            Initial guess for PID gains [kp, ki, kd]
        pid_gains_upper_bounds: list or Array, optional
            Upper bounds for PID gains [kp, ki, kd]. Lower bounds are set to 0
        Ms: float, optional
            Maximum sensitivity
        Mt: float, optional
            Maximum complementary sensitivity
        add_filter: bool, optional
            Add measurement filter (currently not implemented)
        method: str, optional
            The method for optimization. Available options are:
                - "scipy-slsqp"
                - "scipy-cobyla"
                - "scipy-trust-constr"
                - "ipopt"
                - "nlopt-slsqp"
                - "nlopt-cobyla"
                - "nlopt-ld_mma"
                - "nlopt-isres"
                - "nlopt-ags"
                - "nlopt-direct"

    Notes:

    The utilities `plot_freq_response`, `plot_time_response`, and
    `plot_freq_and_time_responses` can be used to visualize the frequency and time
    responses of the closed-loop system.

    Post initialization the `tune` method should be called to obtain the optimal PID
    gains. See `notebooks/opt_framework/pid_autotuning.ipynb` for an example.

    """

    def __init__(
        self,
        plant,
        n=100,
        sim_time=2.0,
        metric="IAE",
        x_op=None,
        u_op=None,
        pid_gains_0=[1.0, 10.0, 0.1],
        pid_gains_upper_bounds=None,
        Ms=100.0,
        Mt=100.0,
        add_filter=False,  # add measurement filter (currently not implemented)
        method="scipy-slsqp",
    ):
        if isinstance(plant, LTISystem):  # LTISystem includes TransferFunction
            linear_plant = plant
        else:
            if x_op is None or u_op is None:
                raise ValueError("Operating point x_op and u_op must be provided")

            _, linear_plant = linearize_plant(plant, x_op, u_op)

        if linear_plant.B.shape[1] != 1 or linear_plant.C.shape[0] != 1:
            raise ValueError("Plant must be SISO")

        self.A, self.B, self.C, self.D = (
            linear_plant.A,
            linear_plant.B,
            linear_plant.C,
            linear_plant.D,
        )

        self.pid, self.integrator, self.diagram = make_closed_loop_pid_system(
            linear_plant,
            metric,
            n=n,
        )

        self.lb = [0.0] * 3
        if pid_gains_upper_bounds is None:
            self.ub = [jnp.inf] * 3
        else:
            self.ub = pid_gains_upper_bounds

        self.base_context = self.diagram.create_context()

        self.n = n
        self.sim_time = sim_time
        self.metric = metric
        self.pid_gains_0 = pid_gains_0
        self.Ms = Ms
        self.Mt = Mt
        self.add_filter = add_filter
        self.method = method

        self.omega_grid = 10.0 ** jnp.linspace(-2, 2, 1000)
        # self.omega_grid = 10.0 ** jnp.linspace(-1, 2, 150)
        self.options = SimulatorOptions(
            enable_autodiff=True,
            max_major_step_length=0.01,  # rtol=1e-08, atol=1e-10
        )

        self.circle_constraint_vectorized = jax.vmap(
            self.circle_constraint_, in_axes=(None, None, None, 0, None, None)
        )  # Deprecated

        self.Ps_vectorized = jax.vmap(self.Ps, in_axes=0)
        self.Cs_vectorized = jax.vmap(self.Cs, in_axes=(None, None, None, 0))
        self.vec_absolute = jax.vmap(jnp.absolute)

    @partial(jax.jit, static_argnums=(0,))
    def objective(self, pid_params):
        kp, ki, kd = pid_params
        C = jnp.array(
            [[(ki * self.n), ((kp * self.n + ki) - (kp + kd * self.n) * self.n)]]
        )
        D = jnp.array([[(kp + kd * self.n)]])
        pid_subcontext = self.base_context[self.pid.system_id].with_parameters(
            {"C": C, "D": D}
        )
        context = self.base_context.with_subcontext(self.pid.system_id, pid_subcontext)
        sol = collimator.simulate(
            self.diagram, context, (0.0, self.sim_time), options=self.options
        )
        return self.integrator.output_ports[0].eval(sol.context) / self.sim_time

    @partial(jax.jit, static_argnums=(0,))
    def Ps(self, s):
        P = (
            self.C @ jnp.linalg.inv(s * jnp.eye(self.A.shape[0]) - self.A) @ self.B
            + self.D
        )
        return P[0, 0]

    @partial(jax.jit, static_argnums=(0,))
    def Cs(self, kp, ki, kd, s):
        return kp + ki / s + kd * s

    @partial(jax.jit, static_argnums=(0,))
    def circle_constraint_(self, kp, ki, kd, omega, c, r):
        """Deprecated: this is needed for `self.constraints_` which is deprecated
        and replaced by `self.constraints`.
        """
        s = omega * 1.0j
        L = self.Ps(s) * self.Cs(kp, ki, kd, s)
        return jnp.absolute(L - c) - r

    @partial(jax.jit, static_argnums=(0,))
    def constraints_(self, pid_params):
        """Deprecated: replaced by `self.constraints`"""
        kp, ki, kd = pid_params
        Ms, Mt = self.Ms, self.Mt
        g_Ms = self.circle_constraint_vectorized(
            kp, ki, kd, self.omega_grid, -1.0, 1.0 / Ms
        )
        g_Mt = self.circle_constraint_vectorized(
            kp, ki, kd, self.omega_grid, -(Mt**2) / (Mt**2 - 1.0), Mt / (Mt**2 - 1.0)
        )
        return jnp.array([jnp.min(g_Ms), jnp.min(g_Mt)])

    @partial(jax.jit, static_argnums=(0,))
    def constraints(self, pid_params):
        kp, ki, kd = pid_params
        S_grid = 1.0 / (
            1.0
            + self.Ps_vectorized(self.omega_grid * 1.0j)
            * self.Cs_vectorized(kp, ki, kd, self.omega_grid * 1.0j)
        )
        T_grid = 1.0 - S_grid

        S_grid = self.vec_absolute(S_grid)
        T_grid = self.vec_absolute(T_grid)

        return jnp.array([self.Ms - jnp.max(S_grid), self.Mt - jnp.max(T_grid)])

    def tune(self):
        x0 = jnp.array(self.pid_gains_0)
        bounds = list(zip(self.lb, self.ub))

        obj = jax.jit(self.objective)
        cons = jax.jit(self.constraints)

        obj_grad = jax.grad(self.objective)
        obj_hess = jax.jit(jax.hessian(self.objective))

        cons_jac = jax.jit(jax.jacfwd(self.constraints))

        print(f"Tuning with {self.method}")
        if self.method in SCIPY_METHODS:
            constraints_scipy = NonlinearConstraint(cons, 0.0, jnp.inf, jac=cons_jac)

            res = minimize(
                obj,
                x0,
                jac=obj_grad,
                method=SCIPY_METHODS[self.method],
                bounds=bounds,
                constraints=constraints_scipy,
                options={"maxiter": 100},
            )

        elif self.method == "ipopt":
            cons_hess = jax.hessian(self.constraints)
            cons_hess_vp = jax.jit(
                lambda x, v: jnp.sum(
                    # pylint: disable-next=not-callable
                    jnp.multiply(v[:, jnp.newaxis, jnp.newaxis], cons_hess(x)),
                    axis=0,
                )
            )

            constraints_ipopt = [
                {"type": "ineq", "fun": cons, "jac": cons_jac, "hess": cons_hess_vp}
            ]

            res = cyipopt.minimize_ipopt(
                obj,
                x0=x0,
                jac=obj_grad,
                hess=obj_hess,
                constraints=constraints_ipopt,
                bounds=bounds,
                options={
                    "max_iter": 500,
                    "disp": 5,
                },
            )

        elif self.method in NLOPT_METHODS:
            if self.method in NLOPT_METHODS_GLOBAL and any(
                ub == jnp.inf for ub in self.ub
            ):
                raise ValueError(
                    f"Method {self.method} requires finite upper bounds for all "
                    "parameters. Please specify `pid_gains_upper_bounds`."
                )

            # Define the objective function for nlopt
            def nlopt_obj(x, grad):
                if grad.size > 0:
                    grad[:] = obj_grad(jnp.array(x))
                # pylint: disable-next=not-callable
                return float(obj(jnp.array(x)))

            # Define the objective function for nlopt
            def nlopt_cons(result, x, grad):
                if grad.size > 0:
                    # pylint: disable-next=not-callable
                    grad[:, :] = -cons_jac(jnp.array(x))
                # pylint: disable-next=not-callable
                result[:] = -cons(jnp.array(x))

            # Initialize nlopt optimizer
            method = NLOPT_METHODS[self.method]()
            opt = nlopt.opt(method, len(x0))

            # Set the objective function
            opt.set_min_objective(nlopt_obj)

            # Set the constraints
            opt.add_inequality_mconstraint(nlopt_cons, [1e-6, 1e-06])

            # Set the bounds
            lower_bounds, upper_bounds = zip(*bounds)
            opt.set_lower_bounds(lower_bounds)
            opt.set_upper_bounds(upper_bounds)

            # Set stopping criteria
            opt.set_maxeval(500)
            opt.set_ftol_rel(1e-5)
            opt.set_xtol_rel(1e-6)
            opt.set_maxtime(30.0)

            # Run the optimization
            x_opt = opt.optimize(x0)
            print(f"{x_opt=}")
            minf = opt.last_optimum_value()

            nlopt_success_codes = {
                nlopt.SUCCESS: "SUCCESS",
                nlopt.STOPVAL_REACHED: "STOPVAL_REACHED",
                nlopt.FTOL_REACHED: "FTOL_REACHED",
                nlopt.XTOL_REACHED: "XTOL_REACHED",
                nlopt.MAXEVAL_REACHED: "MAXEVAL_REACHED",
                nlopt.MAXTIME_REACHED: "MAXTIME_REACHED",
            }

            nlopt_error_codes = {
                nlopt.FAILURE: "FAILURE",
                nlopt.INVALID_ARGS: "INVALID_ARGS",
                nlopt.OUT_OF_MEMORY: "OUT_OF_MEMORY",
                nlopt.ROUNDOFF_LIMITED: "ROUNDOFF_LIMITED",
                nlopt.FORCED_STOP: "FORCED_STOP",
            }

            nlopt_status_codes = {**nlopt_success_codes, **nlopt_error_codes}

            res = OptResults(
                x=x_opt,
                fun=minf,
                success=opt.last_optimize_result() in nlopt_success_codes,
                message=nlopt_status_codes[opt.last_optimize_result()],
            )

        else:
            raise ValueError("Invalid method")
        return res.x, res

    def plot_freq_response(
        self, pid_params, plant_tf_num, plant_tf_den, Ms=None, Mt=None
    ):
        if Ms is None:
            Ms = self.Ms

        if Mt is None:
            Mt = self.Mt

        kp, ki, kd = pid_params
        Cs = ct.TransferFunction([kd, kp, ki], [1, 0], name="PID")
        Ps = ct.TransferFunction(plant_tf_num, plant_tf_den, name="Plant")

        # Plot Gang of Four transfer functions
        ct.gangof4_plot(Ps, Cs, omega=self.omega_grid)

        fig1 = plt.gcf()
        axs = fig1.get_axes()

        axs[3].set_title(r"$T = \dfrac{PC}{1+PC}$")
        axs[1].set_title(r"$PS = \dfrac{P}{1+PC}$")
        axs[2].set_title(r"$CS = \dfrac{C}{1+PC}$")
        axs[0].set_title(r"$S = \dfrac{1}{1+PC}$")

        if Ms is not None:
            axs[0].hlines(
                Ms,
                self.omega_grid.min(),
                self.omega_grid.max(),
                colors="r",
                linestyles="--",
            )

        if Mt is not None:
            axs[3].hlines(
                Mt,
                self.omega_grid.min(),
                self.omega_grid.max(),
                colors="b",
                linestyles="--",
            )

        # Set x-axis labels for the bottom plots
        axs[2].set_xlabel("Frequency (rad/sec)")
        axs[3].set_xlabel("Frequency (rad/sec)")

        fig1.tight_layout()

        # Plot Nyquist plot
        fig2 = plt.figure()
        ct.nyquist_plot(
            Ps * Cs, omega=self.omega_grid, warn_nyquist=False, warn_encirclements=False
        )
        axs = fig2.get_axes()
        ax = axs[0]

        def gen_circle_points(c, r):
            t = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 100)
            return jnp.array([c + r * jnp.cos(t), r * jnp.sin(t)])

        if Ms is not None:
            c1, r1 = -1.0, 1.0 / Ms
            xc1, yc1 = gen_circle_points(c1, r1)
            ax.plot(xc1, yc1, "r--")

        if Mt is not None:
            c2, r2 = -(Mt**2) / (Mt**2 - 1.0), Mt / (Mt**2 - 1.0)
            xc2, yc2 = gen_circle_points(c2, r2)
            ax.plot(xc2, yc2, "b--")

        ax.set_title("Nyquist Plot")
        fig2.tight_layout()

        return fig1, fig2

    def plot_time_response(self, pid_params):
        kp, ki, kd = pid_params
        C = jnp.array(
            [[(ki * self.n), ((kp * self.n + ki) - (kp + kd * self.n) * self.n)]]
        )
        D = jnp.array([[(kp + kd * self.n)]])
        pid_subcontext = self.base_context[self.pid.system_id].with_parameters(
            {"C": C, "D": D}
        )
        context = self.base_context.with_subcontext(self.pid.system_id, pid_subcontext)

        recorded_signals = {
            "objective": self.diagram["integrator"].output_ports[0],
            "ref": self.diagram["ref"].output_ports[0],
            "plant": self.diagram.output_ports[0],
            "pid": self.diagram["pid"].output_ports[0],
        }

        sol = collimator.simulate(
            self.diagram,
            context,
            (0.0, self.sim_time),
            recorded_signals=recorded_signals,
        )

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(sol.time, sol.outputs["plant"], label=r"plant: $y$")
        ax1.plot(sol.time, sol.outputs["ref"], label=r"reference: $y_r$")
        ax2.plot(sol.time, sol.outputs["objective"], label=f"objective: {self.metric}")
        ax3.plot(sol.time, sol.outputs["pid"], label=r"pid-control: $u$")

        ax3.set_xlabel("Time (s)")
        for ax in (ax1, ax2, ax3):
            ax.legend()

        fig.suptitle("Time Response")
        fig.tight_layout()

        print(
            f"objective = "
            f"{self.integrator.output_ports[0].eval(sol.context)/self.sim_time}"
        )
        return fig

    def plot_freq_and_time_responses(
        self, pid_params, plant_tf_num, plant_tf_den, Ms=None, Mt=None
    ):
        fig1, fig2 = self.plot_freq_response(
            pid_params, plant_tf_num, plant_tf_den, Ms, Mt
        )
        fig3 = self.plot_time_response(pid_params)
        return fig1, fig2, fig3
