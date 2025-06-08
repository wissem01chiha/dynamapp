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

import warnings
from typing import Any, Optional

import jax.numpy as jnp

import collimator
from collimator.dashboard.serialization.ui_types import NodePathType
from collimator.framework.context import ContextBase
from collimator.framework.diagram import Diagram
from collimator.framework.port import OutputPort
from collimator.framework.system_base import SystemBase
from collimator.library import PID, Adder, Integrator, PIDDiscrete, Power, SourceBlock
from collimator.library.utils import extract_columns, read_csv
from collimator.logging import logger
from collimator.optimization.framework.base.optimizable import (
    DesignParameter,
    StochasticParameter,
)

from .framework import (
    IPOPT,
    CompositeTransform,
    DistributionConfig,
    Evosax,
    LogitTransform,
    LogTransform,
    NLopt,
    NormalizeTransform,
    Optax,
    OptaxWithStochasticVars,
    Optimizable,
    OptimizableWithStochasticVars,
    Scipy,
)

UI_DEFAULTS = {
    "learning_rate": 0.003,
    "num_epochs": 1000,
    "clip_range": None,
    "batch_size": 10,
    "num_batches": 10,
    "pop_size": 50,
    "num_generations": 50,
}

UI_NAME_TO_METHOD_OPTAX = {
    "adam": "adam",
    "rmsprop": "rmsprop",
    "stochastic_gradient_descent": "sgd",
}

UI_NAME_TO_METHOD_SCIPY = {
    # "bfgs": "BFGS",
    "l_bfgs_b": "L-BFGS-B",
    "nelder_mead": "Nelder-Mead",
    # "conjugate_gradient": "CG",
    "sequential_least_squares": "SLSQP",
}

UI_NAME_TO_METHOD_EVOSAX = {
    "particle_swarm_optimization": "PSO",
    "simulated_annealing": "SimAnneal",
    "covariance_matrix_adaptation_evolution_strategy": "CMA_ES",
    # "genetic_algorithm": "SimpleGA",
}

UI_NAME_TO_METHOD_NLOPT = {
    "method_of_moving_asymptotes": "mma",
    "dividing_rectangles": "direct",
}

DOESNT_SUPPORT_BOUNDS_NATIVELY = list(UI_NAME_TO_METHOD_OPTAX.keys())
SUPPORTS_CONSTRAINTS = list(UI_NAME_TO_METHOD_NLOPT.keys()) + [
    "sequential_least_squares"
]


class FloatWithSystem(float):
    """A subclass of float that may contain a reference to a system.

    As such, this behaves just like a normal float unless you want to get the
    associated system metadata.
    """

    def __init__(self, value, system: SystemBase = None):
        super().__init__()
        self.system = system

    def __new__(cls, value, system: SystemBase = None):
        obj = super().__new__(cls, value)
        obj.system = system
        return obj


class InterpBlock(SourceBlock):
    def __init__(self, time, data, **kwargs):
        self.time = time
        self.data = data
        super().__init__(lambda t: jnp.interp(t, self.time, self.data), **kwargs)


def _make_parameter_estimation_diagram(
    subdiagram,
    data_file,
    time_column,
    input_port_names_to_column_names,
    output_port_names_to_column_names,
    constraint_port_names,
):
    df = read_csv(data_file, header_as_first_row=True)
    time = extract_columns(df, time_column)

    # Note: `get_input_port` returns a tuple (port,i) but `get_output_port` returns port
    input_port_indices_to_column_data = {
        subdiagram.get_input_port(input_port_name)[0].index: extract_columns(
            df, column_name
        )
        for input_port_name, column_name in input_port_names_to_column_names.items()
    }

    output_port_indices_to_column_data = {
        subdiagram.get_output_port(output_port_name).index: extract_columns(
            df, column_name
        )
        for output_port_name, column_name in output_port_names_to_column_names.items()
    }

    constraint_port_indices = (
        [
            subdiagram.get_output_port(port_name).index
            for port_name in constraint_port_names
        ]
        if constraint_port_names
        else []
    )

    builder = collimator.DiagramBuilder()
    subdiagram = builder.add(subdiagram)

    # Wire input data ports to subdiagram
    for input_port_index, input_data in input_port_indices_to_column_data.items():
        source = builder.add(InterpBlock(time, input_data))
        builder.connect(
            source.output_ports[0], subdiagram.input_ports[input_port_index]
        )

    square_error_integrals = []

    # Wire objective (output) ports to the negative terminal of error ports
    # then square, and integrate
    for output_port_index, output_data in output_port_indices_to_column_data.items():
        error = builder.add(Adder(2, operators="+-"))
        reference = builder.add(InterpBlock(time, output_data))
        builder.connect(reference.output_ports[0], error.input_ports[0])
        builder.connect(
            subdiagram.output_ports[output_port_index], error.input_ports[1]
        )

        square_error = builder.add(Power(2.0))
        square_error_integral = builder.add(Integrator(initial_state=0.0))

        builder.connect(error.output_ports[0], square_error.input_ports[0])
        builder.connect(
            square_error.output_ports[0], square_error_integral.input_ports[0]
        )

        square_error_integrals.append(square_error_integral)

    num_objectives = len(square_error_integrals)
    if num_objectives > 1:
        adder = builder.add(Adder(num_objectives))
        for idx, square_error_integral in enumerate(square_error_integrals):
            builder.connect(
                square_error_integral.output_ports[0], adder.input_ports[idx]
            )
        objective_port = adder.output_ports[0]
    else:
        objective_port = square_error_integrals[0].output_ports[0]

    constraint_ports = [
        subdiagram.output_ports[index] for index in constraint_port_indices
    ]

    return (
        builder.build(parameters=subdiagram.parameters),
        objective_port,
        constraint_ports,
    )


def _process_pid_blocks(
    diagram, pid_blocks: list[SystemBase], pid_gains_init=None
) -> tuple[dict[str, PIDDiscrete], dict[str, PID], dict[str, float]]:
    optimizable_parameters = {}
    discrete_pid_blocks = {}
    continuous_pid_blocks = {}

    if pid_gains_init is None:
        # Q: should we use the existing values of kp,ki,kd if they are non-0? -- @jp
        kp_init, ki_init, kd_init = 0.1, 0.1, 0.1
    else:
        kp_init, ki_init, kd_init = pid_gains_init

    for block in pid_blocks:
        if not isinstance(block, (PID, PIDDiscrete)):
            raise ValueError(
                f"Block {block.name_path_str} is not of type PID or PIDDiscrete."
                " Only these types are supported for PID optimization."
            )

        block_sys_id = block.system_id

        # NOTE: Double underscore is not allowed in UI thus guarantees uniqueness
        # in this context. That is, it is impossible for a model parameter to
        # have the name __kp_x (when coming from UI).
        params = {
            f"__kp_{block_sys_id}": kp_init,
            f"__ki_{block_sys_id}": ki_init,
            f"__kd_{block_sys_id}": kd_init,
        }
        if isinstance(block, PIDDiscrete):
            discrete_pid_blocks[block_sys_id] = block
        elif isinstance(block, PID):
            # The value of n will be fetched from the block instance
            continuous_pid_blocks[block_sys_id] = block
        optimizable_parameters.update(params)

    return (
        discrete_pid_blocks,
        continuous_pid_blocks,
        optimizable_parameters,
    )


def _get_unbounded_transform(params):
    """
    A transformation to map parameters to an unbounded space. This is useful for
    optimization algorithms that do not support bounded parameters natively.

    Note: For the PID tuning job, presently a Log transformation is used, which only
    enforces positivity. However, if the upper bound also needed to be enforced, this
    function may be used. We need more testing to check which approach is better.
    """
    params_min = {p.param_name: p.min for p in params}
    params_max = {p.param_name: p.max for p in params}
    normalize = NormalizeTransform(params_min, params_max)
    logit = LogitTransform()
    unbounded_transform = CompositeTransform([normalize, logit])
    return unbounded_transform


def _get_bounds(parameters):
    bounds = {
        p.param_name: (
            p.min if p.min is not None else -jnp.inf,
            p.max if p.max is not None else jnp.inf,
        )
        for p in parameters
    }
    return bounds


def _resolve_block_parameters(params: dict, blocks: dict[int, SystemBase]):
    rp = params.copy()

    for block_sys_id, blk in blocks.items():
        path = blk.name_path_str
        kp = rp.pop(f"__kp_{block_sys_id}")
        ki = rp.pop(f"__ki_{block_sys_id}")
        kd = rp.pop(f"__kd_{block_sys_id}")
        # NOTE: using uppercase Kp, Ki, Kd to match the parameter names in JSON
        # See also OptimalParameterJson
        rp[f"{path}.Kp"] = FloatWithSystem(kp, system=blk)
        rp[f"{path}.Ki"] = FloatWithSystem(ki, system=blk)
        rp[f"{path}.Kd"] = FloatWithSystem(kd, system=blk)

    return rp


class OptimizableModel(Optimizable):
    def __init__(
        self,
        job_type: str,
        diagram: Diagram,
        objective_port: OutputPort,
        constraint_ports: list[OutputPort],
        design_parameters: list[DesignParameter],
        sim_t_span: tuple[float, float] = (0.0, 1.0),
        data_file=None,
        time_column=None,
        input_port_names_to_column_names=None,
        output_port_names_to_column_names=None,
        constraint_port_names=None,
        pid_blocks=None,
        pid_gains_init=None,
        algorithm=None,
    ):
        self.job_type = job_type
        transformation = None

        if job_type == "design":
            opt_diagram = diagram
            opt_design_parameters = design_parameters
            opt_constraint_ports = constraint_ports
            self.objective_port = objective_port
            self.prepare_context = self.prepare_context_design_estimation

            bounds = _get_bounds(opt_design_parameters)

        elif job_type == "estimation":
            opt_diagram, self.objective_port, opt_constraint_ports = (
                _make_parameter_estimation_diagram(
                    diagram,
                    data_file,
                    time_column,
                    input_port_names_to_column_names,
                    output_port_names_to_column_names,
                    constraint_port_names,
                )
            )
            opt_design_parameters = design_parameters
            self.prepare_context = self.prepare_context_design_estimation

            bounds = _get_bounds(opt_design_parameters)

        elif job_type == "pid":
            (
                self.discrete_pids,
                self.continuous_pids,
                pid_parameters,
            ) = _process_pid_blocks(diagram, pid_blocks, pid_gains_init)

            opt_diagram = diagram
            opt_design_parameters = []

            pid_gains_min = 0.0
            pid_gains_max = 1e03  # FIXME: get from user? 1000 seems reasonable

            for k, v in pid_parameters.items():
                opt_design_parameters.append(
                    DesignParameter(
                        param_name=k,
                        initial=v,
                        min=pid_gains_min,
                        max=pid_gains_max,
                    )
                )
            self.objective_port = objective_port
            self.prepare_context = self.prepare_context_pid
            opt_constraint_ports = constraint_ports

            if algorithm in DOESNT_SUPPORT_BOUNDS_NATIVELY:
                bounds = None
                # Use either unbounded or log transform
                # needs testing to determine which transform works better.
                # log tranformation would disregard pid_gains_max, which is not bad
                use_unbounded_transform = False
                if use_unbounded_transform:
                    transformation = _get_unbounded_transform(opt_design_parameters)
                else:
                    transformation = LogTransform()
            else:
                bounds = _get_bounds(opt_design_parameters)

        else:
            raise ValueError(f"Unknown job type: {job_type}")

        params_0 = {p.param_name: p.initial for p in opt_design_parameters}
        self.params_keys = [p.param_name for p in opt_design_parameters]
        self.constraint_ports = opt_constraint_ports if opt_constraint_ports else None

        super().__init__(
            opt_diagram,
            opt_diagram.create_context(),
            sim_t_span,
            params_0,
            bounds=bounds,
            transformation=transformation,
            init_min_max=None,  # if exposed in the UI, construct from design_parameters
        )

    def optimizable_params(self, context):
        params = {k: context.parameters[k] for k in self.params_keys}
        return params

    def objective_from_context(self, context):
        return self.objective_port.eval(context)

    def constraints_from_context(self, context):
        if not self.constraint_ports:
            return None
        return jnp.array([port.eval(context) for port in self.constraint_ports])

    def prepare_context(self, context, params):
        # This is a dummy implementation to satisfy the abstract method requirement
        # Reassignment to correct methods depending on job-type happens at
        # initialization
        return context

    def prepare_context_design_estimation(self, context, params):
        return context.with_parameters(params)

    def prepare_context_pid(self, context, params):
        for block_sys_id, blk in self.continuous_pids.items():
            kp = params[f"__kp_{block_sys_id}"]
            ki = params[f"__ki_{block_sys_id}"]
            kd = params[f"__kd_{block_sys_id}"]
            # We don't want to optimize over n:
            n = blk.parameters["n"].value
            C = jnp.array([[(ki * n), ((kp * n + ki) - (kp + kd * n) * n)]])
            D = jnp.array([[(kp + kd * n)]])
            subcontext = context[block_sys_id].with_parameters({"C": C, "D": D})
            context = context.with_subcontext(block_sys_id, subcontext)

        for block_sys_id in self.discrete_pids:
            kp = params[f"__kp_{block_sys_id}"]
            ki = params[f"__ki_{block_sys_id}"]
            kd = params[f"__kd_{block_sys_id}"]
            subcontext = context[block_sys_id].with_parameters(
                {"kp": kp, "ki": ki, "kd": kd}
            )
            context = context.with_subcontext(block_sys_id, subcontext)

        return context

    def resolve_block_parameters(self, params: dict) -> dict[str, FloatWithSystem]:
        if self.job_type != "pid":
            return {k: FloatWithSystem(v) for k, v in params.items()}
        return _resolve_block_parameters(
            params, {**self.continuous_pids, **self.discrete_pids}
        )


class OptimizableModelWithStochasticVars(OptimizableWithStochasticVars):
    def __init__(
        self,
        job_type: str,
        diagram: Diagram,
        objective_port: OutputPort,
        constraint_ports: list[OutputPort],
        design_parameters: list[DesignParameter],
        stochastic_parameters: list[StochasticParameter],
        sim_t_span: tuple[float, float] = (0.0, 1.0),
        pid_blocks=None,
        pid_gains_init=None,
        algorithm=None,
    ):
        self.job_type = job_type
        transformation = None

        if job_type == "design":
            opt_diagram = diagram
            opt_design_parameters = design_parameters
            self.prepare_context = self.prepare_context_design

            bounds = _get_bounds(opt_design_parameters)

        elif job_type == "estimation":
            raise ValueError("Parameter estimation not supported for stochastic vars")

        elif job_type == "pid":
            (
                self.discrete_pids,
                self.continuous_pids,
                pid_parameters,
            ) = _process_pid_blocks(diagram, pid_blocks, pid_gains_init)

            opt_diagram = diagram
            opt_design_parameters = []

            pid_gains_min = 0.0
            pid_gains_max = 1e03  # FIXME: get from user

            for k, v in pid_parameters.items():
                opt_design_parameters.append(
                    DesignParameter(
                        param_name=k,
                        initial=v,
                        min=pid_gains_min,
                        max=pid_gains_max,
                    )
                )

            self.prepare_context = self.prepare_context_pid

            if algorithm in DOESNT_SUPPORT_BOUNDS_NATIVELY:
                bounds = None
                # transformation = _get_unbounded_transform(opt_design_parameters)
                transformation = LogTransform()
            else:
                bounds = _get_bounds(opt_design_parameters)

        else:
            raise ValueError(f"Unknown job type: {job_type}")

        self.objective_port = objective_port
        params_0 = {p.param_name: p.initial for p in opt_design_parameters}

        self.params_keys = [p.param_name for p in opt_design_parameters]
        self.vars_keys = [p.param_name for p in stochastic_parameters]

        opt_context = opt_diagram.create_context()

        distribution_names = [p.distribution.name for p in stochastic_parameters]
        distribution_shapes = [opt_context.parameters[k].shape for k in self.vars_keys]
        distribution_configs = [p.distribution.options for p in stochastic_parameters]
        distribution_config_vars = DistributionConfig(
            names=self.vars_keys,
            shapes=distribution_shapes,
            distributions=distribution_names,
            distributions_configs=distribution_configs,
        )

        self.constraint_ports = constraint_ports if constraint_ports else None

        super().__init__(
            opt_diagram,
            opt_context,
            sim_t_span,
            params_0,
            vars_0=None,
            distribution_config_vars=distribution_config_vars,
            bounds=bounds,
            transformation=transformation,
            seed=None,
        )

    def optimizable_params(self, context: ContextBase):
        params = {k: context.parameters[k] for k in self.params_keys}
        return params

    def stochastic_vars(self, context: ContextBase):
        params = {k: context.parameters[k] for k in self.vars_keys}
        return params

    def objective_from_context(self, context):
        return self.objective_port.eval(context)

    def constraints_from_context(self, context):
        if not self.constraint_ports:
            return None
        return jnp.array([port.eval(context) for port in self.constraint_ports])

    def prepare_context(self, context, params, vars):
        # This is a dummy implementation to satisfy the abstract method requirement
        # Reassignment to correct methods depending on job-type happens at
        # initialization
        return context

    def prepare_context_design(self, context, params, vars):
        params_and_vars = params.copy()
        params_and_vars.update(vars)
        return context.with_parameters(params_and_vars)

    def prepare_context_pid(self, context, params, vars):
        for block_sys_id, blk in self.continuous_pids.items():
            kp = params[f"__kp_{block_sys_id}"]
            ki = params[f"__ki_{block_sys_id}"]
            kd = params[f"__kd_{block_sys_id}"]
            # We don't want to optimize over n:
            n = blk.parameters["n"].value
            C = jnp.array([[(ki * n), ((kp * n + ki) - (kp + kd * n) * n)]])
            D = jnp.array([[(kp + kd * n)]])
            subcontext = context[block_sys_id].with_parameters({"C": C, "D": D})
            context = context.with_subcontext(block_sys_id, subcontext)

        for block_sys_id in self.discrete_pids:
            kp = params[f"__kp_{block_sys_id}"]
            ki = params[f"__ki_{block_sys_id}"]
            kd = params[f"__kd_{block_sys_id}"]
            subcontext = context[block_sys_id].with_parameters(
                {"kp": kp, "ki": ki, "kd": kd}
            )
            context = context.with_subcontext(block_sys_id, subcontext)

        return context.with_parameters(vars)

    def resolve_block_parameters(self, params: dict) -> dict[str, FloatWithSystem]:
        if self.job_type != "pid":
            return {k: FloatWithSystem(v) for k, v in params.items()}
        return _resolve_block_parameters(
            params, {**self.continuous_pids, **self.discrete_pids}
        )


def _options_pop(options, key: str):
    if key in options:
        return options.pop(key)
    return UI_DEFAULTS.get(key)


def jobs_router(
    job_type: str,
    diagram: Diagram,
    algorithm: str,
    options: dict[str, Any],
    design_parameters: Optional[list[DesignParameter]] = None,
    sim_t_span: Optional[tuple[float, float]] = None,
    objective_port: Optional[OutputPort] = None,
    constraint_ports: Optional[OutputPort] = None,
    stochastic_parameters: Optional[list[StochasticParameter]] = None,
    data_file: Optional[str] = None,
    time_column: Optional[str] = None,
    input_port_names_to_column_names: Optional[dict[str, str]] = None,
    output_port_names_to_column_names: Optional[dict[str, str]] = None,
    constraint_port_names: Optional[list[str]] = None,
    pid_blocks: Optional[list[NodePathType]] = None,
    pid_gains_init: Optional[tuple[float, float, float]] = None,
    print_every=100,
    metrics_writer=None,
) -> tuple[dict[str, FloatWithSystem], dict[str, Any]]:
    """
    Args:
        options: contains options for both the integration layer and the underlying
            optimization algorithm.  The options are algorithm-specific.

    The main router for UI jobs:
    Parameters:
        job_type: str
            Type of job to perform. One of "design", "estimation", "pid"

        diagram: Diagram
            The diagram to be optimized.

            For "design" and "pid", this is the top-level diagram. For these jobs,
            the `objective_port` and `constraint_ports` are explicitly selected.

            For "estimation", this is a submodel. For this job, `data_file`,
            `time_column`, `input_port_names_to_column_names`,
            `output_port_names_to_column_names`, and `constraint_port_names` are
            required. These will be used to a new diagram with `objective_port` and
            `constraint_ports`, so that the same methodology as for "design" and "pid"
            can be used.

        algorithm: str
            The optimization algorithm to use

        options: dict
            contains options for both the integration layer and the underlying
            optimization algorithm.  The options are algorithm-specific.

        design_parameters: list[DesignParameter]
            List of design parameters. For "pid" job, these will be autimatically
            generated and correspond the PID gains of the `pid_blocks`.

        sim_t_span: tuple[float, float]
            Simulation time span

        objective_port: OutputPort
            The output port to optimize. Required for "design" and "pid" jobs. Auto
            generated for "estimation" job.

        constraint_ports: list[OutputPort]
            List of constraint ports. Optional for "design" and "pid" jobs. Auto
            generated for "estimation" job from `constraint_ports_names`.

        stochastic_parameters: list[StochasticParameter]
            List of stochastic parameters

        data_file: str
            Path to the data file. For "estimation" job only.

        time_column: str
            Column name for the time data in the data file. For "estimation" job only.

        input_port_names_to_column_names: dict[str, str]
            For "estimation" job only. Mapping of submodel input port names to
            corresponding column names in the data file.

        output_port_names_to_column_names: dict[str, str]
            For "estimation" job only. Mapping of submodel output port names to
            corresponding column names in the data file.

        constraint_port_names: list[str]
            For "estimation" job only. List of submodel output port names to be used
            as constraints.

        pid_blocks: list[NodePathType]
            For "pid" job only. List of PID blocks to optimize.

        pid_gains_init: tuple[float, float, float]
            For "pid" jobs only. Initial gains for PID blocks. A single set is used
            for initialization of all the PID blocks in the model.

        print_every: int
            Print stats every `print_every` iterations. Applicable to Optax and Evosax
            algorithms. For Optax algorithms, this corresponds to epochs, while for
            Evosax algorithms, this corresponds to generations.

        metrics_writer: MetricsWriter (optional)
            If specified, a CSV file will be written with the optimization metrics,
            that depend on each algorithm. Not all algorithms support such outputs.
            The output is updated on-the-fly and can be used to visualize progress.
    """
    # TODO: Extend this feature for other jobs too
    if algorithm in DOESNT_SUPPORT_BOUNDS_NATIVELY and job_type == "pid":
        warnings.warn(
            f"Algorithm {algorithm} does not support bounds natively. Parameteric "
            "transformations will be used to make the parameteric space unbounded."
        )

    if algorithm not in SUPPORTS_CONSTRAINTS and constraint_ports:
        raise ValueError(f"Algorithm {algorithm} does not support constraints.")

    if not sim_t_span:
        sim_t_span = (0.0, 1.0)

    opt_model = OptimizableModel(
        job_type,
        diagram,
        objective_port,
        constraint_ports,
        design_parameters,
        sim_t_span=sim_t_span,
        data_file=data_file,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=constraint_port_names,
        time_column=time_column,
        pid_blocks=pid_blocks,
        pid_gains_init=pid_gains_init,
        algorithm=algorithm,
    )

    learning_rate = _options_pop(options, "learning_rate")
    num_epochs = _options_pop(options, "num_epochs")
    clip_range = _options_pop(options, "clip_range")
    pop_size = _options_pop(options, "pop_size")
    num_generations = _options_pop(options, "num_generations")
    batch_size = _options_pop(options, "batch_size")
    num_batches = _options_pop(options, "num_batches")

    if algorithm in UI_NAME_TO_METHOD_OPTAX:
        opt_method = UI_NAME_TO_METHOD_OPTAX[algorithm]
        logger.info('Using Optax optimizer with method "%s"', opt_method)

        if not stochastic_parameters:
            optim = Optax(
                optimizable=opt_model,
                opt_method=opt_method,
                learning_rate=learning_rate,
                opt_method_config=options,
                num_epochs=num_epochs,
                clip_range=clip_range,
                print_every=print_every,
                metrics_writer=metrics_writer,
            )

        else:
            opt_model = OptimizableModelWithStochasticVars(
                job_type,
                diagram,
                objective_port,
                constraint_ports,
                design_parameters,
                stochastic_parameters,
                sim_t_span=sim_t_span,
                pid_blocks=pid_blocks,
                pid_gains_init=pid_gains_init,
                algorithm=algorithm,
            )
            optim = OptaxWithStochasticVars(
                optimizable=opt_model,
                opt_method=opt_method,
                learning_rate=learning_rate,
                opt_method_config=options,
                batch_size=batch_size,
                num_batches=num_batches,
                num_epochs=num_epochs,
                clip_range=clip_range,
                print_every=print_every,
                metrics_writer=metrics_writer,
            )

    elif algorithm in UI_NAME_TO_METHOD_SCIPY:
        opt_method = UI_NAME_TO_METHOD_SCIPY[algorithm]
        logger.info('Using Scipy optimizer with method "%s"', opt_method)
        optim = Scipy(
            optimizable=opt_model,
            opt_method=opt_method,
            opt_method_config=options,
            metrics_writer=metrics_writer,
        )

    elif algorithm in UI_NAME_TO_METHOD_NLOPT:
        opt_method = UI_NAME_TO_METHOD_NLOPT[algorithm]
        logger.info('Using NLopt optimizer with method "%s"', opt_method)
        optim = NLopt(
            optimizable=opt_model,
            opt_method=opt_method,
            ftol_rel=1e-04,  # FIXME: defaults, in case we want to get these from UI
            ftol_abs=1e-06,
            xtol_rel=1e-04,
            xtol_abs=1e-06,
            cons_tol=1e-06,
            maxeval=500,
            maxtime=0,
        )

    elif algorithm == "ipopt":
        logger.info("Using IPOPT optimizer")
        optim = IPOPT(
            optimizable=opt_model,
            options={"disp": 0},  # FIXME: More options from UI?
        )

    elif algorithm in UI_NAME_TO_METHOD_EVOSAX:
        opt_method = UI_NAME_TO_METHOD_EVOSAX[algorithm]
        logger.info('Using Evosax optimizer with method "%s"', opt_method)
        optim = Evosax(
            optimizable=opt_model,
            opt_method=opt_method,
            opt_method_config=options,
            pop_size=pop_size,
            num_generations=num_generations,
            seed=None,
            print_every=print_every,
            metrics_writer=metrics_writer,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    optimal_parameters = optim.optimize()

    # Converts optimal parameters to FloatWithSystem objects (behave like regular
    # floats), and enrich them with block/system metadata.
    optimal_parameters = opt_model.resolve_block_parameters(optimal_parameters)

    return optimal_parameters, optim.metrics
