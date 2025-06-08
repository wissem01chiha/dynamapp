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
Standalone tests for optimization framework. These are separated from any UI or json
processing.
"""

import platform
import pytest
import numpy as np
import jax.numpy as jnp
from collimator import DiagramBuilder, Parameter
from collimator.library import Adder, Constant, Gain, Integrator, Power
from collimator.optimization import (
    IPOPT,
    DistributionConfig,
    Evosax,
    NLopt,
    Optax,
    OptaxWithStochasticVars,
    Optimizable,
    OptimizableWithStochasticVars,
    Scipy,
)
from collimator.optimization import (
    NormalizeTransform,
    LogitTransform,
    CompositeTransform,
)


def _make_diagram(m=1.0, v0=0.1, x0=1.0):
    model_parameters = {
        "c": Parameter(np.array(1.0)),
        "k": Parameter(np.array(1.0)),
    }

    builder = DiagramBuilder()
    k_x = builder.add(Gain(model_parameters["k"], name="k_x"))
    c_v = builder.add(Gain(model_parameters["c"], name="c_v"))
    adder = builder.add(Adder(2, operators="--", name="adder"))
    one_over_m = builder.add(Gain(1.0 / m, name="one_over_m"))
    v = builder.add(Integrator(v0, name="v"))
    x = builder.add(Integrator(x0, name="x"))

    builder.connect(k_x.output_ports[0], adder.input_ports[0])
    builder.connect(c_v.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], one_over_m.input_ports[0])
    builder.connect(one_over_m.output_ports[0], v.input_ports[0])
    builder.connect(v.output_ports[0], x.input_ports[0])

    builder.connect(v.output_ports[0], c_v.input_ports[0])
    builder.connect(x.output_ports[0], k_x.input_ports[0])

    ref_x = builder.add(Constant(0.0, name="ref_x"))
    ref_v = builder.add(Constant(0.0, name="ref_v"))

    err_v = builder.add(Adder(2, operators="+-", name="err_v"))
    err_x = builder.add(Adder(2, operators="+-", name="err_x"))
    sq_err_v = builder.add(Power(2.0, name="sq_err_v"))
    sq_err_x = builder.add(Power(2.0, name="sq_err_x"))
    cost_v = builder.add(Integrator(0.0, name="cost_v"))
    cost_x = builder.add(Integrator(0.0, name="cost_x"))
    objective = builder.add(Adder(2, operators="++", name="objective"))

    builder.connect(ref_v.output_ports[0], err_v.input_ports[0])
    builder.connect(v.output_ports[0], err_v.input_ports[1])
    builder.connect(ref_x.output_ports[0], err_x.input_ports[0])
    builder.connect(x.output_ports[0], err_x.input_ports[1])

    builder.connect(err_v.output_ports[0], sq_err_v.input_ports[0])
    builder.connect(err_x.output_ports[0], sq_err_x.input_ports[0])
    builder.connect(sq_err_v.output_ports[0], cost_v.input_ports[0])
    builder.connect(sq_err_x.output_ports[0], cost_x.input_ports[0])
    builder.connect(cost_v.output_ports[0], objective.input_ports[0])
    builder.connect(cost_x.output_ports[0], objective.input_ports[1])

    diagram = builder.build(parameters=model_parameters)

    return diagram


class OptimizableModel(Optimizable):
    def __init__(
        self,
        diagram,
        base_context,
        params_0,
        **kwargs,
    ):
        super().__init__(
            diagram=diagram,
            base_context=base_context,
            params_0=params_0,
            **kwargs,
        )

        self.params_keys = list(self.params_0.keys())

        self.system_for_objective = diagram["objective"]
        self.objective_port_index = 0

    def optimizable_params(self, context):
        params = {k: context.parameters[k] for k in self.params_keys}
        return params

    def objective_from_context(self, context):
        objective = self.system_for_objective.output_ports[
            self.objective_port_index
        ].eval(context)
        return objective

    def prepare_context(self, context, params):
        new_context = context.with_parameters(params)
        return new_context


class OptimizableModelWithConstraints(Optimizable):
    def __init__(
        self,
        diagram,
        base_context,
        params_0,
        **kwargs,
    ):
        self.params_keys = list(params_0.keys())

        self.system_for_objective = diagram["objective"]
        self.objective_port_index = 0

        super().__init__(
            diagram=diagram,
            base_context=base_context,
            params_0=params_0,
            **kwargs,
        )

    def optimizable_params(self, context):
        params = {k: context.parameters[k] for k in self.params_keys}
        return params

    def objective_from_context(self, context):
        objective = self.system_for_objective.output_ports[
            self.objective_port_index
        ].eval(context)
        return objective

    def constraints_from_context(self, context):
        constraint = (
            self.system_for_objective.output_ports[self.objective_port_index].eval(
                context
            )
            - 1.67
        )
        return jnp.array(constraint)

    def prepare_context(self, context, params):
        new_context = context.with_parameters(params)
        return new_context


class OptimizableModelStoachastic(OptimizableWithStochasticVars):
    def __init__(
        self,
        diagram,
        base_context,
        params_0=None,
        **kwargs,
    ):
        self.params_keys = list(params_0.keys())
        self.vars_keys = ["k"]

        self.system_for_objective = diagram["objective"]
        self.objective_port_index = 0

        super().__init__(
            diagram=diagram,
            base_context=base_context,
            params_0=params_0,
            **kwargs,
        )

    def optimizable_params(self, context):
        params = {k: context.parameters[k] for k in self.params_keys}
        return params

    def stochastic_vars(self, context):
        params = {k: context.parameters[k] for k in self.vars_keys}
        return params

    def objective_from_context(self, context):
        objective = self.system_for_objective.output_ports[
            self.objective_port_index
        ].eval(context)
        return objective

    def prepare_context(self, context, params, vars):
        params_and_vars = params.copy()
        params_and_vars.update(vars)
        new_context = context.with_parameters(params_and_vars)
        return new_context


@pytest.mark.slow
def test_optimization_unbounded():
    diagram = _make_diagram()
    base_context = diagram.create_context()
    params_0 = {"c": 0.5}
    sim_t_span = (0.0, 2.0)

    optimizable_model = OptimizableModel(
        diagram,
        base_context,
        params_0=params_0,
        sim_t_span=sim_t_span,
    )

    # Scipy with L-BFGS-B
    optim = Scipy(
        optimizable_model,
        "L-BFGS-B",
        opt_method_config={"maxiter": 20},
        use_autodiff_grad=True,
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.65, atol=0.1)

    # JAX's scipy minimize with BFGS
    optim = Scipy(optimizable_model, "BFGS", opt_method_config={}, use_jax_scipy=True)
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.65, atol=0.1)

    # Optax adam
    optim = Optax(
        optimizable_model, "adam", 0.005, {}, num_epochs=1000, print_every=100
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.65, atol=0.1)

    # Evosax Particle Swarm Optimization
    optim = Evosax(
        optimizable_model,
        "PSO",
        pop_size=50,
        num_generations=50,
        print_every=10,
        seed=42,
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.65, atol=0.1)

    # Evosax Simulated Annealing
    optim = Evosax(
        optimizable_model,
        "PSO",
        pop_size=50,
        num_generations=50,
        print_every=10,
        seed=42,
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.65, atol=0.1)


@pytest.mark.slow
def test_optimization_bounded():
    diagram = _make_diagram()
    base_context = diagram.create_context()
    params_0 = {"c": 0.5}
    sim_t_span = (0.0, 2.0)

    optimizable_model = OptimizableModel(
        diagram,
        base_context,
        params_0=params_0,
        sim_t_span=sim_t_span,
        bounds={"c": (0.0, 1.62)},
    )

    # Scipy with L-BFGS-B
    optim = Scipy(
        optimizable_model,
        "L-BFGS-B",
        opt_method_config={"maxiter": 20},
        use_autodiff_grad=True,
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.62, atol=0.01)

    # Evosax with PSO
    optim = Evosax(
        optimizable_model,
        "PSO",
        pop_size=50,
        num_generations=50,
        print_every=10,
        seed=42,
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.62, atol=0.01)

    # inf bounds with non-bound supporting algorithms should also work
    optimizable_model = OptimizableModel(
        diagram,
        base_context,
        params_0=params_0,
        sim_t_span=sim_t_span,
        bounds={"c": (-jnp.inf, jnp.inf)},
    )
    optim = Scipy(
        optimizable_model,
        "BFGS",
        opt_method_config={"maxiter": 20},
        use_autodiff_grad=True,
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.65, atol=0.1)


# if nlopt publishes a compatible version, reactivate this for all
@pytest.mark.skipif(
    platform.machine() == "arm64", reason="nlopt not available for arm64"
)
@pytest.mark.slow
def test_optimization_constrained():
    diagram = _make_diagram()
    base_context = diagram.create_context()
    params_0 = {"c": 0.5}
    sim_t_span = (0.0, 2.0)
    optimizable_model = OptimizableModelWithConstraints(
        diagram,
        base_context,
        params_0=params_0,
        sim_t_span=sim_t_span,
    )

    # Scipy SLSQP
    optim = Scipy(
        optimizable_model,
        "SLSQP",
        opt_method_config={"maxiter": 20},
        use_autodiff_grad=True,
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 0.36, atol=0.1)
    assert np.isclose(
        optimizable_model.constraints_flat(opt_param["c"]), 0.0, atol=1e-03
    )

    # NLopt SLSQP
    optim = NLopt(optimizable_model, "slsqp")
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 0.36, atol=0.1)
    assert np.isclose(
        optimizable_model.constraints_flat(opt_param["c"]), 0.0, atol=1e-03
    )


@pytest.mark.skip(reason="Hessian computation needs investigation")
@pytest.mark.slow
def test_optimization_constrainted_ipopt():
    diagram = _make_diagram()
    base_context = diagram.create_context()
    params_0 = {"c": 0.5}
    sim_t_span = (0.0, 2.0)
    optimizable_model = OptimizableModelWithConstraints(
        diagram,
        base_context,
        params_0=params_0,
        sim_t_span=sim_t_span,
    )

    optim = IPOPT(optimizable_model, options={"maxiter": 20, "disp": 0})
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 0.36, atol=0.1)
    assert np.isclose(
        optimizable_model.constraints_flat(opt_param["c"]), 0.0, atol=1e-03
    )


@pytest.mark.slow
def test_optimization_stochastic():
    diagram = _make_diagram()
    base_context = diagram.create_context()
    params_0 = {"c": 0.5}
    sim_t_span = (0.0, 2.0)

    names = ["k"]
    shapes = [()]
    distributions = ["uniform"]
    distributions_configs = [{"min": 0.5, "max": 1.5}]

    distribution_config_vars = DistributionConfig(
        names, shapes, distributions, distributions_configs
    )

    optimizable_model = OptimizableModelStoachastic(
        diagram,
        base_context,
        params_0=params_0,
        sim_t_span=sim_t_span,
        distribution_config_vars=distribution_config_vars,
    )

    # Optax SGD
    optim = OptaxWithStochasticVars(
        optimizable_model,
        "sgd",
        0.01,
        {},
        batch_size=10,
        num_batches=10,
        num_epochs=200,
        print_every=10,
    )
    opt_param = optim.optimize()
    assert np.isclose(opt_param["c"], 1.74, atol=0.2)


@pytest.mark.slow
def test_optimization_transforms():
    diagram = _make_diagram()
    base_context = diagram.create_context()
    params_0 = {"c": 0.5}
    sim_t_span = (0.0, 2.0)

    params_min = {"c": 0.0}
    params_max = {"c": 1.62}

    normalize = NormalizeTransform(params_min, params_max)
    logit = LogitTransform()
    composite = CompositeTransform([normalize, logit])

    optimizable_model = OptimizableModel(
        diagram,
        base_context,
        params_0=params_0,
        sim_t_span=sim_t_span,
        bounds=None,
        transformation=composite,
    )

    # Optax sgd
    optim = Optax(
        optimizable_model, "adam", 0.005, {}, num_epochs=10000, print_every=500
    )
    opt_param = optim.optimize()
    print(opt_param)
    assert np.isclose(opt_param["c"], 1.62, atol=0.02)
    assert opt_param["c"] >= params_min["c"]
    assert opt_param["c"] <= params_max["c"]


@pytest.mark.slow
def test_optimization_transforms_stochastic():
    diagram = _make_diagram()
    base_context = diagram.create_context()
    params_0 = {"c": 1.0}
    sim_t_span = (0.0, 2.0)

    params_min = {"c": 0.45}
    params_max = {"c": 1.15}

    normalize = NormalizeTransform(params_min, params_max)
    logit = LogitTransform()
    composite = CompositeTransform([normalize, logit])

    names = ["k"]
    shapes = [()]
    distributions = ["uniform"]
    distributions_configs = [{"min": 0.5, "max": 1.5}]

    distribution_config_vars = DistributionConfig(
        names, shapes, distributions, distributions_configs
    )

    optimizable_model = OptimizableModelStoachastic(
        diagram,
        base_context,
        params_0=params_0,
        sim_t_span=sim_t_span,
        bounds=None,
        transformation=composite,
        distribution_config_vars=distribution_config_vars,
    )

    # Optax SGD
    optim = OptaxWithStochasticVars(
        optimizable_model,
        "sgd",
        0.01,
        {},
        batch_size=10,
        num_batches=10,
        num_epochs=500,
        print_every=100,
    )
    opt_param = optim.optimize()
    print(opt_param)
    assert np.isclose(opt_param["c"], 1.0, atol=0.2)
    assert opt_param["c"] >= params_min["c"]
    assert opt_param["c"] <= params_max["c"]


if __name__ == "__main__":
    # test_optimization_unbounded()
    # test_optimization_bounded()
    # test_optimization_constrained()
    # test_optimization_stochastic()
    # test_optimization_transforms()
    test_optimization_transforms_stochastic()
