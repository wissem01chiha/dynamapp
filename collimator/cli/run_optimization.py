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

import glob
import json
import os
import sys
from typing import Any

import math
import click
import numpy as np
from simpleeval import EvalWithCompoundTypes

from collimator.logging import logdata, logger
from collimator.library import PID, PIDDiscrete
from collimator.framework import Diagram, LeafSystem
from collimator.dashboard.serialization.ui_types import (
    OptimizationRequestJson,
    OptimizationResultsJson,
    StochasticParameterJson,
    DesignParameterJson,
)

from collimator.optimization import ui_jobs
from collimator.dashboard.serialization import from_model_json, model_json
from collimator.optimization.framework.base.metrics import MetricsWriter
from collimator.optimization.framework.base.optimizable import (
    DesignParameter,
    Distribution,
    StochasticParameter,
)


_REQUIRED = object()


def _tofloats(v) -> np.ndarray:
    return np.array(v, dtype=float)


def _parse_option(s: str | bool, default: float = None, floats=False) -> Any:
    if isinstance(s, bool):
        return s

    if s is None or s == "":
        if default is _REQUIRED:
            raise ValueError("Value is None and no default provided.")
        return default

    # Safe(r), simple call to eval. Note that we could use plain eval() since we're
    # already inside bwrap. But this is always gonna be safer, although we pass
    # np so it's not completely safe.
    value = EvalWithCompoundTypes(
        names={"np": np, "math": math, "true": True, "false": False},
    ).eval(s)

    if floats:
        return _tofloats(value)

    return value


def _find_blocks(diagram: Diagram, paths: list[str], allowed_types: tuple[type] = None):
    found: list[LeafSystem] = []

    for path in paths:
        block = diagram.find_system_with_path(path)
        if not block:
            raise ValueError(f"Block with path '{path}' not found.")
        if allowed_types and not isinstance(block, allowed_types):
            types = ", ".join([k.__class__.__name__ for k in allowed_types])
            blk_type = block.__class__.__name__
            raise ValueError(f"Block at '{path}' ({blk_type}) is not one of: {types}.")
        found.append(block)

    return found


def _find_signal(diagram: Diagram, path: str):
    block_path = ".".join(path.split(".")[:-1])
    port_name = path.split(".")[-1]

    block = diagram.find_system_with_path(block_path)
    if not block:
        raise ValueError(f"Block with path '{block_path}' not found.")
    port = block.get_output_port(port_name)
    if not port:
        raise ValueError(
            f"Port with name '{port_name}' not found in block '{block_path}'."
        )

    return port


def _parse_design_params(
    api_params: list[DesignParameterJson],
) -> list[DesignParameter]:
    return [
        DesignParameter(
            param_name=p.param_name,
            initial=_parse_option(p.initial, default=_REQUIRED, floats=True),
            min=_parse_option(p.min, default=-np.inf, floats=True),
            max=_parse_option(p.max, default=np.inf, floats=True),
        )
        for p in api_params
    ]


def _parse_stochastic_params(
    api_params: list[StochasticParameterJson],
) -> list[StochasticParameter]:
    def _options(p: StochasticParameterJson):
        d = {}
        if p.min is not None:
            d["min"] = _parse_option(p.min)
        if p.max is not None:
            d["max"] = _parse_option(p.max)
        if p.mean is not None:
            d["mean"] = _parse_option(p.mean)
        if p.std_dev is not None:
            d["std_dev"] = _parse_option(p.std_dev)
        return d

    return [
        StochasticParameter(
            param_name=p.param_name,
            distribution=Distribution(name=p.distribution, options=_options(p)),
        )
        for p in api_params
    ]


def _parse_options(api_options: dict) -> dict[str, float | bool | None]:
    return {k: _parse_option(v) for k, v in api_options.items() if v is not None}


def _dict_get(d: dict, path: str, default=None):
    keys = path.split(".")
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return default
    return d


def run_optimization(
    optimization_request: OptimizationRequestJson = None,
    request="request.json",
    model="model.json",
    metrics_output: str = None,
):
    """Run an optimization request from model.json and request.json."""

    if optimization_request is None:
        with open(request, encoding="utf-8") as f:
            optimization_request = json.load(f)
            optimization_request = optimization_request["optimization_request"]

    optim = OptimizationRequestJson.from_api(optimization_request)
    logger.debug("Running optimization: %s", optim)

    if optim.type not in ["design", "pid", "estimation"]:
        raise ValueError(f"Unknown optimization type: {optim.type}")

    human_type = {
        "design": "parameter design optimization",
        "pid": "PID tuning",
        "estimation": "parameter estimation",
    }
    logger.info(
        "Starting %s...",
        human_type[optim.type],
        **logdata(request=optimization_request),
    )

    modeldir = os.path.dirname(model)

    # NOTE: same as load_model_from_dir but with an optional hack
    # override for 'json_model_with_cost' (see below)
    file_pattern = os.path.join(modeldir, "submodel-*-latest.json")
    submodel_files = glob.glob(file_pattern)
    for submodel_file in submodel_files:
        ref_id = os.path.basename(submodel_file).split("-")[1:-1]
        ref_id = "-".join(ref_id)
        with open(submodel_file, "r", encoding="utf-8") as f:
            submodel = model_json.Model.from_json(f.read())
            from_model_json.register_reference_submodel(ref_id, submodel)

    # HACK: see https://github.com/collimator-ai/collimator/pull/6348
    # NOTE: This is actually used by the AI chat integration.
    if optim.json_model_with_cost:
        model_dict = json.loads(optim.json_model_with_cost)
    else:
        with open(model, "r", encoding="utf-8") as f:
            model_dict = json.load(f)

    sim_context = from_model_json.load_model(model_dict)
    diagram = sim_context.diagram

    pid_blocks = (
        _find_blocks(diagram, optim.pid_blocks, (PID, PIDDiscrete))
        if optim.pid_blocks
        else None
    )
    logger.debug("PID blocks: %s", pid_blocks)

    error_signal = (
        _find_signal(diagram, optim.error_signal) if optim.error_signal else None
    )
    logger.debug("Error signal: %s", error_signal)

    objective = _find_signal(diagram, optim.objective) if optim.objective else None
    logger.debug("Objective signal: %s", objective)

    constraints = [_find_signal(diagram, s) for s in optim.constraints]
    logger.debug("Constraint signals: %s", constraints)

    stochastic_parameters = _parse_stochastic_params(optim.stochastic_parameters or [])
    logger.debug("Stochastic parameters: %s", stochastic_parameters)

    design_parameters = _parse_design_params(optim.design_parameters or [])
    logger.debug("Design parameters: %s", design_parameters)

    options = _parse_options(optim.options or {})
    algorithm = optim.algorithm
    logger.debug("Algorithm: %s with options: %s", algorithm, options)

    data_file = optim.data_file
    time_column = optim.time_column
    input_columns = optim.input_columns
    output_columns = optim.output_columns
    constraint_port_names = optim.constraint_port_names
    logger.debug(
        'Data file: "%s" time column: "%s", inputs: %s, outputs: %s, constraints: %s',
        data_file,
        time_column,
        input_columns,
        output_columns,
        constraint_port_names,
    )

    submodel_path = optim.submodel_path
    logger.debug("Submodel path for param estimation: '%s'", submodel_path)

    # Note: we use the model's default start/stop times for optimization.
    # This can be very inefficient since simulations are often set up for much longer
    # times than optimization runs. But the UI does not provide other options and
    # adding that to the optimization modal would make it even more cluttered than
    # it already is.
    start_time = float(_dict_get(model_dict, "configuration.start_time", default=0.0))
    stop_time = float(_dict_get(model_dict, "configuration.stop_time", default=1.0))

    if optim.type == "estimation":
        # NOTE: we could drop 'allowed_types' if we want to run estimations on any block
        blocks = _find_blocks(diagram, submodel_path.split("."), allowed_types=Diagram)
        if not blocks or len(blocks) != 1:
            raise ValueError(f"Could not find submodel at path '{submodel_path}'.")
        diagram = blocks[0]

    metrics_writer = None
    if metrics_output is not None:
        metrics_writer = MetricsWriter(metrics_output)

    optimal_params, metrics = ui_jobs.jobs_router(
        optim.type,
        diagram,
        algorithm=algorithm,
        options=options,
        design_parameters=design_parameters,
        sim_t_span=(start_time, stop_time),
        objective_port=objective,
        constraint_ports=constraints,
        stochastic_parameters=stochastic_parameters,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_columns,
        output_port_names_to_column_names=output_columns,
        constraint_port_names=constraint_port_names,
        pid_blocks=pid_blocks,
        metrics_writer=metrics_writer,
    )

    logger.info(
        "Completed %s. Optimal parameters found: %s",
        human_type[optim.type],
        optimal_params,
    )

    return optimal_params, metrics


# CLI integration (mostly for quick local testing)
@click.command(
    name="optimize", help="Run an optimization request from model.json and request.json"
)
@click.option(
    "--model",
    default="model.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    show_default=True,
    help="Path to model JSON to load and run",
)
@click.option(
    "--request",
    default="request.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    show_default=True,
    help="Path to request JSON to execute",
)
@click.option(
    "--metrics-output",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help="Path to write metrics to (CSV file)",
)
def collimator_optimize(model, request, metrics_output):
    """CLI entrypoint to run an optimization from model.json and request.json."""
    optimal_params, metrics = run_optimization(
        request=request,
        model=model,
        metrics_output=metrics_output,
    )
    results_json = OptimizationResultsJson.from_results(optimal_params, metrics)
    json.dump(results_json.to_api(), sys.stdout, indent=2)
