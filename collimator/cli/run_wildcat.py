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

import argparse
import ast
import enum
import functools
import json
import logging
import os
import pickle
import time
import traceback
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib

import collimator
import collimator.logging as wildcat_logger
from collimator import Simulator
from collimator.cli.run_optimization import run_optimization
from collimator.backend import numpy_api as cnp
from collimator.framework import build_recorder
from collimator.framework.error import CollimatorError
import collimator.dashboard.serialization.ui_types as ts
from collimator.dashboard.serialization import json_array, model_json
from collimator.dashboard.serialization.ui_types import OptimizationResultsJson

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s][%(levelname)s]: %(message)s",
)

logging.getLogger("jax._src.callback").setLevel(logging.CRITICAL)


@contextmanager
def set_cwd(path):
    """Sets the cwd within the context

    Args:
        path (str): The path to the cwd

    Yields:
        None
    """

    origin = os.path.abspath(path)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Create a dictionary with the required log record attributes
        extras = record.__dict__.get("extras") or {}
        log_record = {
            **extras,
            "level": str(record.levelname),
            "message": str(record.getMessage()),
            "timestamp": record.created,
        }

        # Use json.dumps to format the dictionary as a JSON string
        try:
            return json.dumps(log_record)
        except TypeError as e:
            logger.error(
                "Error serializing log record to JSON: %s. Record: %s",
                e,
                log_record,
            )
            extras = {k: str(v) for k, v in extras.items()}
            if "__error__" in extras:  # make it visible (fallback to text)
                extras["error"] = extras.pop("__error__")
            log_record = {
                **extras,
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": record.created,
            }
            return json.dumps(log_record)


class TaskType(enum.StrEnum):
    OPTIMIZATION = "optimization"
    BATCH_SIMULATION = "batch_simulation"
    SIMULATION = "simulation"
    VALIDATION = "validation"
    GENERATE_MODEL_PYTHON_CODE = "generate_model_python_code"


class WildcatApplicationError(Exception):
    def __init__(self, internal_msg=None, user_msg=None):
        """An error that is raised by the Wildcat application.

        Args:
            internal_msg (str): The error message for the application. Will show up in server logs.
            user_msg (str, optional): The error message shown to the user. Defaults to None.
        """
        self.internal_msg = internal_msg
        self.user_msg = user_msg
        super().__init__(internal_msg)


class _VarRecorder(ast.NodeVisitor):
    """Used to record all variables in a Python expression."""

    def __init__(self):
        super().__init__()
        self.vars = set()

    def visit_Name(self, node):
        self.vars.add(node.id)
        return node.id


def uses_model_parameters(python_code: str, model_params: set[str]):
    # look for variables used in the expression
    tree = ast.parse(python_code, mode="exec")
    var_recorder = _VarRecorder()
    var_recorder.visit(tree)

    for var in var_recorder.vars:
        if var in model_params:
            return True
    return False


def run_batch_simulation(
    parameters: list[dict[str, Any]], default_parameters: dict[str, Any] = None
):
    logger.info("Running batch simulation with parameters: %s", parameters)

    model_parameters_json = {}
    if default_parameters:
        model_parameters_json = {
            k: model_json.Parameter(value=str(v), is_string=False)
            for k, v in default_parameters.items()
        }
    model_parameters_json.update(
        {
            k: model_json.Parameter(value=str(v), is_string=False)
            for k, v in parameters[0].items()
        }
    )
    sim_context = collimator.load_model_from_dir(
        modeldir=".", model="model.json", parameter_overrides=model_parameters_json
    )

    # If there is an init_script, check if model parameters are used in it and
    # raise an error
    if sim_context.init_script:
        # FIXME: it's possible to detect model parameters used in the script but
        # the script will still overwrite them since we indiscriminately pass
        # all model parameters to it, they will end up in the init script namespace
        # as resolved parameters

        # with open(sim_context.init_script) as init_script_file:
        #     py_code = init_script_file.read()

        # if uses_model_parameters(py_code, set(model_parameters_json.keys())):
        #     raise ValueError(
        #         "Can't use model parameters in init scripts in ensemble simulations."
        #     )
        raise ValueError("Ensemble simulations do not support init scripts.")

    diagram = sim_context.diagram
    recorded_signals = sim_context.recorded_signals

    sim_context.simulator_options.recorded_signals = recorded_signals
    sim_context.simulator_options.save_time_series = True

    param_keys = list(parameters[0].keys())

    for config in parameters:
        if set(config.keys()) != set(param_keys):
            raise ValueError("All parameter configurations must have the same keys")

    simulator = Simulator(diagram, options=sim_context.simulator_options)

    advance_to = simulator.advance_to
    if cnp.active_backend == "jax":
        # if any model param is used as a static param, bail out
        for param_name in param_keys:
            static_dependents = sim_context.model_parameters[
                param_name
            ].static_dependents()
            if static_dependents:
                system_names = [dep.system.name for dep in static_dependents]
                system_names = ", ".join(system_names)
                raise ValueError(
                    f"Parameter '{param_name}' is used as a static parameter in block(s): {system_names}.\n"
                    "Static parameters are not yet supported in batch simulation with backend JAX."
                )
        advance_to = jax.jit(advance_to)

    def _run_sim(config):
        for k, v in zip(param_keys, config):
            diagram.parameters[k].set(v)

        context = diagram.create_context()

        # output ports may have changed
        for port_path, port in recorded_signals.items():
            recorded_signals[port_path] = port.system.output_ports[port.index]

        results = advance_to(sim_context.stop_time, context)
        return results.results_data

    configs = [[v for v in config.values()] for config in parameters]

    results = []
    for config in configs:
        result = _run_sim(config)
        t, outputs = result.finalize()
        outputs["time"] = t
        results.append(outputs)

    return results


def _save_progress(
    logsdir, start_time, stop_time, progress_last_saved_ts, current_time
):
    # Throttle writes to progress.json
    now = time.time()
    last = progress_last_saved_ts[0]
    if now - last < 0.1:
        return
    progress_last_saved_ts[0] = now

    with open(os.path.join(logsdir, "progress.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "start_time": float(start_time),
                "current_time": float(current_time),
                "target_time": float(stop_time),
            },
            f,
        )


def run_simulation(parameters: dict[str, dict] = None) -> dict[str, jnp.ndarray]:
    # FIXME: ideally we don't have to save the model json as a file and can directly
    # pass the json string or dict to load_model.
    model = collimator.load_model(
        modeldir=".",
        model="model.json",
        logsdir="logs",
        npydir="npy",
        parameter_overrides=parameters,
    )

    start_time = model.sim_context.start_time
    stop_time = model.sim_context.stop_time

    progress_last_saved_ts = [time.time()]
    _save_progress(
        model.logsdir, start_time, stop_time, progress_last_saved_ts, start_time
    )
    model.simulator_options.major_step_callback = functools.partial(
        _save_progress, model.logsdir, start_time, stop_time, progress_last_saved_ts
    )

    return model.simulate(start_time=start_time, stop_time=stop_time)


def run_validation(parameters: dict[str, dict] = None):
    collimator.load_model(
        "./",
        model="model.json",
        logsdir="logs",
        npydir="npy",
        parameter_overrides=parameters,
    )


def generate_code(parameters: dict[str, dict] = None):
    build_recorder.start()
    collimator.load_model(
        "./",
        model="model.json",
        logsdir="logs",
        npydir="npy",
        parameter_overrides=parameters,
    )
    return build_recorder.generate_code()


def _get_collimator_error(e: Exception) -> CollimatorError:
    if not e:
        return None
    if isinstance(e, CollimatorError):
        return e
    return _get_collimator_error(e.__cause__)


def _remove_xla_runtime_error(e: Exception):
    if not e:
        return
    if isinstance(e.__cause__, jaxlib.xla_extension.XlaRuntimeError):
        e.__cause__ = None
        return
    _remove_xla_runtime_error(e.__cause__)


def _format_error(e: Exception) -> str:
    # remove xla runtime error because it's printing our internal stack trace
    # which should not be shown to the user. Note that it will still be visible
    # in the server logs.
    _remove_xla_runtime_error(e)
    msg = str(e)
    return f"{e.__class__.__name__}: {msg}"


def run_wildcat(
    work_dir: str,
    task_type: TaskType,
    log_level: str = "INFO",
    **run_fn_kwargs,
):
    wildcat_logger.set_log_level(log_level)
    wildcat_logger.unset_stream_handler()
    logs_filepath = os.path.join(work_dir, "logs", "logs.txt")
    wildcat_logger.set_file_handler(logs_filepath, formatter=JsonFormatter())

    with set_cwd(work_dir):
        t0 = time.perf_counter()
        try:
            match task_type:
                case TaskType.OPTIMIZATION:
                    results = run_optimization(**run_fn_kwargs)
                    return OptimizationResultsJson.from_results(*results)
                case TaskType.BATCH_SIMULATION:
                    return run_batch_simulation(**run_fn_kwargs)
                case TaskType.SIMULATION:
                    # FIXME the returned value of a simulation run is not used,
                    # so we could probably avoid passing it back through
                    # results.pkl and ray. Results *.npy are sent to s3 directly.
                    return run_simulation(**run_fn_kwargs)
                case TaskType.VALIDATION:
                    return run_validation(**run_fn_kwargs)
                case TaskType.GENERATE_MODEL_PYTHON_CODE:
                    return generate_code(**run_fn_kwargs)
                case _:
                    raise ValueError(f"Unknown task type: {task_type}")

        except CollimatorError as e:
            traceback.print_exc()
            # See JsonFormatter in run_wildcat
            # The frontend understands this __error__ object
            error_object = ts.ErrorLog.from_error(e).to_api(omit_none=True)
            extras = {"__error__": error_object}
            msg = _format_error(e)
            wildcat_logger.error(msg, extra={"extras": extras})
            raise WildcatApplicationError(str(e), user_msg=msg) from e

        except BaseException as e:
            traceback.print_exc()
            msg = _format_error(e)
            collimator_error = _get_collimator_error(e)
            if collimator_error:
                error_object = ts.ErrorLog.from_error(e).to_api(omit_none=True)
                extras = {"__error__": error_object}
                wildcat_logger.error(msg, extra={"extras": extras})
            else:
                wildcat_logger.error(msg)
            raise WildcatApplicationError(str(e), user_msg=msg) from e

        finally:
            t1 = time.perf_counter()
            wildcat_logger.info("Total time: %.3fs", t1 - t0)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        default=".",
        help="Working directory.",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        required=False,
        help="Pickled or JSON kwargs filepath.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="simulation",
        help="Task type. One of 'simulation', 'ensemble' or 'optimization'.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        required=False,
        default="INFO",
        help="Wildcat log level.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main():
    args = _get_args()

    pickled_kwargs = {}
    if args.kwargs:
        if args.kwargs.endswith(".json"):
            with open(args.kwargs, "r", encoding="utf-8") as f:
                pickled_kwargs = json.load(f)
        else:
            with open(args.kwargs, "rb") as f:
                pickled_kwargs = pickle.load(f)

    results = run_wildcat(
        args.work_dir, args.task_type, log_level=args.log_level, **pickled_kwargs
    )

    # This is useful mostly for local debugging via manual calls to run_wildcat
    if args.output_json:
        outfile = "/dev/stdout" if args.output_json == "-" else args.output_json
        with open(outfile, "w", encoding="utf-8") as f:
            json_array.dump(results, f, indent=2)

    # The pickled file is actually used by simworker
    # FIXME: ensure safety when reading back from it
    with open(f"{args.work_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
