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

from dataclasses import dataclass
import dataclasses
from typing import Optional, Any, Union
import sys
import ts_type as ts

from dataclasses_jsonschema import JsonSchemaMixin
import numpy as np
import jax

from collimator.framework.error import CollimatorError
from collimator.framework.parameter import Parameter
from collimator.framework.system_base import SystemBase

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum


class ToApiMixin:
    """Mixin to convert dataclasses to JSON-compatible dictionaries."""

    @classmethod
    def _to_api(cls, d):
        # This converts lists of jax and numpy arrays into serializable lists
        if isinstance(d, (list, tuple)):
            d = np.asarray(d)
            return [cls._to_api(v) for v in d]
        if isinstance(d, dict):
            return {k: cls._to_api(v) for k, v in d.items()}
        if isinstance(d, (np.ndarray, np.number, jax.Array)):
            return np.asarray(d).tolist()
        return d

    @classmethod
    def _to_api_filter_none(cls, d: dict) -> dict:
        # This drops all None fields from dicts, which gives us better JSON payloads
        ret = {}
        if isinstance(d, (list, tuple)):
            return [cls._to_api_filter_none(v) for v in d]
        if not isinstance(d, dict):
            return d
        for k, v in d.items():
            if v is not None:
                ret[k] = cls._to_api_filter_none(v)
        return ret

    def to_api(self, omit_none=True):
        """Converts the object to a dictionary that can be serialized to JSON."""
        d = dataclasses.asdict(self)
        if omit_none:
            d = self._to_api_filter_none(d)
        return self._to_api(d)


@ts.gen_type
class PortDirection(StrEnum):
    IN = "in"
    OUT = "out"


@ts.gen_type
class TimeMode(StrEnum):
    CONSTANT = "Constant"
    DISCRETE = "Discrete"
    CONTINUOUS = "Continuous"
    HYBRID = "Hybrid"
    ACAUSAL = "Acausal"


@ts.gen_type
@dataclass
class Port(JsonSchemaMixin, ToApiMixin):
    index: int
    dtype: str
    dimension: list[int]
    time_mode: TimeMode
    discrete_interval: Optional[float]
    name: str


@ts.gen_type
@dataclass
class Node(JsonSchemaMixin, ToApiMixin):
    namepath: list[str]
    uuidpath: list[str]
    outports: list[Port]
    time_mode: TimeMode
    discrete_interval: Optional[float]


@ts.gen_type
@dataclass
class SignalTypes(JsonSchemaMixin, ToApiMixin):
    nodes: list[Node]


@ts.gen_type
@dataclass
class ErrorLoopItem(JsonSchemaMixin, ToApiMixin):
    name_path: Optional[str] = None
    uuid_path: Optional[list[str]] = None
    port_direction: Optional[PortDirection] = None
    port_index: Optional[int] = None


@ts.gen_type
@dataclass
class ErrorLog(JsonSchemaMixin, ToApiMixin):
    kind: str
    name_path: Optional[str] = None
    uuid_path: Optional[list[str]] = None
    port_direction: Optional[str] = None
    port_name: Optional[str] = None
    port_index: Optional[int] = None
    parameter_name: Optional[str] = None
    loop: Optional[list[ErrorLoopItem]] = None

    @classmethod
    def from_error(cls, error: CollimatorError):
        def _path(loc) -> str | None:
            return ".".join(loc.name_path) if loc.name_path else None

        return cls(
            kind=error.__class__.__name__,
            name_path=_path(error),
            uuid_path=error.ui_id_path,
            port_direction=error.port_direction,
            port_name=error.port_name,
            port_index=error.port_index,
            parameter_name=error.parameter_name,
            loop=(
                [
                    ErrorLoopItem(
                        name_path=_path(loc),
                        uuid_path=loc.ui_id_path,
                        port_direction=loc.port_direction,
                        port_index=loc.port_index,
                    )
                    for loc in error.loop
                ]
                if error.loop
                else None
            ),
        )


@ts.gen_type
@dataclass
class OptimalParameterJson(JsonSchemaMixin, ToApiMixin):
    param_name: str  # Model or block parameter name
    optimal_value: str  # Expression of the optimal value (eg. '[1.2]')
    block_namepath: Optional[list[str]] = None  # For PID tuning and param estimation
    block_uuidpath: Optional[list[str]] = None  # For PID tuning and param estimation

    @classmethod
    def from_array(cls, name: str, value: Any) -> "OptimalParameterJson":
        def expr(v) -> str:
            if isinstance(v, Parameter):
                ex, _ = v.value_as_api_param(False, False)
                return ex
            return str(v)

        # Used by PID tuning
        if hasattr(value, "system") and isinstance(value.system, SystemBase):
            return cls(
                param_name=name.split(".")[-1],  # drops block path
                optimal_value=expr(value),
                block_namepath=value.system.name_path,
                block_uuidpath=value.system.ui_id_path,
            )

        return cls(name, expr(value))


@ts.gen_type
@dataclass
class OptimizationMetricJson(JsonSchemaMixin, ToApiMixin):
    name: str
    value: Union[list[float], float]

    @classmethod
    def from_array(cls, name: str, value: Any) -> "OptimizationMetricJson":
        return cls(name=name, value=value)


@ts.gen_type
@dataclass
class OptimizationResultsJson(JsonSchemaMixin, ToApiMixin):
    optimal_parameters: list[OptimalParameterJson]
    metrics: list[OptimizationMetricJson]

    @classmethod
    def from_results(
        cls, optimal_params: dict[str, Any], metrics: dict[str, Any]
    ) -> "OptimizationResultsJson":
        return cls(
            optimal_parameters=[
                OptimalParameterJson.from_array(name, value)
                for name, value in optimal_params.items()
            ],
            metrics=[
                OptimizationMetricJson.from_array(name, value)
                for name, value in metrics.items()
            ],
        )


# API request types


NodePathType = str
SignalPathType = str


class OptimizationType(StrEnum):
    DESIGN = "design"
    ESTIMATION = "estimation"
    PID = "pid"


class OptimizationAlgorithm(StrEnum):
    ADAM = "adam"
    RMS_PROP = "rmsprop"
    STOCHASTIC_GRADIENT_DESCENT = "stochastic_gradient_descent"
    PARTICLE_SWARM_OPTIMIZATION = "particle_swarm_optimization"
    SIMULATED_ANNEALING = "simulated_annealing"
    BFGS = "bfgs"
    L_BFGS_B = "l_bfgs_b"
    SEQUENTIAL_LEAST_SQUARES = "sequential_least_squares"
    CMA_ES = "covariance_matrix_adaptation_evolution_strategy"
    GENETIC_ALGORITHM = "genetic_algorithm"


class StochasticDistribution(StrEnum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"


OptimizationAlgoParam = Optional[Union[str, bool]]


@ts.gen_type
@dataclass
class DesignParameterJson(JsonSchemaMixin, ToApiMixin):
    param_name: str
    initial: str = "0.0"
    min: Optional[str] = None
    max: Optional[str] = None


@ts.gen_type
@dataclass
class StochasticParameterJson(JsonSchemaMixin, ToApiMixin):
    param_name: str
    distribution: StochasticDistribution
    min: Optional[str] = None
    max: Optional[str] = None
    mean: Optional[str] = None
    std_dev: Optional[str] = None


@ts.gen_type
@dataclass
class OptimizationRequestJson(JsonSchemaMixin, ToApiMixin):
    type: OptimizationType
    algorithm: OptimizationAlgorithm
    options: dict[str, OptimizationAlgoParam] = dataclasses.field(default_factory=dict)
    objective: Optional[SignalPathType] = None
    constraints: list[SignalPathType] = dataclasses.field(default_factory=list)
    data_file: Optional[str] = None
    time_column: Optional[str] = None
    input_columns: Optional[dict[str, str]] = None
    output_columns: Optional[dict[str, str]] = None
    constraint_port_names: Optional[list[str]] = None
    submodel_path: Optional[NodePathType] = None
    pid_blocks: Optional[list[NodePathType]] = None
    error_signal: Optional[SignalPathType] = None
    stochastic_parameters: list[StochasticParameterJson] = dataclasses.field(
        default_factory=list
    )
    design_parameters: list[DesignParameterJson] = dataclasses.field(
        default_factory=list
    )
    json_model_with_cost: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict) -> "OptimizationRequestJson":
        return cls.from_dict(data)


@ts.gen_type
@dataclass
class FitSindyResult:
    coefficients: list[list[float]]
    base_feature_names: list[str]
    feature_names: list[str]
    equations: list[str]
    has_control_input: bool


@ts.gen_type
@dataclass
class SignalImage:
    signal_name: str
    x_range: list[int]
    y_range: list[int]


@ts.gen_type
@dataclass
class EnsembleResults:
    signal_images: list[SignalImage]
