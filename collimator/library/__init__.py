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

from collimator.backend.backend import IS_JAXLITE

from .generic import (
    SourceBlock,
    FeedthroughBlock,
    ReduceBlock,
)
from .primitives import (
    Abs,
    Arithmetic,
    Adder,
    Chirp,
    Clock,
    Comparator,
    Constant,
    CrossProduct,
    DeadZone,
    Demultiplexer,
    DerivativeDiscrete,
    DiscreteClock,
    DiscreteInitializer,
    DotProduct,
    EdgeDetection,
    Exponent,
    FilterDiscrete,
    Gain,
    IfThenElse,
    Integrator,
    IntegratorDiscrete,
    IOPort,
    Logarithm,
    LogicalOperator,
    LogicalReduce,
    LookupTable1d,
    LookupTable2d,
    MatrixConcatenation,
    MatrixInversion,
    MatrixMultiplication,
    MatrixTransposition,
    MinMax,
    Multiplexer,
    Offset,
    PIDDiscrete,
    Power,
    Product,
    ProductOfElements,
    Pulse,
    Quantizer,
    Ramp,
    RateLimiter,
    Reciprocal,
    Relay,
    Saturate,
    Sawtooth,
    ScalarBroadcast,
    Sine,
    SignalDatatypeConversion,
    Slice,
    SquareRoot,
    Stack,
    Step,
    Stop,
    SumOfElements,
    Trigonometric,
    UnitDelay,
    ZeroOrderHold,
)
from .battery_cell import BatteryCell
from .custom import (
    CustomJaxBlock,
    CustomPythonBlock,
)
from .wrappers import (
    ode_block,
    feedthrough_block,
)
from .linear_system import (
    LTISystem,
    TransferFunction,
    linearize,
    PID,
    Derivative,
    LTISystemDiscrete,
    TransferFunctionDiscrete,
)

from .random import (
    RandomNumber,
    WhiteNoise,
)

from .rotations import (
    CoordinateRotation,
    CoordinateRotationConversion,
    RigidBody,
)

from .data_source import (
    DataSource,
)
from .state_machine import (
    StateMachine,
)

from .reference_subdiagram import ReferenceSubdiagram


if not IS_JAXLITE:
    from .mpc import (
        LinearDiscreteTimeMPC,
        LinearDiscreteTimeMPC_OSQP,
    )

    from .nmpc import (
        DirectShootingNMPC,
        DirectTranscriptionNMPC,
        HermiteSimpsonNMPC,
    )

    from .lqr import (
        LinearQuadraticRegulator,
        DiscreteTimeLinearQuadraticRegulator,
        FiniteHorizonLinearQuadraticRegulator,
    )

    from .mujoco import (
        MJX,
        MuJoCo,
    )

    from .nn import (
        MLP,
    )

    from .costs_and_losses import (
        QuadraticCost,
    )

    from .fmu_import import (
        ModelicaFMU,
    )

    from .sindy import (
        Sindy,
    )

    from .ansys import (
        PyTwin,
    )

    from .ros2 import (
        Ros2Publisher,
        Ros2Subscriber,
    )

    from .state_estimators import (
        KalmanFilter,
        InfiniteHorizonKalmanFilter,
        ContinuousTimeInfiniteHorizonKalmanFilter,
        ExtendedKalmanFilter,
        UnscentedKalmanFilter,
    )

    from .predictor import (
        PyTorch,
        TensorFlow,
    )

    from .video import VideoSink, VideoSource

    from .quanser import QuanserHAL, QubeServoModel

else:
    # NOTE We could improve this by defining a different list based on Emscripten
    # vs. full environment. For now, we just raise an error at runtime. Much simpler.

    class JaxliteNotSupportedError(RuntimeError):
        pass

    class _InvalidBlock:
        def __init__(self, _class: str, *args, **kwargs):
            raise JaxliteNotSupportedError(
                f"Block not available with jaxlite: {_class}"
            )

    def _invalid(name):
        return lambda *args, **kwargs: _InvalidBlock(name, *args, **kwargs)

    LinearDiscreteTimeMPC = _invalid("LinearDiscreteTimeMPC")
    LinearDiscreteTimeMPC_OSQP = _invalid("LinearDiscreteTimeMPC_OSQP")
    DirectShootingNMPC = _invalid("DirectShootingNMPC")
    DirectTranscriptionNMPC = _invalid("DirectTranscriptionNMPC")
    HermiteSimpsonNMPC = _invalid("HermiteSimpsonNMPC")
    LinearQuadraticRegulator = _invalid("LinearQuadraticRegulator")
    DiscreteTimeLinearQuadraticRegulator = _invalid(
        "DiscreteTimeLinearQuadraticRegulator"
    )
    FiniteHorizonLinearQuadraticRegulator = _invalid(
        "FiniteHorizonLinearQuadraticRegulator"
    )
    MJX = _invalid("MJX")
    MuJoCo = _invalid("MuJoCo")
    MLP = _invalid("MLP")
    QuadraticCost = _invalid("QuadraticCost")
    ModelicaFMU = _invalid("ModelicaFMU")
    Sindy = _invalid("Sindy")
    PyTwin = _invalid("PyTwin")
    Ros2Publisher = _invalid("Ros2Publisher")
    Ros2Subscriber = _invalid("Ros2Subscriber")
    KalmanFilter = _invalid("KalmanFilter")
    InfiniteHorizonKalmanFilter = _invalid("InfiniteHorizonKalmanFilter")
    ContinuousTimeInfiniteHorizonKalmanFilter = _invalid(
        "ContinuousTimeInfiniteHorizonKalmanFilter"
    )
    ExtendedKalmanFilter = _invalid("ExtendedKalmanFilter")
    UnscentedKalmanFilter = _invalid("UnscentedKalmanFilter")
    PyTorch = _invalid("PyTorch")
    TensorFlow = _invalid("TensorFlow")
    VideoSink = _invalid("VideoSink")
    VideoSource = _invalid("VideoSource")
    QuanserHAL = _invalid("QuanserHAL")
    QubeServoModel = _invalid("QubeServoModel")


__all__ = [
    "Arithmetic",
    "SourceBlock",
    "FeedthroughBlock",
    "ReduceBlock",
    "Abs",
    "Constant",
    "Sine",
    "BatteryCell",
    "Clock",
    "Comparator",
    "CoordinateRotation",
    "CoordinateRotationConversion",
    "CrossProduct",
    "CustomJaxBlock",
    "CustomPythonBlock",
    "DataSource",
    "DeadZone",
    "Derivative",
    "DerivativeDiscrete",
    "DiscreteInitializer",
    "DotProduct",
    "DiscreteClock",
    "EdgeDetection",
    "Exponent",
    "FilterDiscrete",
    "Gain",
    "IfThenElse",
    "Offset",
    "Reciprocal",
    "LogicalOperator",
    "LogicalReduce",
    "MatrixConcatenation",
    "MatrixInversion",
    "MatrixMultiplication",
    "MatrixTransposition",
    "ModelicaFMU",
    "MinMax",
    "Multiplexer",
    "Demultiplexer",
    "Adder",
    "PID",
    "Product",
    "ProductOfElements",
    "Power",
    "Integrator",
    "IntegratorDiscrete",
    "IOPort",
    "Logarithm",
    "LookupTable1d",
    "LookupTable2d",
    "Chirp",
    "Pulse",
    "Quantizer",
    "RandomNumber",
    "Relay",
    "RigidBody",
    "Sawtooth",
    "ScalarBroadcast",
    "Sindy",
    "SumOfElements",
    "Slice",
    "StateMachine",
    "Stack",
    "Step",
    "Stop",
    "SquareRoot",
    "Ramp",
    "RateLimiter",
    "Saturate",
    "PIDDiscrete",
    "WhiteNoise",
    "ZeroOrderHold",
    "UnitDelay",
    "ode_block",
    "feedthrough_block",
    "LTISystem",
    "LTISystemDiscrete",
    "TransferFunction",
    "TransferFunctionDiscrete",
    "linearize",
    "LinearDiscreteTimeMPC",
    "LinearDiscreteTimeMPC_OSQP",
    "DirectShootingNMPC",
    "DirectTranscriptionNMPC",
    "HermiteSimpsonNMPC",
    "MJX",
    "MuJoCo",
    "MLP",
    "QuadraticCost",
    "Trigonometric",
    "ReferenceSubdiagram",
    "KalmanFilter",
    "InfiniteHorizonKalmanFilter",
    "ContinuousTimeInfiniteHorizonKalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "LinearQuadraticRegulator",
    "DiscreteTimeLinearQuadraticRegulator",
    "FiniteHorizonLinearQuadraticRegulator",
    "PyTwin",
    "PyTorch",
    "TensorFlow",
    "VideoSink",
    "VideoSource",
    "Ros2Publisher",
    "Ros2Subscriber",
    "SignalDatatypeConversion",
    "QubeServoModel",
    "QuanserHAL",
]
