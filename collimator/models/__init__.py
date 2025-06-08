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

from .pendulum import Pendulum, PendulumDiagram, animate_pendulum
from .lotka_volterra import LotkaVolterra
from .fitzhugh_nagumo import FitzHughNagumo
from .van_der_pol import VanDerPol
from .bouncing_ball import BouncingBall
from .planar_quadrotor import PlanarQuadrotor, animate_planar_quadrotor
from .rimless_wheel import RimlessWheel
from .compass_gait import CompassGait
from .compact_ev import CompactEV, DummyBlock
from .acrobot import Acrobot, animate_acrobot
from .cartpole import CartPole, animate_cartpole
from .battery_ecm import Battery
from .hairer import (
    EulerRigidBody,
    ArenstorfOrbit,
    Lorenz,
    Pleiades,
)

__all__ = [
    "Pendulum",
    "PendulumDiagram",
    "animate_pendulum",
    "LotkaVolterra",
    "FitzHughNagumo",
    "VanDerPol",
    "BouncingBall",
    "PlanarQuadrotor",
    "animate_planar_quadrotor",
    "RimlessWheel",
    "CompassGait",
    "CompactEV",
    "DummyBlock",
    "Acrobot",
    "animate_acrobot",
    "CartPole",
    "animate_cartpole",
    "Battery",
    "EulerRigidBody",
    "ArenstorfOrbit",
    "Lorenz",
    "Pleiades",
]
