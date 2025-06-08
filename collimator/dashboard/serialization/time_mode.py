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
These funtions are used to classify nodes and signals of a wildcat diagram in terms of the
time_modes that are displayed in the UI.
Wildcat does not rely on time_mode propagation nor assignment per-se to interpret the model,
so there is not necessray a perfect set of rules here.
The outcome is merely guidance for the user in the UI, so errors are not critical.
"""

from collimator.framework.dependency_graph import DependencyTicket
from .ui_types import TimeMode


def time_mode_node(node) -> TimeMode:
    # FIXME: it would probably be better to distribute this 'time_mode'
    # attribute to the blocks. e.g. LeafSystem.time_mode='agnostic', then in
    # each block def, if block not agnostic, explicitly set it to one of:
    #   'continuous', 'discrete', 'hybrid'
    # as appropriate for the block. Note, for user defined blocks, it is set from
    # the UI.
    discrete_blocks = [
        "DerivativeDiscrete",
        "DirectShootingNMPC",
        "DirectTranscriptionNMPC",
        "HermiteSimpsonNMPC",
        "DiscreteInitializer",
        "DiscreteClock",
        "DiscreteTimeLinearQuadraticRegulator",
        "EdgeDetection",
        "FilterDiscrete",
        "IntegratorDiscrete",
        "LinearDiscreteTimeMPC",
        "LinearDiscreteTimeMPC_OSQP",
        "LTISystemDiscrete",
        "ModelicaFMU",
        "PIDDiscrete",
        "Pulse",
        "RateLimiter",
        "TransferFunctionDiscrete",
        "ZeroOrderHold",
        "UnitDelay",
    ]

    user_defined_blocks = ["CustomJaxBlock", "CustomPythonBlock", "StateMachine"]

    continuous_blocks = [
        "Integrator",
        "LTISystem",
        "TransferFunction",
        "PID",
        "Derivative",
        "ContinuousTimeSindyWithControl",
    ]

    # node time_mode rules.
    if node.__class__.__name__ in discrete_blocks:
        return TimeMode.DISCRETE
    elif (
        node.__class__.__name__ in user_defined_blocks and node.time_mode == "discrete"
    ):
        return TimeMode.DISCRETE
    elif node.__class__.__name__ == "StateMachine":
        # HACK: when StateMachine time_mode==Agnostic, its output still has dependency
        # on DependencyTicket.xd because that is how output values are held when no
        # action sets them. So we cannot rely on the dep graph to assign time mode for
        # the StateMachine agnostic case (not that this would make things any different in wildcat).
        return TimeMode.CONTINUOUS
    elif node.__class__.__name__ in continuous_blocks:
        return TimeMode.CONTINUOUS

    return None


def time_mode_node_with_ports(ports_tm: list[TimeMode]) -> TimeMode:
    # node time_mode rules for block which do not have time mode assigned based on their
    # class.
    if TimeMode.CONTINUOUS in ports_tm and TimeMode.DISCRETE in ports_tm:
        # this case is actually used for subdiagrams, and ports_tm is actaully
        # a list of the subdiagams' nodes time_modes.
        return TimeMode.HYBRID
    elif TimeMode.CONTINUOUS in ports_tm:
        return TimeMode.CONTINUOUS
    elif TimeMode.DISCRETE in ports_tm:
        return TimeMode.DISCRETE
    else:
        # if nothing, this is the only option left.
        return TimeMode.CONSTANT


def time_mode_port(out_port, node_cls_tm) -> TimeMode:
    # extract the dependencies we care about for this port.
    dep_none = out_port.tracker.depends_on([DependencyTicket.nothing])
    dep_time = out_port.tracker.depends_on([DependencyTicket.time])
    dep_xc = out_port.tracker.depends_on([DependencyTicket.xc])
    dep_xd = out_port.tracker.depends_on([DependencyTicket.xd])
    xd_dep = out_port.tracker.is_prerequisite_of([DependencyTicket.xd])
    xcdot_dep = out_port.tracker.is_prerequisite_of([DependencyTicket.xcdot])

    # Port time_mode rules.
    if node_cls_tm is not None:
        return node_cls_tm
    elif (dep_time or dep_xc) and xcdot_dep:
        return TimeMode.CONTINUOUS
    elif (dep_xd and dep_xc) and not xd_dep:
        return TimeMode.CONTINUOUS
    elif dep_xd or xd_dep:
        return TimeMode.DISCRETE
    elif dep_none and not dep_time and not dep_xc and not dep_xd:
        return TimeMode.CONSTANT
    else:
        return TimeMode.CONTINUOUS
