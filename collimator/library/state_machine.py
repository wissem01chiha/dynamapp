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
StateMachine which meets the specification from design.
"""

from ..framework import LeafSystem, DependencyTicket
import dataclasses
from dataclasses import dataclass, field
from collections import namedtuple
from typing import List, Dict
from collimator.framework.state import LeafState
from collimator.framework.error import StaticError, BlockInitializationError
import jax
import jax.numpy as jnp
import numpy as np
import ast

__all__ = [
    "StateMachine",
]


@dataclass
class StateMachineTransition:
    """
    Attribues:
        guard:
            - must be python expression that evaluates to Bool.
            - may contain any number of <input_name>s or <output_name>s
        actions:
            - must be python statement.
            - may contain any number of <input_name>s or <output_name>s
    """

    dst: int = 0
    guard: str = "True"
    actions: List[str] = field(default_factory=list)


@dataclass
class StateMachineState:
    """Captures all elements of a state"""

    name: str = None
    # ordered list of exit transitions, ordered by priority
    transitions: List[StateMachineTransition] = field(default_factory=list)

    def with_transitions(self, transitions):
        return dataclasses.replace(self, transitions=transitions)


@dataclass
class StateMachineData:
    """Captures all elements of a state machine"""

    states: Dict[int, StateMachineState]
    intial_state: int
    inital_actions: str = field(default_factory=list)

    def set_unguarded_transition(self, st_idx, trn_idx):
        st = self.states[st_idx]
        transitions_new = st.transitions.copy()
        unguarded_transition = transitions_new.pop(trn_idx)
        transitions_new.append(unguarded_transition)
        st_new = st.with_transitions(transitions=transitions_new)
        states_new = self.states.copy()
        states_new[st_idx] = st_new
        return dataclasses.replace(self, states=states_new)


class ValidateAST(ast.NodeVisitor):
    """Validates the AST of Guard/Action strings."""

    def __init__(self, parsed_ast):
        # this will likely have to be expanded, but it's a good start.
        self.allowed_ast_classes = [
            ast.Module,
            ast.Assign,
            ast.Expr,
            ast.Store,
            ast.Compare,
            ast.BinOp,
            ast.BoolOp,
            ast.UnaryOp,
            ast.Load,
            ast.Subscript,  # for indexing arrays
            # data storage
            ast.Constant,
            ast.Name,
            ast.List,
            # comparison
            ast.Gt,
            ast.GtE,
            ast.Lt,
            ast.LtE,
            ast.Eq,
            ast.In,
            ast.NotIn,
            # arithmetic
            ast.Add,
            ast.Mult,
            ast.Sub,
            ast.Div,
            ast.AugAssign,
            ast.USub,
            # bool op
            ast.And,
            ast.Or,
            ast.Not,
        ]

        self.allowed = True
        self.generic_visit(parsed_ast)

    def generic_visit(self, node):
        if type(node) not in self.allowed_ast_classes:
            self.allowed = False
            return
        ast.NodeVisitor.generic_visit(self, node)


class StateMachine(LeafSystem):
    """Finite State Machine similar to Mealy Machine.
    https://en.wikipedia.org/wiki/Mealy_machine

    The state machine can be executed either periodically or by zero_crossings.

    Each state as 0 or more exit transitions. These are prioritized such that
    when 2 exits are simultaneously valid, the higher priority is executed.
    It is not allowed for a state to have more than one exit transition with no
    guard. Guardless exits only make sense in the periodic case.

    Each transitions may have 0 or more actions. Each action is a python
    statement that modifies the value of an output. When a transitions is executed
    (i.e. it's guard evaluates to true), its actions are then processed.

    If 'time' is needed for guards or actions, pass 'time' in from clock block.

    Whether executed periodically or by zero_crossings, the states are constant
    between transitions executions.
    In the zero_crossing case, all guards for transitions exiting the current
    state are continuously checked, and if any 'triggers', then the earlist point
    in time that any guard becomes true is determined, the actions of the earliest
    (and highest priority if multiple trigger simultaneously) guard are executed at
    that time, and the simulation continues afterwards.

    Input ports:
        User specified.

    Output ports:
        User specified.

    Parameters:
        dt:
            Either Float or None.
            When not None, state machine is executed periodically.
            When None, the transitions are monitored by zero_crossing
            events.
    """

    def __init__(
        self,
        sm_data: StateMachineData,
        inputs: List[str] = None,  # [name]
        outputs: List[str] = None,  # [name]
        dt=None,
        time_mode: str = "agnostic",
        name: str = None,
        ui_id: str = None,
        **kwargs,
    ):
        super().__init__(name=name, ui_id=ui_id)

        if time_mode not in ["discrete", "agnostic"]:
            raise BlockInitializationError(
                f"Invalid time mode '{time_mode}' for PythonScript block", system=self
            )

        if time_mode == "discrete" and dt is None:
            raise BlockInitializationError(
                "When in discrete time mode, dt is required for block", system=self
            )

        self._sm = sm_data
        self.time_mode = time_mode
        _is_periodic = time_mode == "discrete"

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        elif isinstance(outputs, dict):
            outputs = list(outputs.keys())

        self._validate_sm_data()

        # delcare inputs
        self._input_names = inputs
        for name in inputs:
            self.declare_input_port(name)

        self._output_names = outputs

        # Create the default discrete state values
        self._create_discrete_state_type(include_state_idx=_is_periodic)
        default_values = self._create_initial_discrete_state(
            include_state_idx=_is_periodic
        )
        self.declare_discrete_state(default_value=default_values, as_array=False)

        # Declare output ports for each state variable
        def _make_output_callback(o_port_name):
            def _output(time, state, *inputs, **parameters):
                return getattr(state.discrete_state, o_port_name)

            return _output

        for o_port_name in outputs:
            self.declare_output_port(
                _make_output_callback(o_port_name),
                name=o_port_name,
                prerequisites_of_calc=[DependencyTicket.xd],
                requires_inputs=False,
            )

        if _is_periodic:
            # delcare the periodic update event
            self.declare_periodic_update(
                self._discrete_update,
                period=dt,
                offset=dt,
            )
        else:
            # wrap the callback generation so that they do not get overwritten
            # in subsequent calls to declare_zero_crossing()
            def _make_guard_callback(t):
                def _guard(_time, state, *inputs, **parameters):
                    # Inputs are in order of port declaration, so they match `self._input_names`
                    local_env = dict(zip(self._input_names, inputs))
                    # get the values of the outputs as they are presently.
                    local_env.update(state.discrete_state._asdict())
                    # we do this so that when a guard goes False-True,
                    # is creates a zero-crossing that can be localized in time.
                    g = jnp.where(eval(t.guard, {}, local_env), 1.0, -1.0)
                    return g

                return _guard

            def _make_reset_callback(t):
                def _reset(_time, state, *inputs, **p):
                    # Inputs are in order of port declaration, so they match `self._input_names`
                    local_env = dict(zip(self._input_names, inputs))
                    # get the values of the outputs as they are presently.
                    local_env.update(state.discrete_state._asdict())
                    updated_outputs = self._exec_actions(t.actions, local_env)
                    return state.with_discrete_state(
                        value=self.DiscreteStateType(**updated_outputs)
                    )

                return _reset

            # declare zero-crossing driven events and mode
            self.declare_default_mode(self._sm.intial_state)
            self.declare_mode_output()
            for st_idx, st in self._sm.states.items():
                for t in st.transitions:
                    self.declare_zero_crossing(
                        guard=_make_guard_callback(t),
                        reset_map=_make_reset_callback(t),
                        direction="negative_then_non_negative",  # we only care when the guard transitions False->True
                        start_mode=st_idx,
                        end_mode=t.dst,
                    )

    def _raise_invalid_code_error(self, invalid_code):
        if not invalid_code:
            return

        msg_str_list = [f"StateMachine {self.name} has some guard/actions errors:\n"]
        for st_name, data in invalid_code.items():
            if data["guards"] or data["actions"]:
                msg_str_list.append(
                    f"State '{st_name}' exit transitions have the following invalid entries:\n"
                )
            if data["guards"]:
                msg_str_list.append("\tguards:\n")
                for g in data["guards"]:
                    msg_str_list.append(f"\t\t{g}\n")
            if data["actions"]:
                msg_str_list.append("\tactions:\n")
                for a in data["actions"]:
                    msg_str_list.append(f"\t\t{a}\n")
        msg = "".join(msg_str_list)
        raise StaticError(message=msg, system=self)

    def _validate_sm_transition(
        self,
        st,
        st_idx,
        st_idx_list,
        trn_idx,
        t,
        unguarded_idx,
        invalid_guards,
        invalid_actions,
    ):
        # validate destination
        if t.dst not in st_idx_list:
            # this can only happen due to manual error when creating sm_data from python
            msg = f"StateMachine {self.name} state[{st.name},{st_idx}] has exit transition with invalid destination index."
            raise StaticError(message=msg, system=self)

        # validate guard
        if not t.guard:
            # guard cannot be empty string. user must intentionally enter empty string to get here.
            msg = f'StateMachine {self.name} state[{st.name},{st_idx}] unguarded transitions must have guard set to "True".'
            raise StaticError(message=msg, system=self)
        elif not isinstance(t.guard, str):
            invalid_guards.append(t.guard)
        elif t.guard == "True":
            if unguarded_idx is not None:
                msg = f"StateMachine {self.name} state[{st.name},{st_idx}] has more than one unguarded exit transition."
                raise StaticError(message=msg, system=self)
            else:
                unguarded_idx = trn_idx
        elif t.guard == "False":
            # "False" fails ast.parse() but is valid
            pass
        else:
            # check that guard string is valid python code and has no nefarious elements
            try:
                guard_check = ValidateAST(ast.parse(t.guard))
                if not guard_check.allowed:
                    invalid_guards.append(t.guard)
            except Exception:
                invalid_guards.append(t.guard)

        # check that actions AST has no nefarious elements
        for a in t.actions:
            # check that action strings is valid python code and has no nefarious elements
            try:
                guard_check = ValidateAST(ast.parse(a))
                if not guard_check.allowed:
                    invalid_actions.append(a)
            except Exception:
                invalid_actions.append(a)

        return unguarded_idx, invalid_guards, invalid_actions

    def _validate_sm_data(self):
        # validate states
        st_idx_list = list(self._sm.states.keys())
        if not st_idx_list:
            msg = f"StateMachine {self.name} must have at least one state."
            raise StaticError(message=msg, system=self)

        if not all(isinstance(idx, int) for idx in st_idx_list):
            msg = f"StateMachine {self.name} state indices must be of type Int."
            raise StaticError(message=msg, system=self)

        if self._sm.intial_state not in st_idx_list:
            msg = f"StateMachine {self.name} initial_state index {self._sm.intial_state} does not correspond to a valid state index."
            raise StaticError(message=msg, system=self)

        # validate transitions
        invalid_code = {}
        for st_idx, st in self._sm.states.items():
            unguarded_idx = None
            invalid_guards = []
            invalid_actions = []
            for trn_idx, t in enumerate(st.transitions):
                (
                    unguarded_idx,
                    invalid_guards,
                    invalid_actions,
                ) = self._validate_sm_transition(
                    st,
                    st_idx,
                    st_idx_list,
                    trn_idx,
                    t,
                    unguarded_idx,
                    invalid_guards,
                    invalid_actions,
                )

            if invalid_guards or invalid_actions:
                invalid_code[st.name] = {
                    "guards": invalid_guards,
                    "actions": invalid_actions,
                }

            if unguarded_idx is not None:
                self._sm = self._sm.set_unguarded_transition(st_idx, unguarded_idx)

        self._raise_invalid_code_error(invalid_code)

    def _create_discrete_state_type(self, include_state_idx=True):
        if include_state_idx:
            # unique identifier for the state machine state variable.
            # FIXME: is there a better way? was not allowed to use leading underscore
            # in data class.
            st_name = "active_state_index"

            if st_name in self._input_names or st_name in self._output_names:
                msg = f"StateMachine {self.name} has port with same name as state {st_name}, this is not allowed."
                raise StaticError(message=msg, system=self)

            self._st_name = st_name

            attribs = [st_name] + self._output_names
        else:
            attribs = self._output_names
        # declare the discrete_state as a namedtuple
        self.DiscreteStateType = namedtuple("DiscreteStateType", attribs)

    def _create_initial_discrete_state(self, include_state_idx=True):
        # initial_outputs = self._exec_entry_point_actions(outputs)

        # execute the entry point actions
        initial_outputs = self._exec_actions(self._sm.inital_actions, {})

        # enforce that all outputs have been initialized
        initialized_output_names = set(initial_outputs.keys())
        all_output_names = set(self._output_names)
        uninitialized_output_names = all_output_names.difference(
            initialized_output_names
        )
        if uninitialized_output_names:
            msg = f"StateMachine {self.name} does not initialize the following output values in the entry point actions: {uninitialized_output_names}"
            raise BlockInitializationError(message=msg, system=self)

        # get and save the output dtype,shape for use in creating the jax.pure_callback
        self.output_port_params = {
            o_port_name: {"dtype": jnp.array(val).dtype, "shape": jnp.array(val).shape}
            for o_port_name, val in initial_outputs.items()
        }

        # prepare the initial state
        if include_state_idx:
            return self.DiscreteStateType(
                active_state_index=self._sm.intial_state,
                **initial_outputs,
            )

        return self.DiscreteStateType(**initial_outputs)

    def _filter_locals(self, local_env):
        # remove any bindings from locals that are not outputs.
        filtered_locals = {}
        for key, value in local_env.items():
            if key in self._output_names:
                filtered_locals[key] = value
        return filtered_locals

    def _exec_actions(self, actions, local_env):
        # execute actions, in context with inputs values, when done
        # all actions, filter out any variable bindings that do
        # correspond to outputs, then repack as dict of jnp.arrays
        for action in actions:
            exec(action, {}, local_env)

        local_env = self._filter_locals(local_env)
        updated_outputs = {k: jnp.array(v) for k, v in local_env.items()}
        return updated_outputs

    def _pure_callback(self, present_state_index, present_outputs, local_env):
        """
        The concept here is to evaluate all possible exit transitions from
        the active state, and then just return the updated (state,output values)
        for the successful transition. In the case no transitions are successful,
        we just return the present state and presen_outputs. Since we have ordered
        the possible transitions in order of priority, executing the lowest index
        successful trasition is 'correct' behavior.

        We do all this in a jax.pure_callback because despite many attempts,
        we could not find a jax compliant implementation to achieve the
        desired behavior.
        """
        # get the active state index, and the possible exit transitions
        present_state_index = int(present_state_index)
        actv_trns = self._sm.states[present_state_index].transitions

        # evaluate the guard for each possible exit transition.
        evaluated_guards = [
            eval(transition.guard, {}, local_env) for transition in actv_trns
        ]

        if np.any(evaluated_guards):
            actv_trn = actv_trns[evaluated_guards.index(True)]
            new_state = actv_trn.dst
            new_outputs = self._exec_actions(actv_trn.actions, local_env)
            outputs = [np.array(new_outputs[k]) for k in self._output_names]
            retval = [np.array(new_state), outputs]
        else:
            outputs = [np.array(present_outputs[k]) for k in self._output_names]
            retval = [np.array(present_state_index), outputs]

        return retval

    def _discrete_update(self, _time, state: LeafState, *inputs, **params):
        # persent state index
        actv_state = state.discrete_state.active_state_index

        # Inputs are in order of port declaration, so they match `self._input_names`
        local_env = dict(zip(self._input_names, inputs))

        # build jax.pure_callback result_shape_dtypes
        # its a nested list like: [actv_st, [outp0, outp1, ... outpN]]
        result_shape_dtypes = [jax.ShapeDtypeStruct((), jnp.int64)]
        result_shape_dtypes_outps = []
        for var in self._output_names:
            port = self.output_port_params[var]
            result_shape_dtypes_outps.append(
                jax.ShapeDtypeStruct(port["shape"], np.dtype(port["dtype"]))
            )
        result_shape_dtypes.append(result_shape_dtypes_outps)

        # get the values of the outputs as they are presently.
        present_outputs = {
            key: value
            for key, value in state.discrete_state._asdict().items()
            if key not in {self._st_name}
        }
        local_env.update(present_outputs)

        update_data = jax.pure_callback(
            self._pure_callback,
            result_shape_dtypes,
            actv_state,
            present_outputs,
            local_env,
        )

        outputs_dict = {k: v for k, v in zip(self._output_names, update_data[1])}

        return self.DiscreteStateType(
            active_state_index=update_data[0],
            **outputs_dict,
        )
