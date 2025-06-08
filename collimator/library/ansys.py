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

from collections import namedtuple

import jax
import jax.numpy as jnp

from ..framework import LeafSystem, parameters
from ..lazy_loader import LazyLoader
from ..backend import io_callback

pytwin = LazyLoader("pytwin", globals(), "pytwin")


class PyTwin(LeafSystem):
    @parameters(
        static=[
            "pytwin_file",
            "pytwin_config",
            "parameters",
            "inputs",
        ]
    )
    def __init__(
        self,
        pytwin_file,
        dt,
        pytwin_config=None,
        parameters=None,
        inputs=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dt = dt

        self.twin_model = pytwin.TwinModel(pytwin_file)
        if pytwin_config:
            self.twin_model.initialize_evaluation(json_config_filepath=pytwin_config)
        elif parameters or inputs:
            self.twin_model.initialize_evaluation(parameters=parameters, inputs=inputs)
        else:
            self.twin_model.initialize_evaluation()

        for input_field, _ in self.twin_model.inputs.items():
            self.declare_input_port(name=input_field)

        self.input_fields = list(self.twin_model.inputs.keys())
        self.output_fields = list(self.twin_model.outputs.keys())

        self.DiscreteState = namedtuple("DiscreteState", self.output_fields)

        self.default_state = self.DiscreteState(
            **jax.tree_util.tree_map(jnp.asarray, self.twin_model.outputs)
        )

        self.declare_discrete_state(default_value=self.default_state, as_array=False)

        self.declare_periodic_update(self._update, period=dt, offset=0.0)

        def _make_output_callback(output_field):
            def _output_callback(time, state, *inputs, **params):
                return getattr(state.discrete_state, output_field)

            return _output_callback

        for output_field, value in self.twin_model.outputs.items():
            self.declare_output_port(
                _make_output_callback(output_field),
                period=dt,
                offset=0.0,
                name=output_field,
                default_value=jnp.asarray(value),
                requires_inputs=False,
            )

    def _update(self, time, state, *inputs, **params):
        return io_callback(
            self._advance_twin_and_get_outputs, self.default_state, *inputs
        )

    def _advance_twin_and_get_outputs(self, *inputs):
        inputs_dict = dict(zip(self.input_fields, inputs))
        self.twin_model.evaluate_step_by_step(step_size=self.dt, inputs=inputs_dict)
        advanced_state = self.DiscreteState(
            **jax.tree_util.tree_map(jnp.asarray, self.twin_model.outputs)
        )
        return advanced_state
