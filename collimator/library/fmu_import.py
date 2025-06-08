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
basic FMU import block based on FMPy package and the example:
https://github.com/CATIA-Systems/FMPy/blob/main/fmpy/examples/custom_input.py
"""

from collections import namedtuple
from typing import TYPE_CHECKING

import jax

from ..framework.error import BlockRuntimeError
from ..framework import (
    LeafSystem,
    BlockInitializationError,
    DependencyTicket,
    parameters,
)
from ..logging import logger
from ..backend import io_callback, numpy_api as cnp
from ..lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    import fmpy
    from fmpy import fmi2
    from fmpy.model_description import ScalarVariable
else:
    fmpy = LazyLoader("fmpy", globals(), "fmpy")
    fmi2 = LazyModuleAccessor(fmpy, "fmi2")
    model_description = LazyModuleAccessor(fmpy, "model_description")
    ScalarVariable = LazyModuleAccessor(model_description, "ScalarVariable")

ValueReference = int


class ModelicaFMU(LeafSystem):
    # Should we pass parameter overrides via kwargs? Sounds like it could conflict
    # in some rare cases (eg. dt, name...). The corresponding definition in
    # block_interface.py is pretty fragile in this regard.
    def __init__(
        self,
        file_name,
        dt,
        name=None,
        input_names: list[str] = None,
        output_names: list[str] = None,
        parameters: dict = None,
        start_time: float = 0.0,
        **kwargs,
    ):
        """Load and execute an FMU for Co-Simulation.

        Args:
            file_name (str): path to FMU file
            dt (float): stepsize for FMU simulation
            name (str, optional): name of block
            input_names (list[str], optional): if set, only expose these inputs
            output_names (list[str], optional): if set, only expose these outputs
            parameters (dict, optional): dictionary of parameter overrides
            kwargs: ignored
        """
        try:
            super().__init__(name=name)
            self._init(
                file_name,
                dt,
                name=name or f"fmu_{self.system_id}",
                input_names=input_names,
                output_names=output_names,
                parameters=parameters,
                start_time=start_time,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize FMU block %s (%s): %s", name, self.system_id, e
            )
            raise BlockInitializationError(str(e), system=self)

    @parameters(static=["file_name"])
    def _init(
        self,
        file_name,
        dt,
        name: str,
        input_names: list[str] = None,
        output_names: list[str] = None,
        parameters: dict = None,
        start_time: float = 0.0,
    ):
        self.dt = dt

        # read the model description
        model_description = fmpy.read_model_description(file_name)

        # extract the FMU
        unzipdir = fmpy.extract(file_name)

        self.fmu = fmu = fmi2.FMU2Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName=name,
        )

        # initialize
        fmu.instantiate()
        # setup and set startTime before entering initialization mode per FMI 2.0.4 section 2.1.6.
        fmu.setupExperiment(startTime=start_time)
        # enter initialization mode before get/set params per FMI 2.0.4 section 4.2.4.
        fmu.enterInitializationMode()

        # collect the value references
        self.fmu_inputs: list[ValueReference] = []
        self.fmu_outputs: list[ValueReference] = []

        inputs_by_name: dict[str, ScalarVariable] = {}
        outputs_by_name: dict[str, ScalarVariable] = {}
        variable_by_id: dict[int, ScalarVariable] = {}

        # FIXME: we rely on the XML file here, but collimator uses a similar
        # JSON file with altered variable names.
        # TODO: implement support for parsing that file and mapping from
        # collimator json name to/from xml name properly.
        def _compatible_param_name(name):
            return name.replace(".", "_")

        for variable in model_description.modelVariables:
            if variable.causality == "input":
                variable_by_id[variable.valueReference] = variable
                inputs_by_name[variable.name] = variable
            elif variable.causality == "output":
                variable_by_id[variable.valueReference] = variable
                outputs_by_name[variable.name] = variable
            elif variable.causality == "parameter" and parameters is not None:
                compat_name = _compatible_param_name(variable.name)
                parameter_value = parameters.get(compat_name, None)
                if parameter_value is None:
                    continue

                logger.debug(
                    "Setting parameter #%d '%s' <%s>: %s %s",
                    variable.valueReference,
                    variable.name,
                    variable.type,
                    parameter_value,
                    type(parameter_value),
                )

                # Values at this point have been wrapped into np.ndarray of
                # shape () via wildcat's JSON parsing. Enumerations are ints.
                match variable.type:
                    case "Boolean":
                        parameter_value = bool(parameter_value)
                        fmu.setBoolean([variable.valueReference], [parameter_value])
                    case "Integer":
                        parameter_value = int(parameter_value)
                        fmu.setInteger([variable.valueReference], [parameter_value])
                    case "Real":
                        parameter_value = float(parameter_value)
                        fmu.setReal([variable.valueReference], [parameter_value])
                    case "String":
                        parameter_value = str(parameter_value)
                        fmu.setString([variable.valueReference], [parameter_value])
                    case "Enumeration":
                        parameter_value = int(parameter_value)
                        fmu.setInteger([variable.valueReference], [parameter_value])
                    case _:
                        # not implemented
                        raise BlockInitializationError(
                            f"Unsupported type for parameter {variable.name} in "
                            + f"FMU block {name}: {variable.type}",
                            system=self,
                        )

        # If input_names or output_names are set, we filter out the variables
        # exposed as I/O ports to match those. This so that the ports in model.json
        # actually match those in the FMU.
        # NOTE: Maybe this is unnecessarily complicated.
        if input_names is not None:
            for name in input_names:
                if name not in inputs_by_name:
                    raise BlockInitializationError(
                        f"Input port {name} found on the block { name} "
                        + f"but not found in FMU {file_name}",
                        system=self,
                    )
                variable = inputs_by_name[name]
                self.fmu_inputs.append(variable.valueReference)
                self.declare_input_port(name=variable.name)
        else:
            for name, variable in inputs_by_name.items():
                self.fmu_inputs.append(variable.valueReference)
                self.declare_input_port(name=name)

        if output_names is not None:
            for name in output_names:
                if name not in outputs_by_name:
                    raise BlockInitializationError(
                        f"Input port {name} found on the block { name} "
                        + f"but not found in FMU {file_name}",
                        system=self,
                    )
                variable = outputs_by_name[name]
                self.fmu_outputs.append(variable.valueReference)
        else:
            for name, variable in outputs_by_name.items():
                self.fmu_outputs.append(variable.valueReference)

        # exit initialization mode after get/set params per FMI 2.0.4 section 4.2.4.
        fmu.exitInitializationMode()

        # Declare a discrete state component for each of the output variables
        self._create_discrete_state_type(fmu, self.fmu_outputs, variable_by_id)

        # Create the default discrete state values
        default_values = {}

        for output_ref in self.fmu_outputs:
            variable = variable_by_id[output_ref]
            match variable.type:
                case "Boolean":
                    start_value = fmu.getBoolean([variable.valueReference])[0]
                case "Integer" | "Enumeration":
                    start_value = fmu.getInteger([variable.valueReference])[0]
                case "Real":
                    start_value = fmu.getReal([variable.valueReference])[0]
                case _:
                    raise NotImplementedError(
                        f"Unsupported type for output port {variable.name} in FMU: {variable.type}"
                    )
            default_values[variable.name] = start_value

        # Map the default values to array-like types so that they have shape and dtype
        default_state = jax.tree_util.tree_map(
            cnp.asarray, self.DiscreteStateType(**default_values)
        )
        self.declare_discrete_state(default_value=default_state, as_array=False)

        # Declare an output port for each of the output variables
        def _make_output_callback(o_port_name):
            def _output(time, state, *inputs, **parameters):
                return getattr(state.discrete_state, o_port_name)

            return _output

        for o_port_name in default_values:
            self.declare_output_port(
                _make_output_callback(o_port_name),
                name=o_port_name,
                prerequisites_of_calc=[DependencyTicket.xd],
                requires_inputs=False,
            )

        # The step function acts as a periodic update that will update all components
        # of the discrete state.
        def _step(time, state, *inputs):
            args = (time, state, *inputs)
            # Use the io_callback so that we can call the untraceable FMU object
            return io_callback(self.exec_step, default_state, *args)

        self.declare_periodic_update(
            _step,
            period=dt,
            offset=dt,
        )

    def _create_discrete_state_type(self, fmu, fmu_outputs, variables):
        self.state_names = [variables[output_ref].name for output_ref in fmu_outputs]
        self.DiscreteStateType = namedtuple("DiscreteState", self.state_names)

    def exec_step(self, time, state, *inputs, **parameters):
        # NOTE: We should get the fmu from the context in order to build a pure
        # function but it is very unlikely this would ever work with FMUs since
        # they have their own internal hidden state. More context here:
        # https://github.com/collimator-ai/collimator/pull/5330/files#r1419062533
        # Also look at that PR to see the previous implementation (it worked with
        # a single I/O port).

        try:
            fmu = self.fmu

            # Note: although it may appear that the order of operations below is
            # backwards, e.g. 1] get_outputs, 2] set_inputs, 3] step, this is
            # actually intentional.
            # Explanation by example assuming 1sec update intervals.
            # The reason get_outputs happens before set_inputs and 'step, is that
            # at t=0, the fmu outputs are already at t=0, so we can just read them.
            # Then, the fmu should get inputs at t=0, and use those to take a step
            # to t=1. The step operation, using inputs at t=0, puts the fmu in a
            # state where it outputs are now at t=1. This we cannot read them until
            # next update interval at t=1.

            # Retrieve the outputs
            fmu_out = fmu.getReal(self.fmu_outputs)
            # Match the outputs with their names in the discrete state
            xd = {name: value for name, value in zip(self.state_names, fmu_out)}

            # Set inputs
            fmu.setReal(self.fmu_inputs, list(inputs))
            # Advance the FMU in time
            fmu.doStep(currentCommunicationPoint=time, communicationStepSize=self.dt)

        except fmi2.FMICallException as e:
            logger.error(
                "Failed to run FMU block %s (%s): %s", self.name, self.system_id, e
            )
            raise BlockRuntimeError(str(e), system=self) from e

        xd = jax.tree_util.tree_map(cnp.asarray, xd)

        return self.DiscreteStateType(**xd)
