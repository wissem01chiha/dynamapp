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

from typing import TYPE_CHECKING, Any, Callable, Protocol
from uuid import uuid4

from collimator.logging import logger
from collimator.framework import Parameter


if TYPE_CHECKING:
    from collimator.framework import Diagram


class ReferenceSubdiagramProtocol(Protocol):
    def __call__(
        self, *args: Any, instance_name: str, parameters: dict[str, Any], **kwargs: Any
    ) -> "Diagram": ...


class ReferenceSubdiagram:
    # TODO: improve documentation here.
    _registry: dict[str, Callable[[Any], "Diagram"]] = {}
    _parameter_definitions: dict[str, list[Parameter]] = {}  # noqa: F821

    @classmethod
    def create_diagram(
        cls,
        ref_id: str,
        instance_name: str,
        *args,
        instance_parameters: dict[str, Any] = None,
        **kwargs,
    ) -> "Diagram":
        """
        Create a diagram based on the given reference ID and parameters.

        Note that for submodels we evaluate all parameters, there is no
        "pure" string parameters.

        Args:
            ref_id (str): The reference ID of the diagram.
            *args: Variable length arguments.
            instance_parameters (dict[str, Any], optional): The instance parameters for the diagram. Defaults to None.
                example: {"gain": 3.0}
            **kwargs: Keyword arguments.

        Returns:
            Diagram: The created diagram.

        Raises:
            ValueError: If the reference subdiagram with the given ref_id is not found.
        """
        if ref_id not in ReferenceSubdiagram._registry:
            raise ValueError(f"ReferenceSubdiagram with ref_id {ref_id} not found.")

        params_def = ReferenceSubdiagram.get_parameter_definitions(ref_id)

        default_params = {p.name: p for p in params_def}

        # override the default values with any 'modified' values.
        new_instance_parameters = {}
        if instance_parameters:
            for param_name, param in instance_parameters.items():
                if param_name not in default_params:
                    raise ValueError(
                        f"Parameter {param_name} not found in parameter definitions."
                    )
                new_instance_parameters[param_name] = Parameter(
                    name=param_name, value=param
                )

        all_params = {**default_params, **new_instance_parameters}

        diagram = ReferenceSubdiagram._registry[ref_id](
            *args,
            instance_name=instance_name,
            parameters=all_params,
            **kwargs,
        )

        diagram.ref_id = ref_id
        diagram.instance_parameters = set(new_instance_parameters.keys())

        for param in params_def:
            if param.name in new_instance_parameters:
                diagram.declare_dynamic_parameter(
                    param.name, new_instance_parameters[param.name]
                )
            else:
                diagram.declare_dynamic_parameter(param.name, param)

        return diagram

    @staticmethod
    def register(
        constructor: ReferenceSubdiagramProtocol,
        # FIXME: rename parameter_definitions to default_parameters
        parameter_definitions: list[Parameter] = None,  # noqa: F821
        ref_id: str = None,
    ) -> str:
        if ref_id is None:
            ref_id = str(uuid4())
        if parameter_definitions is None:
            parameter_definitions = []

        logger.debug("Registering ReferenceSubdiagram with ref_id %s", ref_id)
        if ref_id in ReferenceSubdiagram._registry:
            logger.debug(
                "ReferenceSubdiagram with ref_id %s already registered.",
                ref_id,
            )

        ReferenceSubdiagram._registry[ref_id] = constructor
        ReferenceSubdiagram._parameter_definitions[ref_id] = parameter_definitions

        return ref_id

    @staticmethod
    def get_parameter_definitions(
        ref_id: str,
    ) -> list[Parameter]:  # noqa: F821
        if ref_id not in ReferenceSubdiagram._parameter_definitions:
            return []
        return ReferenceSubdiagram._parameter_definitions[ref_id]
