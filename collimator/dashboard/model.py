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

import json
import logging

from collimator.dashboard.serialization import model_json

from . import api

logger = logging.getLogger(__name__)


def get_reference_submodel(project_uuid: str, model_uuid: str) -> model_json.Model:
    """
    Retrieves the reference submodel for a given project UUID and model UUID.

    Args:
        project_uuid (str): The UUID of the project.
        model_uuid (str): The UUID of the model.

    Returns:
        model_json.Model: The reference submodel as a Model object.

    Raises:
        None

    """
    try:
        model_dict = api.get(f"/project/{project_uuid}/submodels/{model_uuid}")
    except api.CollimatorNotFoundError:
        return None
    model_json_str = json.dumps(model_dict)
    return model_json.Model.from_json(model_json_str)


def get_reference_submodel_by_name(
    project_uuid: str, model_name: str
) -> model_json.Model:
    """
    Retrieves the reference submodel for a given project UUID and model name.

    Args:
        project_uuid (str): The UUID of the project.
        model_name (str): The name of the model.

    Returns:
        model_json.Model: The reference submodel as a Model object.

    Raises:
        None

    """
    try:
        submodels = api.get(f"/project/{project_uuid}/submodels?name={model_name}")
    except api.CollimatorNotFoundError:
        return None

    if submodels["count"] > 0:
        submodel_uuid = submodels["submodels"][0]["uuid"]
        return get_reference_submodel(project_uuid, submodel_uuid)
    return None


def create_reference_submodel(
    project_uuid: str,
    model: model_json.Model,
    parameter_definitions: list[model_json.ParameterDefinition] = None,
    model_uuid: str = None,
):
    """
    Create a reference submodel.

    Args:
        project_uuid (str): The UUID of the project.
        model (model_json.Model): The model object.
        parameter_definitions (list[model_json.ParameterDefinition], optional): The list of parameter definitions. Defaults to an empty list.
        model_uuid (str, optional): The UUID of the model. Defaults to None.

    Returns:
        dict: The response from the API.
    """

    if parameter_definitions is None:
        parameter_definitions = []

    body = {
        "name": model.name,
        "diagram": model.diagram,
        "parameter_definitions": parameter_definitions,
        "submodels": model.subdiagrams,
    }
    if model_uuid:
        body["uuid"] = model_uuid

    return api.post(f"/project/{project_uuid}/submodels", body=body)


def update_reference_submodel(
    project_uuid: str,
    model_uuid: str,
    model: model_json.Model,
    edit_id: str,
    parameter_definitions: list[model_json.ParameterDefinition] = None,
):
    """
    Update a reference submodel in the dashboard.

    Args:
        project_uuid (str): The UUID of the project.
        model_uuid (str): The UUID of the submodel to be updated.
        model (model_json.Model): The updated model object.
        edit_id (str): The ID of the edit.
        parameter_definitions (list[model_json.ParameterDefinition], optional): The list of parameter definitions. Defaults to None.

    Returns:
        dict: The response from the API call.
    """
    body = {
        "name": model.name,
        "diagram": model.diagram,
        "submodels": model.subdiagrams,
        "edit_id": edit_id,
    }

    if parameter_definitions is not None:
        body["parameter_definitions"] = parameter_definitions

    return api.call(
        f"/project/{project_uuid}/submodels/{model_uuid}",
        "PUT",
        body=body,
    )


def get_model(model_uuid: str) -> model_json.Model:
    """
    Retrieves a model from the API based on its UUID.

    Args:
        model_uuid (str): The UUID of the model to retrieve.

    Returns:
        model_json.Model: The retrieved model object.

    Raises:
        None

    """
    try:
        model_dict = api.get(f"/models/{model_uuid}")
    except api.CollimatorNotFoundError:
        return None
    model_json_str = json.dumps(model_dict)
    return model_json.Model.from_json(model_json_str)


def create_model(project_uuid: str, model: model_json.Model):
    """
    Creates a new model in the dashboard.

    Args:
        project_uuid (str): The UUID of the project to which the model belongs.
        model (model_json.Model): The model object containing the details of the model.

    Returns:
        dict: The response from the API containing the details of the created model.
    """

    body = {
        "name": model.name,
        "diagram": model.diagram,
        "project_uuid": project_uuid,
        "configuration": model.configuration,
    }

    return api.post(
        "/models",
        body=body,
    )


def update_model(model_uuid: str, model: model_json.Model):
    """
    Update the specified model with the given model data.

    Args:
        model_uuid (str): The UUID of the model to be updated.
        model (model_json.Model): The updated model data.

    Returns:
        dict: The response from the API call.
    """

    body = {
        "configuration": model.configuration,
        "diagram": model.diagram,
        "name": model.name,
        "submodels": model.subdiagrams,
        "state_machines": model.state_machines,
        "version": model.version if model else 1,
    }

    if model.parameters is not None:
        body["parameters"] = model.parameters

    return api.call(
        f"/models/{model_uuid}",
        "PUT",
        body=body,
    )
