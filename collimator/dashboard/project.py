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

import asyncio
import concurrent.futures
import os
import logging
import mimetypes
import time
from typing import Any, Callable, IO, AnyStr


from collimator.dashboard.serialization import (
    model_json,
    from_model_json,
    to_model_json,
    SimulationContext,
)
from collimator.dashboard.schemas import (
    FileSummary,
    ModelKind,
    ModelSummary,
    ProjectSummary,
)
from collimator.framework import Diagram, Parameter
from collimator.library import ReferenceSubdiagram
from collimator.simulation.types import SimulationResults

from . import api, model as model_api, uiprops, results, ensemble
from ..lazy_loader import LazyLoader

requests = LazyLoader("requests", globals(), "requests")

logger = logging.getLogger(__name__)


class InitScriptVariables:
    def __init__(self, variables: dict[str, Any]):
        self._variables = variables

    def get_variable(self, name: str) -> Parameter:
        var = self._variables.get(name)
        if var is None:
            raise KeyError(f"Variable '{name}' not found")
        return Parameter(name=name, value=var)

    def as_dict(self) -> dict[str, Any]:
        return self._variables


class Project:
    """
    Represents a project in the Collimator dashboard.
    """

    def __init__(
        self,
        summary: ProjectSummary,
        models: dict[str, model_json.Model],
        files: list[str],
        submodels: dict[str, str],
        init_scripts: dict[str, InitScriptVariables] = None,
    ):
        """
        Initialize a new Project instance.

        Args:
            summary (ProjectSummary): The summary of the project.
            models (dict[str, model_json.Model]): A dictionary of models associated with the project.
                The keys are the model names and the values are the json representations of the models.
            files (list[str]): A list of file paths associated with the project.
            submodels (dict[str, str]): A dictionary of submodels associated with the project.
                The keys are the submodel names and the values are the reference IDs of the submodels.
            init_scripts (dict[str, InitScriptVariables], optional): A dictionary of initialization scripts
                associated with the project. Defaults to None.

        """
        self._models = models
        self._submodels = submodels
        self._summary = summary
        self._files = files
        self._init_scripts = init_scripts or {}

    def get_model(self, name: str) -> SimulationContext:
        """
        Retrieves a model from the project.

        Args:
            name (str): The name of the model to retrieve.

        Returns:
            SimulationContext: The simulation context for the retrieved model.

        Raises:
            KeyError: If the specified model name is not found.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")

        model = self._models.get(name)
        init_script = None
        if model.configuration.workspace.init_scripts:
            init_script = model.configuration.workspace.init_scripts[0].file_name
        _globals = {}
        if init_script in self._init_scripts:
            _globals = self._init_scripts[init_script].as_dict()

        if model.diagram.nodes:
            logger.info('Loading model "%s"', model.name)
            try:
                return from_model_json.loads_model(model.to_json(), namespace=_globals)
            except BaseException as e:
                logger.error(
                    "Failed to load model %s: %s", model.name, e, exc_info=True
                )

    def create_submodel_instance(
        self,
        name: str,
        instance_name: str,
        instance_parameters: dict[str, Any] = None,
        **kwargs,
    ) -> Diagram:
        """
        Creates an instance of a submodel.

        Args:
            name (str): The name of the submodel.
            instance_name (str): The name of the instance.
            instance_parameters (dict[str, Any], optional): Parameters for the instance. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Diagram: The created submodel instance.

        Raises:
            KeyError: If the specified submodel is not found.
            ValueError: If the specified submodel is found but has a None reference ID.
        """
        if name not in self._submodels:
            raise KeyError(
                f"Submodel '{name}' not found. Available submodels: {list(self._submodels.keys())}"
            )
        ref_id = self._submodels.get(name)
        if ref_id is None:
            raise ValueError(f"Submodel '{name}' not found")
        return ReferenceSubdiagram.create_diagram(
            ref_id,
            instance_name,
            instance_parameters=instance_parameters,
            **kwargs,
        )

    def _save_single_submodel(
        self,
        model: model_json.Model,
        submodel_uuid: str = None,
    ):
        project_uuid = self.summary.uuid
        submodel = None
        if submodel_uuid:
            submodel = model_api.get_reference_submodel(project_uuid, submodel_uuid)
        if submodel is None:
            for submodel_name, ref_id in self._submodels.items():
                if ref_id == submodel_uuid:
                    submodel = model_api.get_reference_submodel_by_name(
                        project_uuid, submodel_name
                    )
                    break

        if submodel is None:
            logger.info("Creating submodel '%s' (%s)", model.name, submodel_uuid)
            response = model_api.create_reference_submodel(
                project_uuid,
                model,
                model.parameter_definitions,
                model_uuid=submodel_uuid,
            )
            submodel_uuid = response["uuid"]
            edit_id = response["edit_id"]
        else:
            edit_id = submodel.edit_id
            submodel_uuid = submodel.uuid
            uiprops.copy(submodel, model)
            model.name = submodel.name
            logger.info("Updating submodel '%s' (%s)", model.name, submodel_uuid)
        model_api.update_reference_submodel(
            project_uuid,
            submodel_uuid,
            model,
            edit_id,
            parameter_definitions=model.parameter_definitions,
        )
        return submodel_uuid

    def _save_submodel(self, diagram: Diagram):
        """
        Saves a submodel into the project.

        Args:
            project_uuid (str): The UUID of the project.
            diagram (Diagram): The diagram object representing the submodel.

        Returns:
            None
        """
        model, ref_submodels = to_model_json.convert(diagram)

        for ref_id, ref_submodel in ref_submodels.items():
            self._save_single_submodel(ref_submodel, submodel_uuid=ref_id)

        self._save_single_submodel(model, submodel_uuid=diagram.ref_id)

    def _save_model(
        self,
        diagram: Diagram,
        configuration: model_json.Configuration = None,
    ) -> str:
        """
        Updates or creates a model in the dashboard based on the provided diagram.

        Args:
            project_uuid (str): The UUID of the project to which the model belongs.
            diagram (Diagram): The diagram object representing the model.
            configuration (model_json.Configuration, optional): The configuration object for the model. Defaults to None.

        Returns:
            str: The UUID of the updated or created model.

        Raises:
            None
        """
        project_uuid = self.summary.uuid
        model_json, ref_submodels = to_model_json.convert(
            diagram,
            configuration=configuration,
        )

        for ref_id, ref_submodel in ref_submodels.items():
            self._save_single_submodel(ref_submodel, submodel_uuid=ref_id)

        model_uuid = diagram.ui_id

        model = None
        if model_uuid:
            model = model_api.get_model(model_uuid)

        if model is None:
            model = _get_model_by_name(project_uuid, diagram.name)
            model_uuid = model.uuid if model else None

        if model is None:
            response = model_api.create_model(project_uuid, model_json)
            model_uuid = response["uuid"]
            logger.info("Creating model '%s' (%s)", model_json.name, model_uuid)
        else:
            logger.info("Updating model '%s' (%s)", model_json.name, model_uuid)
            model_json.version = model.version
            uiprops.copy(model, model_json)

        model_api.update_model(model_uuid, model_json)

        return model_uuid

    def save_model(
        self, diagram: Diagram, configuration: model_json.Configuration = None
    ) -> str:
        """
        Save the given diagram as a model. If the diagram already exists, it will be updated.

        Args:
            diagram (Diagram): The diagram to be saved as a model.
            configuration (model_json.Configuration, optional): The configuration for the model. Defaults to None.

        Returns:
            The UUID of the saved model.
        """
        models = {m.name: m for m in self.summary.models}
        if diagram.name in models:
            diagram.ui_id = models[diagram.name].uuid
        return self._save_model(diagram, configuration=configuration)

    def save_submodel(
        self,
        constructor: Callable,
        name: str,
        default_parameters: list[Parameter] = None,
    ) -> str:
        """
        Saves a submodel with the given reference ID and name.

        Args:
            constructor (Callable): The constructor function for the submodel.
            name (str): The name of the submodel.
            default_parameters (list[Parameter], optional): A list of default parameters for the submodel. Defaults to None.

        Returns:
            str: The reference ID of the saved submodel.
        """
        submodel = model_api.get_reference_submodel_by_name(self.summary.uuid, name)
        ref_id = submodel.uuid if submodel else None
        ref_id = ReferenceSubdiagram.register(
            constructor, default_parameters, ref_id=ref_id
        )

        submodel = ReferenceSubdiagram.create_diagram(ref_id, name)
        self._save_submodel(submodel)
        self._submodels[submodel.name] = ref_id
        return ref_id

    @property
    def uuid(self) -> str:
        return self._summary.uuid

    @property
    def summary(self) -> ProjectSummary:
        return self._summary

    @property
    def init_scripts(self) -> dict[str, InitScriptVariables]:
        return self._init_scripts


def _get_model_by_name(project_uuid: str, model_name: str) -> model_json.Model:
    project_response = api.get(f"/projects/{project_uuid}")
    project_summary = _parse_project_summary(project_response)
    for model_summary in project_summary.models:
        if model_summary.name == model_name:
            model_uuid = model_summary.uuid
            return model_api.get_model(model_uuid)
    return None


def _get_reference_submodels(
    project_uuid: str, model: model_json.Model, memo: set[str]
) -> dict[str, model_json.Model]:
    """Finds submodels & HLBs referenced by a given model or submodel"""
    if model.uuid in memo:
        return {}
    memo.add(model.uuid)
    refs = {}
    for node in model.diagram.nodes:
        if node.type != "core.ReferenceSubmodel":
            continue
        if node.submodel_reference_uuid in memo:
            continue
        submodel = model_api.get_reference_submodel(
            project_uuid, node.submodel_reference_uuid
        )
        if submodel is None:
            logger.warning(
                "Could not find submodel for node '%s' (%s)",
                node.name,
                node.submodel_reference_uuid,
            )
            continue
        refs[node.submodel_reference_uuid] = submodel
        refs.update(_get_reference_submodels(project_uuid, submodel, memo))

    for diagram_uuid, diagram in model.subdiagrams.diagrams.items():
        if diagram.uuid in memo:
            continue
        group = model_json.Model(
            uuid=diagram_uuid,
            diagram=diagram,
            subdiagrams=model_json.Subdiagrams(diagrams={}, references={}),
            state_machines={},  # FIXME
            parameters={},
            parameter_definitions=[],
            configuration=None,
            name="",
        )
        refs.update(_get_reference_submodels(project_uuid, group, memo))

    return refs


def _parse_project_summary(project_summary: dict) -> ProjectSummary:
    models = []
    for model in project_summary.get("models", []):
        models.append(
            ModelSummary(
                uuid=model["uuid"],
                kind=ModelKind(model.get("kind", "Model")),
                name=model["name"],
            )
        )

    submodels = []
    for model in project_summary.get("reference_submodels", []):
        submodels.append(
            ModelSummary(
                uuid=model["uuid"],
                kind=ModelKind(model.get("kind", "Submodel")),
                name=model["name"],
            )
        )

    files = []
    for file in project_summary.get("files", []):
        files.append(
            FileSummary(
                uuid=file["uuid"],
                name=file["name"],
                status=file["status"],
            )
        )

    return ProjectSummary(
        uuid=project_summary["uuid"],
        title=project_summary["title"],
        models=models,
        files=files,
        reference_submodels=submodels,
    )


def _download_file(file: FileSummary, project_uuid: str, destination: str):
    response = api.get(f"/projects/{project_uuid}/files/{file.uuid}/download")
    logger.debug("Downloading %s to %s", response["download_link"], destination)
    resp = requests.get(response["download_link"])
    if resp.status_code != 200:
        logger.error(
            "Failed to download data file %s from project %s", file.name, project_uuid
        )
        return
    with open(destination, "wb") as f:
        f.write(resp.content)


def get_project_by_name(project_name: str) -> Project:
    """
    Retrieves a project by its name.

    Args:
        project_name (str): The name of the project to retrieve.

    Returns:
        Project: The project object.

    Raises:
        CollimatorNotFoundError: If the project with the specified name is not found.
        CollimatorApiError: If multiple projects with the same name are found.

    """

    response = api.get("/projects")
    results = []
    for project in response["projects"]:
        if project["title"] == project_name:
            results.append(project)

    user_profile = api.get("/user/profile")

    if len(results) == 0:
        raise api.CollimatorNotFoundError(f"Project '{project_name}' not found")
    elif len(results) > 1:
        uuids = []
        for p in results:
            if p["is_default"]:
                uuids.append(f"- {p['uuid']} (public project)")
            elif p["owner_uuid"] == user_profile["uuid"]:
                uuids.append(f"- {p['uuid']} (private project)")
            else:
                uuids.append(f"- {p['uuid']} (shared with me)")
        uuids = "\n".join(uuids)

        raise api.CollimatorApiError(
            f"Multiple projects found with name '{project_name}':\n{uuids}\n"
            "Please use get_project_by_uuid() instead."
        )

    project_uuid = results[0]["uuid"]
    return get_project_by_uuid(project_uuid)


def get_project_by_uuid(project_uuid: str) -> Project:
    """
    Retrieves a project with the given UUID and downloads its files.

    Args:
        project_uuid (str): The UUID of the project to retrieve.

    Returns:
        Project: The downloaded project, including its models and files.
    """

    logger.info("Downloading project %s...", project_uuid)

    project_response = api.get(f"/projects/{project_uuid}")
    project_summary = _parse_project_summary(project_response)

    # download project files
    files = []

    project_dir = os.getcwd()
    logger.info("Project dir: %s", project_dir)

    for file in project_summary.files:
        if file.status == "processing_completed":
            dst = os.path.join(project_dir, file.name)
            _download_file(file, project_uuid, dst)
            files.append(dst)
        else:
            logger.warning(
                "File %s is not ready to be downloaded (status: %s)",
                file.name,
                file.status,
            )

    # Must first register submodels
    submodels_response = api.get(f"/project/{project_uuid}/submodels")
    visited = set()
    ref_submodels: dict[str, model_json.Model] = {}
    submodels = {}

    for model_summary in submodels_response["submodels"]:
        submodel = model_api.get_reference_submodel(project_uuid, model_summary["uuid"])
        if submodel is None:
            logger.warning(
                "Could not find submodel %s (%s)",
                model_summary["name"],
                model_summary["uuid"],
            )
            continue
        ref_submodels.update(
            {
                model_summary["uuid"]: submodel,
                **_get_reference_submodels(project_uuid, submodel, visited),
            }
        )
    for model_summary in project_summary.models:
        model = model_api.get_model(model_summary.uuid)
        if model is None:
            logger.warning(
                "Could not find model %s (%s)", model_summary.name, model_summary.uuid
            )
            continue
        ref_submodels.update(_get_reference_submodels(project_uuid, model, visited))
    for submodel_uuid, submodel in ref_submodels.items():
        logger.info("Registering submodel %s", submodel.name)
        from_model_json.register_reference_submodel(submodel_uuid, submodel)
        submodels[submodel.name] = submodel_uuid

    # Load models
    models = {}
    init_scripts_vars = {}
    for model_summary in project_summary.models:
        if model_summary.kind == ModelKind.MODEL:
            model = model_api.get_model(model_summary.uuid)
            init_scripts = model.configuration.workspace.init_scripts
            if len(init_scripts) > 1:
                raise NotImplementedError("Only one init script is supported")
            elif len(init_scripts) == 1 and init_scripts[0]:
                filename = init_scripts[0].file_name
                init_script_path = os.path.join(project_dir, filename)
                if os.path.exists(init_script_path):
                    logger.info("Evaluating %s", init_scripts[0])
                    with open(init_script_path, "r") as f:
                        import numpy as np

                        _globals = {**globals(), "np": np}
                        exec(f.read(), _globals)
                        init_scripts_vars[filename] = InitScriptVariables(_globals)

            models[model_summary.name] = model

    return Project(
        summary=project_summary,
        models=models,
        files=files,
        submodels=submodels,
        init_scripts=init_scripts_vars,
    )


def create_project(name: str) -> Project:
    """
    Creates a new project with the given name.

    Args:
        name (str): The name of the project.

    Returns:
        ProjectSummary: The summary of the created project.
    """
    logger.info("Creating project %s...", name)
    response = api.call("/projects", "POST", body={"title": name})
    summary = _parse_project_summary(response)
    return Project(summary=summary, models={}, files=[], submodels={})


def get_or_create_project(name: str) -> Project:
    """
    Retrieves a project by its name or creates a new one if it doesn't exist.

    Args:
        name (str): The name of the project.

    Returns:
        ProjectSummary: The summary of the retrieved or created project.
    """
    try:
        return get_project_by_name(name)
    except api.CollimatorNotFoundError:
        return create_project(name)


def delete_project(project_uuid: str):
    """
    Deletes a project with the given UUID.

    Args:
        project_uuid (str): The UUID of the project to delete.
    """
    logger.info("Deleting project %s...", project_uuid)
    api.call(f"/projects/{project_uuid}", "DELETE")


def upload_file(project_uuid: str, name: str, fp: IO[AnyStr], overwrite=True):
    """
    Uploads a file to a project.

    Args:
        project_uuid (str): The UUID of the project.
        name (str): The name of the file.
        fp (IO[AnyStr]): The file pointer of the file to be uploaded.
        overwrite (bool, optional): Flag indicating whether to overwrite an existing file with the same name.
            Defaults to True.

    Returns:
        dict: A dictionary containing the summary of the uploaded file.

    Raises:
        api.CollimatorApiError: If the file upload fails.
    """

    mime_type, _ = mimetypes.guess_type(name)
    size = os.fstat(fp.fileno()).st_size

    logger.info(
        "Uploading file %s (type: %s, size: %d) to project %s...",
        name,
        mime_type,
        size,
        project_uuid,
    )
    body = {
        "name": name,
        "content_type": mime_type,
        "overwrite": overwrite,
        "size": size,
    }
    put_url_response = api.call(f"/projects/{project_uuid}/files", "POST", body=body)
    s3_presigned_url = put_url_response["put_presigned_url"]
    s3_response = requests.put(
        s3_presigned_url,
        headers={"Content-Type": mime_type},
        data=fp,
        verify=False,
    )
    if s3_response.status_code != 200:
        logger.error("s3 upload failed: %s", s3_response.text)
        raise api.CollimatorApiError(
            f"Failed to upload file {name} to project {project_uuid}"
        )
    file_uuid = put_url_response["summary"]["uuid"]
    process_response = api.call(
        f"/projects/{project_uuid}/files/{file_uuid}/process", "POST"
    )
    return process_response["summary"]


def upload_files(project_uuid: str, files: list[str], overwrite=True):
    """
    Uploads multiple files to a project.

    Args:
        project_uuid (str): The UUID of the project.
        files (list[str]): A list of file paths to be uploaded.
        overwrite (bool, optional): Flag indicating whether to overwrite existing files.
            Defaults to True.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file in files:
            with open(file, "rb") as fp:
                futures.append(
                    executor.submit(
                        upload_file, project_uuid, os.path.basename(file), fp, overwrite
                    )
                )
        for future in concurrent.futures.as_completed(futures):
            future.result()


def stop_simulation(model_uuid: str, simulation_uuid: str) -> dict:
    """
    Stops a running simulation.

    Args:
        model_uuid (str): The UUID of the model for which the simulation is running.
        simulation_uuid (str): The UUID of the simulation to stop.

    Returns:
        dict: The response from the API call.

    """
    return api.post(
        f"/models/{model_uuid}/simulations/{simulation_uuid}/events",
        body={"command": "stop"},
    )


class SimulationFailedError(api.CollimatorApiError):
    pass


async def simulate(
    model_uuid: str,
    timeout: int = None,
    ignore_cache: bool = False,
    parameters: dict[
        str,
        ensemble.SweepValues
        | ensemble.RandomDistribution
        | Parameter
        | float
        | int
        | bool
        | str,
    ] = None,
    num_simulations: int = 1,
    recorded_signals: list[str] = None,
) -> SimulationResults | list[SimulationResults]:
    """
    Runs a simulation for the specified model UUID.

    Args:
        model_uuid (str): The UUID of the model to run the simulation for.
        timeout (int, optional): The maximum time to wait for the simulation
            to complete, in seconds. Defaults to None.
        ignore_cache (bool, optional): Flag indicating whether to ignore the
            cache. Defaults to False.
        parameters (dict[str, SweepValues | RandomDistribution | Parameter], optional):
            A dictionary of parameters to sweep. Defaults to None.
        num_simulations (int, optional): The number of simulations to run,
            used in the case of monte-carlo. Defaults to 1.
        recorded_signals (list[str], optional): A list of signals to record. Must be
            specified in the case of ensemble simulations. Defaults to None.

    Returns:
        SimulationResults | list[EnsembleSimulationResults]: The results of the simulation.

    Raises:
        TimeoutError: If the simulation does not complete within the specified timeout.
        SimulationFailedError: If the simulation fails.

    """
    start_time = time.perf_counter()
    body = {"ignore_cache": ignore_cache}
    is_ensemble = False
    if parameters:
        has_random_distrib = any(
            isinstance(v, ensemble.RandomDistribution) for v in parameters.values()
        )
        has_sweep_values = any(
            isinstance(v, ensemble.SweepValues) for v in parameters.values()
        )
        is_ensemble = has_random_distrib or has_sweep_values

        if has_random_distrib and has_sweep_values:
            raise ValueError("Cannot have both sweep values and random distributions")

        body["parameters"] = {}

        model_parameter_sweeps = []
        for param_name, param_value in parameters.items():
            if param_name is None:
                raise ValueError("Parameter name cannot be None")

            if isinstance(param_value, ensemble.SweepValues):
                model_parameter_sweeps.append(
                    {
                        "parameter_name": param_name,
                        "sweep_expression": str(param_value.values),
                    }
                )
            elif isinstance(param_value, ensemble.RandomDistribution):
                model_parameter_sweeps.append(
                    {
                        "parameter_name": param_name,
                        "sweep_expression": f"np.random.{param_value.distribution}(**{param_value.parameters})",
                    }
                )
            elif isinstance(param_value, (float, int, bool, str)):
                body["parameters"][param_name] = model_json.Parameter(
                    value=str(param_value), is_string=False
                ).to_dict()
            else:
                expr, _ = param_value.value_as_api_param(
                    allow_param_name=False, allow_string_literal=False
                )
                json_param = model_json.Parameter(value=expr, is_string=False)
                body["parameters"][param_name] = json_param.to_dict()

    body["model_overrides"] = {}
    if recorded_signals:
        recorded_signals += ["time"]
        body["model_overrides"]["recorded_signals"] = {"signal_ids": recorded_signals}

    if is_ensemble:
        body["save_npz"] = True
        body["target"] = "ensemble"
        body["model_overrides"]["ensemble_config"] = {
            "model_parameter_sweeps": model_parameter_sweeps,
            "sweep_strategy": "all_combinations" if has_sweep_values else "monte_carlo",
        }
        if has_random_distrib:
            body["model_overrides"]["ensemble_config"]["num_sims"] = num_simulations

    summary = api.post(f"/models/{model_uuid}/simulations", body=body)

    # wait for simulation completion
    while summary["status"] not in ("completed", "failed"):
        await asyncio.sleep(1)
        logger.info("Waiting for simulation to complete...")
        summary = api.get(f"/models/{model_uuid}/simulations/{summary['uuid']}")
        if timeout is not None and time.perf_counter() - start_time > timeout:
            stop_simulation(model_uuid, summary["uuid"])
            raise TimeoutError

    logs = api.get(f"/models/{model_uuid}/simulations/{summary['uuid']}/logs")
    logger.info(logs)

    if summary["status"] == "failed":
        raise SimulationFailedError(summary["fail_reason"])

    signals = results.get_signals(model_uuid, summary["uuid"], signals=recorded_signals)

    if is_ensemble:
        ensemble_results = []
        for run_id, time_signal in signals["time"].items():
            params = run_id.split(" ")
            params = {c.split("=")[0]: c.split("=")[1] for c in params}
            ensemble_results.append(
                SimulationResults(
                    context=None,
                    time=time_signal,
                    outputs={s: signals[s][run_id] for s in signals if s != "time"},
                    parameters=params,
                )
            )
        return ensemble_results

    return SimulationResults(
        context=None,
        time=signals["time"],
        outputs=signals,
    )
