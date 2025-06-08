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

import dataclasses
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum


# TODO: these dataclasses should be generated from openapi.yaml spec
class ModelKind(StrEnum):
    MODEL = "Model"
    SUBMODEL = "Submodel"


@dataclasses.dataclass
class ModelSummary:
    uuid: str
    kind: ModelKind
    name: str


@dataclasses.dataclass
class FileSummary:
    uuid: str
    name: str  # url
    status: str


@dataclasses.dataclass
class ProjectSummary:
    uuid: str
    title: str
    models: list[ModelSummary]
    reference_submodels: list[ModelSummary]
    files: list[FileSummary]
