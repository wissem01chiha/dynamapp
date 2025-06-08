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
import json
import logging
import os
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from ..lazy_loader import LazyLoader

if TYPE_CHECKING:
    import requests
else:
    requests = LazyLoader("requests", globals(), "requests")


logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.INFO)
logging.getLogger("s3transfer").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)

__API_BASE_URL__ = os.environ.get(
    "COLLIMATOR_API_URL", "https://app.collimator.ai/api/v0"
)
__API_TOKEN__ = os.environ.get("COLLIMATOR_API_TOKEN", None)

__HEADERS__ = {
    "X-Collimator-API-Caller": "pycollimator",
    "X-Request-ID": str(uuid4()),
    "Accept": "application/json",
    "X-Collimator-API-Token": __API_TOKEN__,
}


class CollimatorApiError(Exception): ...


class CollimatorRetryableApiError(CollimatorApiError): ...


class CollimatorAuthenticationError(CollimatorApiError): ...


class CollimatorNotFoundError(CollimatorApiError): ...


def set_api_url(url: str):
    global __API_BASE_URL__
    __API_BASE_URL__ = url


def set_api_token(token: str):
    global __API_TOKEN__
    __API_TOKEN__ = token
    __HEADERS__["X-Collimator-API-Token"] = __API_TOKEN__


def _handle_error(response: "requests.Response", headers: dict[str, str] = None):
    request_id = (headers is not None) and headers.get("X-Request-ID") or None
    if response.status_code >= 400:
        if response.status_code == 401:
            logger.debug("API error 401, response: %s", response.text)
            raise CollimatorAuthenticationError(
                "Authentication failed. Please check your token or reload the page."
            )
        elif response.status_code == 504:
            logger.debug("API error 504, retrying...")
            raise CollimatorRetryableApiError
        elif response.status_code == 404:
            raise CollimatorNotFoundError
        else:
            logger.debug(
                "API error, status_code: %s, response: %s",
                response.status_code,
                response.text,
            )
            raise CollimatorApiError(
                f"API call failed (status: {response.status_code}). Support request ID: "
                f"{request_id}. Detailed error:\n{response.text}"
            )


def _convert_dataclasses_to_dict(d: Any):
    if isinstance(d, dict):
        return {k: _convert_dataclasses_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_convert_dataclasses_to_dict(v) for v in d]
    elif dataclasses.is_dataclass(d):
        return _convert_dataclasses_to_dict(dataclasses.asdict(d))
    else:
        return d


def get(*args, **kwargs):
    return call(*args, **kwargs, method="GET")


def post(*args, **kwargs):
    return call(*args, **kwargs, method="POST")


def put(*args, **kwargs):
    return call(*args, **kwargs, method="PUT")


def call(
    api,
    method,
    body: dict = None,
    headers: dict = None,
    body_type="json",
    response_type="json",
    retries=3,
):
    url = f"{__API_BASE_URL__}{api}"

    if not headers:
        headers = {}

    if "X-Collimator-API-Token" not in __HEADERS__:
        logger.warning(
            "No API token provided. Please set COLLIMATOR_API_TOKEN envionment variable."
        )

    headers = {**__HEADERS__, **headers}
    if hasattr(body, "to_dict"):
        body = body.to_dict()
    else:
        body = _convert_dataclasses_to_dict(body)
    timeout = (5, 30)
    try:
        if body is None:
            logger.debug("%s %s", method, url)
            response = requests.request(method, url, headers=headers, timeout=timeout)
        elif body_type == "json":
            logger.debug("%s %s %s", method, url, json.dumps(body, indent=2))
            headers["Content-Type"] = "application/json"
            response = requests.request(
                method, url, json=body, headers=headers, timeout=timeout
            )
        elif body_type == "files":
            logger.debug("%s %s %s", method, url, body)
            response = requests.request(
                method, url, files=body, headers=headers, timeout=timeout
            )
        _handle_error(response, headers)
    except CollimatorRetryableApiError as e:
        if retries > 0:
            logger.debug("%s retrying...", method)
            return call(
                api,
                method,
                body,
                headers,
                body_type,
                response_type,
                retries - 1,
            )
        raise e

    logger.debug(
        "method: %s, response: %s, status: %s",
        method,
        response.text,
        response.status_code,
    )
    if response.status_code not in (requests.codes["no_content"],):
        if response.headers.get("content-type") and response.headers[
            "content-type"
        ].strip().startswith("application/json"):
            return response.json()
        return response.text
