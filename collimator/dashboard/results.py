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

import concurrent.futures
from functools import lru_cache
import io

import numpy as np

from collimator.dashboard import api

from ..lazy_loader import LazyLoader

requests = LazyLoader("requests", globals(), "requests")


@lru_cache()
def _download(s3_url):
    return requests.get(s3_url).content


def get_signals(
    model_uuid, simulation_uuid, signals: list[str] = None
) -> dict[str, np.ndarray]:
    response = api.get(f"/models/{model_uuid}/simulations/{simulation_uuid}/signals")
    results = {}
    futures = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for s3_url in response["s3_urls"]:
            signal_name = s3_url.get("name").replace(".npy", "").replace(".npz", "")
            if signals is not None and signal_name not in signals:
                continue
            futures[signal_name] = executor.submit(_download, s3_url.get("url"))
        for name, future in futures.items():
            result = future.result()
            results[name] = np.load(io.BytesIO(result))

    return results
