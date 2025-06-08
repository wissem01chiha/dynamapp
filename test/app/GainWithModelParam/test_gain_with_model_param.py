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

import pytest
import collimator.testing as test

pytestmark = pytest.mark.app


def test_GainWithModelParam(request):
    """This model contains a constant block linked to a gain block
    that has its gain value set to "[0.0, a]" where "a" is a model parameter.
    """
    test.run(
        pytest_request=request,
        stop_time=0.1,
        model_json="model.json",
        check_only=True,
    )
