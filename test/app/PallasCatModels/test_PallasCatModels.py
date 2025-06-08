#!/bin/env pytest
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
import os
import shutil


def copy_to_workdir(srcdir, file_name, workdir):
    src = os.path.join(srcdir, file_name)
    dst = os.path.join(workdir, file_name)
    shutil.copyfile(src, dst)


@pytest.mark.skip(reason="for manual use only")
def test_pallascat_model(projdir: str = None):
    # get list of all files in the run/*model_name* directory of pallascat download
    model_files = [
        f for f in os.listdir(projdir) if os.path.isfile(os.path.join(projdir, f))
    ]
    print(f"model_files={model_files}")

    # create test paths so that we can copy the files to a workdir for wildcat
    # FIXME: this might need fixing (I didnt test because this is a personal/manual check) -- @jp
    test_paths = test.get_paths(
        None, testdir_=__file__, test_name_=os.path.basename(projdir)
    )
    print(f"test_paths={test_paths}")

    # copy the files
    for model_file in model_files:
        print(f"model_file={model_file}")
        copy_to_workdir(projdir, model_file, test_paths["workdir"])

    # make wildcat look in workdir for json files
    test_paths["testdir"] = test_paths["workdir"]

    # run wildcat
    test.run(test_paths=test_paths)


if __name__ == "__main__":
    absdir = "/home/albert/git/collimator/src/"
    projdir = "tests/cmlc-vs-wildcat/workdir/Auto - Electric Vehicle-f0c3a89f-8958-4182-9ffb-20079e588e7e/run/Main - Class 2A Truck Conventional"
    projdir = "tests/cmlc-vs-wildcat/workdir/Hybrid Examples-2d8c997a-ca33-47e7-b534-bea03c606f53/run/Newton's Cradle"
    projdir = "tests/cmlc-vs-wildcat/workdir/Hybrid Examples-2d8c997a-ca33-47e7-b534-bea03c606f53/run/Double Bouncing Ball"
    projdir = "tests/cmlc-vs-wildcat/workdir/Discrete-Time Examples-3d16c368-8e7b-4048-b312-91ffe22aedac/run/Simple Counter"
    projdir = "tests/cmlc-vs-wildcat/workdir/Continuous-Time Examples-dfc1f667-6754-4270-bfaa-f14e8f5341f2/run/VanDerPol"
    projdir = "tests/cmlc-vs-wildcat/workdir/Auto - Electric Vehicle-f0c3a89f-8958-4182-9ffb-20079e588e7e/run/Main - Class 2A Truck Conventional"
    projdir = "tests/cmlc-vs-wildcat/workdir/Auto - Electric Vehicle-f0c3a89f-8958-4182-9ffb-20079e588e7e/run/Test Harness - Automatic Transmission"
    projdir = "tests/cmlc-vs-wildcat/workdir/Aero - Satellite-98a3530d-3b97-4909-b5f3-64666669363a/run/SingleSatellite"
    projdir = "tests/cmlc-vs-wildcat/workdir/Tutorial Project-616eedd9-9e37-4a80-a3e0-6ada3ea14bfb/run/Tutorial Model"
    projdir = "tests/cmlc-vs-wildcat/workdir/Robotics - PUMA560-d5455562-f2bb-47ea-afa0-2eb93f6e9235/run/puma560"
    # projdir = "tests/cmlc-vs-wildcat/workdir/DataBookUW - Chapters 9+-607a7ef1-1a77-4890-a34e-1b19e4c70063/run/CruiseControl"
    # projdir = "tests/cmlc-vs-wildcat/workdir/DataBookUW - Chapter 8 (continuous)-a92e261e-5329-49fa-9a39-55fcb3a0965a/run/CartPole_NoControl"
    # projdir = "tests/cmlc-vs-wildcat/workdir/Biomedical - Pacemaker-c7c21691-1f4f-455f-9380-d2c66a45796d/run/heart_with_DDD_pacemaker"
    # projdir = "tests/cmlc-vs-wildcat/workdir/DataBookUW - Chapter 8 (continuous)-a92e261e-5329-49fa-9a39-55fcb3a0965a/run/CartPole_KF"
    projdir = "tests/cmlc-vs-wildcat/workdir/Energy - Wind Turbine-a6baba87-ecbc-48a1-9614-f644e536c5e2/run/WindTurbine_model"
    # projdir = "tests/cmlc-vs-wildcat/workdir/Biomedical - Pacemaker-cb01128c-1785-400c-971b-b6a641fa8069/run/heart"
    # projdir = "tests/cmlc-vs-wildcat/workdir/Biomedical - Pacemaker-cb01128c-1785-400c-971b-b6a641fa8069/run/heart_with_DDD_pacemaker"
    # projdir = "tests/cmlc-vs-wildcat/workdir/Biomedical - Pacemaker-cb01128c-1785-400c-971b-b6a641fa8069/run/heart_with_VVI_pacemaker"
    # projdir = "tests/cmlc-vs-wildcat/workdir/DC Motor Controller (notebook)-5efda0a7-6028-4373-8987-44fd07953f7b/run/DC Motor PI"
    # projdir = "tests/cmlc-vs-wildcat/workdir/Aero - F16 Fighter Jet-cab06f5b-e918-4abd-9d22-14a605392e51/run/F16 Model"

    abs_projdir = absdir + projdir
    test_pallascat_model(abs_projdir)
