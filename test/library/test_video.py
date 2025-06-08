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

import os
from pathlib import Path
import typing as T

import numpy as np
import pytest

import collimator
import collimator.testing as test
from collimator.lazy_loader import LazyLoader
from collimator.library import (
    Gain,
    SignalDatatypeConversion,
    VideoSink,
    VideoSource,
    WhiteNoise,
)


cv2 = LazyLoader("cv2", globals(), "cv2")

if T.TYPE_CHECKING:
    import cv2


@pytest.mark.slow
def test_VideoSink(request):
    test_paths = test.get_paths(request)
    file_name = str(test_paths["workdir"] / "test_video.mp4")

    def _make_sink_diagram():
        builder = collimator.DiagramBuilder()

        source = builder.add(WhiteNoise(0.1, 3, shape=(480, 640, 3), name="source"))
        gain = builder.add(Gain(255.0, name="gain"))
        convert = builder.add(SignalDatatypeConversion(np.uint8, name="convert"))
        sink = builder.add(VideoSink(0.1, file_name, name="sink"))

        builder.connect(source.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], convert.input_ports[0])
        builder.connect(convert.output_ports[0], sink.input_ports[0])

        return builder.build()

    diagram = _make_sink_diagram()
    context = diagram.create_context()

    recorded_signals = {
        "frame_id": diagram["sink"].get_output_port("frame_id"),
    }
    results = collimator.simulate(
        diagram, context, (0.0, 1.0), recorded_signals=recorded_signals
    )

    assert results.time[-1] == 1.0
    assert results.outputs["frame_id"][0] == 0
    assert results.outputs["frame_id"][-1] == 10

    # Check that the video file was created
    assert os.path.exists(file_name)

    # Probe the video file
    cap = cv2.VideoCapture(file_name)
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret
    assert frame.shape == (480, 640, 3)
    cap.release()


def test_VideoSource():
    srcdir = Path(os.path.dirname(__file__)).absolute()

    def _make_source_diagram(no_repeat: bool):
        builder = collimator.DiagramBuilder()

        _source = builder.add(
            VideoSource(
                srcdir / "assets" / "test_video.mp4", no_repeat=no_repeat, name="source"
            )
        )

        return builder.build()

    # no repeat = True (no loop)

    diagram = _make_source_diagram(no_repeat=True)
    context = diagram.create_context()

    recorded_signals = {
        "frame_id": diagram["source"].get_output_port("frame_id"),
        "stopped": diagram["source"].get_output_port("stopped"),
    }

    results = collimator.simulate(
        diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
    )

    assert results.time[-1] == 2.0
    assert results.outputs["frame_id"][0] == 0
    assert results.outputs["frame_id"][-1] == 10
    assert results.outputs["stopped"][0] == 0
    assert results.outputs["stopped"][-1] == 1

    # no repeat = False (loop)

    diagram = _make_source_diagram(no_repeat=False)
    context = diagram.create_context()

    recorded_signals = {
        "frame_id": diagram["source"].get_output_port("frame_id"),
    }

    with pytest.raises(Exception):
        diagram["source"].get_output_port("stopped")

    results = collimator.simulate(
        diagram, context, (0.0, 2.4), recorded_signals=recorded_signals
    )

    assert results.time[-1] == 2.4
    assert results.outputs["frame_id"][0] == 0
    assert results.outputs["frame_id"][-1] == 2  # 2.4s == 11 frames x 2 + 2
