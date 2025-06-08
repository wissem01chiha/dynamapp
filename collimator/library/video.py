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

"""Implementation of the VideoSink and VideoSource blocks

Requires cv2 (OpenCV) to be installed. To install, run:
  pip install opencv-python-headless
"""

import os
from typing import TYPE_CHECKING

# import jax
import numpy as np

from collimator import LeafSystem
from collimator.framework.error import (
    BlockInitializationError,
    BlockRuntimeError,
    StaticError,
)
from collimator.framework.system_base import parameters
from collimator.lazy_loader import LazyLoader
from collimator.backend import io_callback, numpy_api as cnp
from collimator.backend.typing import Array
from collimator.logging import logdata, logger


if TYPE_CHECKING:
    import cv2
    from cv2 import VideoWriter

else:
    cv2 = LazyLoader("cv2", globals(), "cv2")


FPS_EPSILON = 1e-4


class VideoSink(LeafSystem):
    """Records RGB frames to a video file.

    Parameters:
        dt: Interval at which to record frames.
        file_name: Name of the video file to write to (optional).
    """

    @parameters(static=["dt", "file_name"])
    def __init__(self, dt: float, file_name: str, **kwargs):
        super().__init__(**kwargs)

        self.dt = dt
        self.fps = 1 / dt
        file_name = str(file_name)
        ext = ".mp4" if not file_name.endswith(".mp4") else ""
        self.file_name = file_name + ext
        self.writer: "VideoWriter" = None
        self.frame_id = 0

        self.declare_input_port("frame")

        def _io_cb(time, state, *inputs, **parameters) -> Array:
            return io_callback(self._video_cb, cnp.intx(0), time, inputs[0])

        self.declare_output_port(
            _io_cb,
            name="frame_id",
            requires_inputs=True,
            period=dt,
            offset=dt,
        )

    def _init_video(self, frame: Array):
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise StaticError(
                f"Input frame must be an RGB image, got invalid shape: {frame.shape}",
                system=self,
            )

        # A note on codecs:
        # vp9 (vp09) works in browsers, but it's a bit slow to encode
        # MPEG-4 (mp4v) is faster, but not supported in browsers
        # H264 (avc1) is supported but plagued with patents
        # av1 (AV01) broke my computer

        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)

        h, w, _ = frame.shape
        self.writer = cv2.VideoWriter(
            self.file_name,
            cv2.VideoWriter_fourcc(*"vp09"),
            self.fps,
            (w, h),
        )
        if not self.writer.isOpened():
            raise StaticError(
                f"Failed to open video file {self.file_name}",
                system=self,
            )

        logger.info("Writing video of size %sx%s to file: %s", w, h, self.file_name)

    def post_simulation_finalize(self) -> None:
        if self.writer is not None:
            self.writer.release()
        return super().post_simulation_finalize()

    def _video_cb(self, time: Array, frame: Array) -> Array:
        image = np.array(frame)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.writer is None:
            self._init_video(image)
        self.writer.write(image)

        # jax.debug.print(
        #     "Wrote frame {frame_id} to video file at time {time}",
        #     frame_id=self.frame_id,
        #     time=time,
        # )

        frame_id = self.frame_id
        self.frame_id += 1
        return frame_id


class VideoSource(LeafSystem):
    """Reads frames from a video file.

    Parameters:
        file_name: Name of the video file to read from.
        no_repeat: Whether to stop at the end of the video or loop back to the beginning.
    """

    @parameters(static=["file_name", "no_repeat"])
    def __init__(self, file_name: str, no_repeat=False, **kwargs):
        super().__init__(**kwargs)

        self.repeat = not no_repeat
        self.file_name = str(file_name)
        self.frame_id: np.intx = 0
        self.reached_end = False

        self.reader = cv2.VideoCapture(self.file_name)
        if not self.reader.isOpened():
            raise BlockInitializationError(
                f"Could not open video file '{self.file_name}'", system=self
            )

        self.width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.depth = 1 if bool(self.reader.get(cv2.CAP_PROP_MONOCHROME)) else 3
        self.fps = self.reader.get(cv2.CAP_PROP_FPS) or 30
        self.video_length = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Opened video file '%s' with size %sx%s, %s frames, %s fps",
            self.file_name,
            self.width,
            self.height,
            self.video_length,
            self.fps,
            **logdata(block=self),
        )

        self.last_frame = np.zeros(
            (self.height, self.width, self.depth), dtype=np.uint8
        )

        def _frame_cb(time, state, *inputs, **parameters) -> Array:
            def cb(time) -> Array:
                return self._source_cb(time)

            return io_callback(cb, self.last_frame, time)

        dt = 1 / self.fps
        self.declare_output_port(
            _frame_cb,
            name="frame",
            period=dt,
            offset=dt,
            requires_inputs=False,
        )

        def _frame_id_cb(time, state, *inputs, **parameters) -> Array:
            return io_callback(self._frame_id_cb, cnp.intx(0))

        self.declare_output_port(
            _frame_id_cb,
            name="frame_id",
            period=dt,
            offset=dt,
            default_value=cnp.intx(0),
            requires_inputs=False,
        )

        if not self.repeat:

            def _stopped_cb(time, state, *inputs, **parameters) -> Array:
                return io_callback(self._stopped_cb, cnp.bool(False))

            self.declare_output_port(
                _stopped_cb,
                name="stopped",
                period=dt,
                offset=dt,
                default_value=cnp.bool(False),
                requires_inputs=False,
            )

    def post_simulation_finalize(self) -> None:
        if self.reader is not None:
            self.reader.release()
        return super().post_simulation_finalize()

    def _source_cb(self, time: float) -> Array:
        if self.reached_end:
            return self.last_frame

        if not self.repeat and int(time * self.fps + FPS_EPSILON) >= self.video_length:
            self.reached_end = True
            return self.last_frame

        # jax.debug.print(
        #     "Reading frame {frame_id} to video file at time {time}",
        #     frame_id=self.reader.get(cv2.CAP_PROP_POS_FRAMES),
        #     time=time,
        # )

        self.frame_id = int(time * self.fps + FPS_EPSILON) % self.video_length
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

        ret, frame = self.reader.read()
        if not ret:
            if self.repeat:
                self.reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.reader.read()
            else:
                self.reached_end = True
                self.reader.release()
                self.reader = None
                return self.last_frame

        if not ret:
            raise BlockRuntimeError("Failed to read frame from video file", system=self)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame = frame
        return frame

    def _frame_id_cb(self) -> Array:
        return self.frame_id

    def _stopped_cb(self) -> Array:
        return self.reached_end
