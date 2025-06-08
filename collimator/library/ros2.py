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

"""ROS2 integration for Wildcat."""

import re
import threading
from typing import TYPE_CHECKING, Any
import numpy as np

from ..backend import numpy_api as cnp, io_callback
from ..framework import LeafSystem, parameters
from ..lazy_loader import LazyLoader
from ..logging import logger

rclpy = LazyLoader("rclpy", globals(), "rclpy")

if TYPE_CHECKING:
    import rclpy
    from rclpy.node import Node, Publisher, Subscription


_NODE_NAME_REGEX = re.compile(r"[\W_]+")


_ros2_init_count = 0
_ros2_lock = threading.Lock()


# Global init/shutdown for ROS2. We can't call rclpy.init() multiple times,
# so we try our best here to init and shutdown only once, across all threads.
def _ros2_init():
    global _ros2_init_count  # pylint: disable=global-statement
    with _ros2_lock:
        if _ros2_init_count == 0 and rclpy.ok() is False:
            rclpy.init()
        _ros2_init_count += 1
        return rclpy.ok()


def _ros2_shutdown():
    global _ros2_init_count  # pylint: disable=global-statement
    with _ros2_lock:
        _ros2_init_count -= 1
        if _ros2_init_count == 0:
            rclpy.shutdown()


def _getattr_path(obj: Any, attr: str, default: Any = None):
    """Returns the nested attribute of any object, using `.` as a separator.

    For instance, given this:
    class B:
        x = 1
    class A:
        b = B()
    a = A()
    _getattr_path(a, "b.x") == 1
    """
    if "." in attr:
        path = attr.split(".")
        subobj = getattr(obj, path[0], default)
        return _getattr_path(subobj, ".".join(path[1:]), default)
    return getattr(obj, attr, default)


def _setattr_path(obj: Any, attr: str, value: Any):
    """Same as getattr but for setting attributes."""
    if "." in attr:
        path = attr.split(".")
        subobj = getattr(obj, path[0])
        return _setattr_path(subobj, ".".join(path[1:]), value)
    return setattr(obj, attr, value)


def _attr2name(attr: str):
    """Helper for attribute (field) name to valid port name conversion."""
    return attr.replace(".", "_")


def _fixup_dtype(dtype: type | str) -> type:
    if dtype == "float" or dtype is None or dtype == float:
        return np.float64
    if dtype == "int" or dtype == int:
        return np.int64
    if dtype == "bool" or dtype == bool:
        return np.bool_
    if isinstance(dtype, str):
        dtype = getattr(np, dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown data type: {dtype}")
    if not isinstance(dtype, type):
        raise ValueError(
            f"Invalid data type (need str or type): {type(dtype)}: {dtype}"
        )
    return dtype


class Ros2Publisher(LeafSystem):
    """
    Ros2Publisher block can emit signals to a ROS2 topic, based on input signal data.
    """

    @parameters(static=["topic", "msg_type", "fields"])
    def __init__(
        self,
        dt: float,
        topic: str,
        msg_type: type,
        fields: dict[str, type],
        **kwargs,
    ):
        """
        Publish messages to a ROS2 topic.

        Args:
            dt: Period of the system, in both sim and real (ros2) time.
            topic: ROS2 topic to publish to. Eg. `/turtle1/cmd_vel`.
            msg_type: ROS2 message type, e.g. `Twist` from `geometry_msgs.msg`.
                      Unlike the corresponding UI parameter, this must be a Python
                      type object.
            fields: Ordered dictionary of default values to extract from the
                    received message. The keys are the full attribute path
                    (with dots) to the value in the message, and the values are
                    the default values. This is used to create the output ports
                    with valid data types. Use Python or Numpy data types, not JAX.

                    For instance, for a `geometry_msgs.msg.Twist` message, the
                    `fields` could be `{"linear.x": float, "angular.z": float}`.
        """

        super().__init__(**kwargs)
        self.logger = logger.getChild("Ros2Publisher:" + self.name_path_str)

        self.node: Node = None
        self.publisher: Publisher = None

        self.dt = dt
        self.topic = topic
        self.msg_type = msg_type
        self.fields = {field: _fixup_dtype(dtype) for field, dtype in fields.items()}

        self.declare_periodic_update(self._update, period=dt, offset=0.0)

        # Extract type & full attribute path from fields. Note that this
        # relies on the fact that the Python (3.7+) dict is ordered; The
        # order must match that of the input ports. Works well with JSON
        # because our I/O ports are ordered arrays.
        # This could likely be simplified / replaced with a Bus signal type.
        self.input_types = []  # [float, float]
        self.input_attr_path = []  # ["linear.x", "angular.z"]
        for msg_field_name, msg_field_type in self.fields.items():
            input_name = _attr2name(msg_field_name)
            self.declare_input_port(name=input_name)
            self.input_types.append(msg_field_type)
            self.input_attr_path.append(msg_field_name)

        self.pre_simulation_initialize()

    def __del__(self):
        self.post_simulation_finalize()

    def pre_simulation_initialize(self):
        if not _ros2_init():
            raise RuntimeError("ROS2 init failed")

        node_name = _NODE_NAME_REGEX.sub("_", self.name_path_str)
        rnd = np.random.randint(0, 1000)
        self.node = rclpy.create_node(f"collimator_{rnd}_" + node_name)
        self.publisher = self.node.create_publisher(
            self.msg_type, self.topic, qos_profile=10
        )

        self.logger.debug(
            "ROS2 publisher %s initialized with node: %s and publisher: %s",
            self.name_path_str,
            self.node,
            self.publisher,
        )

    def post_simulation_finalize(self) -> None:
        if self.node:
            self.logger.debug("ROS2 publisher %s clean up", self.name_path_str)
            self.node.destroy_publisher(self.publisher)
            self.publisher = None
            self.node.destroy_node()
            self.node = None
            _ros2_shutdown()

    def _update(self, time, state, *inputs, **params):
        return io_callback(self._publish_message, None, *inputs)

    def _publish_message(self, *inputs):
        msg = self.msg_type()

        for i, input_value in enumerate(inputs):
            value = self.input_types[i](input_value)
            _setattr_path(msg, self.input_attr_path[i], value)

        self.logger.debug("Publishing message to topic %s: %s", self.topic, msg)
        self.publisher.publish(msg)

        # Spin rclpy loop to ensure the message is sent. Also, sync the clocks
        # using dt. This is a bit of a hack for now until we have proper clock
        # synchronization.
        rclpy.spin_once(self.node, timeout_sec=self.dt)


class Ros2Subscriber(LeafSystem):
    """
    Ros2Subscriber block listens to messages over a ROS2 topic and outputs them as
    signals in collimator.
    """

    @parameters(static=["topic", "msg_type", "fields", "read_before_start"])
    def __init__(
        self,
        dt,
        topic: str,
        msg_type: type,
        fields: dict[str, type],
        read_before_start=True,
        **kwargs,
    ):
        """Subscribe to a ROS2 topic and extract message values to output ports.

        Args:
            dt: Period of the system, in both sim and real (ros2) time.
            topic: ROS2 topic to subscribe to. Eg. `/turtle1/pose`.
            msg_type: ROS2 message type, e.g. `Pose` from `turtlesim.msg`.
                      Unlike the corresponding UI parameter, this must be a Python
                      type object.
            fields: Ordered dictionary of default values to extract from the
                    received message. The keys are the full attribute path
                    (with dots) to the value in the message, and the values are
                    the default values. This is used to create the output ports
                    with valid data types. Use Python or Numpy data types, not JAX.

                    For instance, for a `geometry_msgs.msg.Twist` message, the
                    `fields` could be `{"linear.x": float, "angular.z": float}`.
            read_before_start: If True, the subscriber will read the first message
                    before the simulation starts. Otherwise, the initial outputs will
                    be 0.
        """

        super().__init__(**kwargs)
        self.logger = logger.getChild("Ros2Subscriber:" + self.name_path_str)

        self.node: Node = None
        self.subscription: Subscription = None
        self._last_msg = None

        if not _ros2_init():
            raise RuntimeError("ROS2 init failed")

        self.dt = dt
        self.msg_type = msg_type
        self.topic = topic
        self.fields = {field: _fixup_dtype(dtype) for field, dtype in fields.items()}
        self.read_before_start = read_before_start

        # Note: Not 100% sure this is absolutely valid, but it worked with JAX.
        # If somehow we aren't getting updates, we may need to create a cache index,
        # see custom.py. See _callback().
        self.declare_periodic_update(self._update, period=dt, offset=0.0)
        self.default_values = {field: dtype() for field, dtype in self.fields.items()}

        def _make_output_cb(field_name: str, dtype: type):
            def _output():
                last_msg = self._last_msg or self.default_values
                value = _getattr_path(last_msg, field_name)
                return dtype(value)

            def _io_cb(time, state, *inputs, **params):
                return io_callback(_output, cnp.asarray(_output()))

            return _io_cb

        for field, dtype in self.fields.items():
            self.declare_output_port(
                callback=_make_output_cb(field, dtype),
                name=_attr2name(field),
                prerequisites_of_calc=[],
                requires_inputs=False,
                period=dt,
                offset=0.0,
                default_value=self.default_values[field],
            )

        self.pre_simulation_initialize()

    def __del__(self):
        self.post_simulation_finalize()

    def pre_simulation_initialize(self):
        if not _ros2_init():
            raise RuntimeError("ROS2 init failed")

        node_name = _NODE_NAME_REGEX.sub("_", self.name_path_str)
        rnd = np.random.randint(0, 1000)
        self.node = rclpy.create_node(f"collimator_{rnd}_" + node_name)
        self.subscription = self.node.create_subscription(
            self.msg_type, self.topic, self._callback, qos_profile=10
        )
        self.logger.debug(
            "ROS2 subscriber %s initialized, listening on topic %s msg_type=%s",
            self.name_path_str,
            self.topic,
            self.msg_type,
        )

        if self.read_before_start:
            self._update_cb()

    def post_simulation_finalize(self) -> None:
        if self.node:
            self.logger.debug("ROS2 subscriber %s clean up", self.name_path_str)
            self.node.destroy_subscription(self.subscription)
            self.subscription = None
            self.node.destroy_node()
            self.node = None
            _ros2_shutdown()

    def _update(self, time, state, *inputs, **params):
        return io_callback(self._update_cb, None)

    def _update_cb(self):
        # This timeout does not seem to block the call
        rclpy.spin_once(self.node, timeout_sec=2.0)

    def _callback(self, msg):
        self.logger.debug("Received message on topic %s: %s", self.topic, msg)

        # This may be wrong because we're not cleanly using the cache
        # like in custom.py. But it works.
        self._last_msg = msg
