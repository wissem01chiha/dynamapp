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

"""Test coordinate rotations blocks

Contains tests for:
- CoordinateRotation
- CoordinateRotationConversion
- RigidBody
"""

import pytest
import numpy as np

from scipy.spatial.transform import Rotation

import collimator
from collimator import library
from collimator.framework import ShapeMismatchError
from collimator.library.rotations import EULER_SEQ


@pytest.fixture
def v1_test():
    return np.array([1.0, 0.0, 0.0])


@pytest.fixture
def rpy_test():
    # Euler angles for a 90 degree pitch-up rotation.
    # This is a _minus_ 90-degree right-hand rotation about
    # the y-axis.  Equivalently, in the "XYZ" intrinsic convention,
    # this is a +90-degree roll followed by a +90-degree yaw.
    # This will convert a vector pointing in the +x direction
    # to one pointing in the +Z direction.
    return np.array([np.pi / 2, 0.0, np.pi / 2])


@pytest.fixture
def scipy_rotation(rpy_test):
    return Rotation.from_euler(EULER_SEQ, rpy_test)


@pytest.fixture
def q_test(scipy_rotation):
    return scipy_rotation.as_quat()


@pytest.fixture
def dcm_test(scipy_rotation):
    return scipy_rotation.as_matrix()


@pytest.fixture
def v2_test(v1_test, scipy_rotation):
    return scipy_rotation.apply(v1_test)


class TestCoordinateRotation:
    def test_fixture(self, v1_test, v2_test):
        # Check that the rotations are working as expected
        assert np.allclose(v2_test, [0.0, 0.0, 1.0])

    def test_quat(self, v1_test, q_test, v2_test, rpy_test):
        builder = collimator.DiagramBuilder()

        v_in = builder.add(library.Constant(v1_test, name="v_in"))
        crot = builder.add(
            library.CoordinateRotation(
                rotation_type="quaternion",
                quaternion=q_test,
                enable_external_rotation_definition=False,
            )
        )

        builder.connect(v_in.output_ports[0], crot.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        v_out = crot.output_ports[0].eval(context)
        assert np.allclose(v_out, v2_test)

    def test_dcm(self, v1_test, dcm_test, v2_test):
        builder = collimator.DiagramBuilder()

        v_in = builder.add(library.Constant(v1_test, name="v_in"))
        crot = builder.add(
            library.CoordinateRotation(
                rotation_type="DCM",
                direction_cosine_matrix=dcm_test,
                enable_external_rotation_definition=False,
                inverse=False,
            )
        )

        builder.connect(v_in.output_ports[0], crot.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        v_out = crot.output_ports[0].eval(context)
        assert np.allclose(v_out, v2_test)

    def test_euler(self, v1_test, rpy_test, v2_test):
        builder = collimator.DiagramBuilder()

        v_in = builder.add(library.Constant(v1_test, name="v_in"))
        crot = builder.add(
            library.CoordinateRotation(
                rotation_type="roll_pitch_yaw",
                roll_pitch_yaw=rpy_test,
                enable_external_rotation_definition=False,
                inverse=False,
            )
        )

        builder.connect(v_in.output_ports[0], crot.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        v_out = crot.output_ports[0].eval(context)
        assert np.allclose(v_out, v2_test)

    def test_input_shape_mismatch(self):
        builder = collimator.DiagramBuilder()

        v_in = builder.add(library.Constant(np.zeros((4,)), name="v_in"))
        crot = builder.add(
            library.CoordinateRotation(
                rotation_type="quaternion",
                quaternion=np.zeros(4),
                enable_external_rotation_definition=False,
            )
        )

        builder.connect(v_in.output_ports[0], crot.input_ports[0])

        system = builder.build()

        with pytest.raises(ShapeMismatchError):
            system.create_context()


class TestCoordinateRotationConversion:
    def test_euler_to_quat(self, rpy_test, q_test):
        builder = collimator.DiagramBuilder()

        euler = builder.add(library.Constant(rpy_test, name="euler"))
        quat = builder.add(
            library.CoordinateRotationConversion(
                "RPY_to_quaternion",
                name="coord_rot_conv",
            )
        )

        builder.connect(euler.output_ports[0], quat.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        q_out = quat.output_ports[0].eval(context)
        assert np.allclose(q_out, q_test)

    def test_quat_to_euler(self, rpy_test, q_test):
        builder = collimator.DiagramBuilder()

        quat = builder.add(library.Constant(q_test, name="quat"))
        euler = builder.add(
            library.CoordinateRotationConversion(
                "quaternion_to_RPY",
                name="coord_rot_conv",
            )
        )

        builder.connect(quat.output_ports[0], euler.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        euler_out = euler.output_ports[0].eval(context)
        assert np.allclose(euler_out, rpy_test)

    def test_euler_to_dcm(self, rpy_test, dcm_test):
        builder = collimator.DiagramBuilder()

        euler = builder.add(library.Constant(rpy_test, name="euler"))
        dcm = builder.add(
            library.CoordinateRotationConversion(
                "RPY_to_DCM",
                name="coord_rot_conv",
            )
        )

        builder.connect(euler.output_ports[0], dcm.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        dcm_out = dcm.output_ports[0].eval(context)
        assert np.allclose(dcm_out, dcm_test)

    def test_dcm_to_euler(self, rpy_test, dcm_test):
        builder = collimator.DiagramBuilder()

        dcm = builder.add(library.Constant(dcm_test, name="dcm"))
        euler = builder.add(
            library.CoordinateRotationConversion(
                "DCM_to_RPY",
                name="coord_rot_conv",
            )
        )

        builder.connect(dcm.output_ports[0], euler.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        euler_out = euler.output_ports[0].eval(context)
        assert np.allclose(euler_out, rpy_test)

    def test_quat_to_dcm(self, q_test, dcm_test):
        builder = collimator.DiagramBuilder()

        quat = builder.add(library.Constant(q_test, name="quat"))
        dcm = builder.add(
            library.CoordinateRotationConversion(
                "quaternion_to_DCM",
                name="coord_rot_conv",
            )
        )

        builder.connect(quat.output_ports[0], dcm.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        dcm_out = dcm.output_ports[0].eval(context)
        assert np.allclose(dcm_out, dcm_test)

    def test_dcm_to_quat(self, q_test, dcm_test):
        builder = collimator.DiagramBuilder()

        dcm = builder.add(library.Constant(dcm_test, name="dcm"))
        quat = builder.add(
            library.CoordinateRotationConversion(
                "DCM_to_quaternion",
                name="coord_rot_conv",
            )
        )

        builder.connect(dcm.output_ports[0], quat.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        quat_out = quat.output_ports[0].eval(context)
        assert np.allclose(quat_out, q_test)

    def test_input_shape_mismatch(self):
        builder = collimator.DiagramBuilder()

        v_in = builder.add(library.Constant(np.zeros((2,)), name="q_in"))
        crc = builder.add(
            library.CoordinateRotationConversion(
                conversion_type="quaternion_to_RPY",
            )
        )

        builder.connect(v_in.output_ports[0], crc.input_ports[0])

        system = builder.build()

        with pytest.raises(ShapeMismatchError):
            system.create_context()


class TestRigidBody:
    def _make_diagram(self, **rb_kwargs):
        rb = library.RigidBody(**rb_kwargs)
        a_B = library.Constant(np.zeros(3), name="a_B")
        tau_B = library.Constant(np.zeros(3), name="tau_B")

        builder = collimator.DiagramBuilder()
        builder.add(a_B, tau_B, rb)

        builder.connect(a_B.output_ports[0], rb.force_input)
        builder.connect(tau_B.output_ports[0], rb.torque_input)

        return builder.build()

    def test_pure_translation(self):
        # Projectile motion with no rotation (6 DOF)

        m = 1.5
        x0 = np.array([0.5, 1.5, 2.5])
        v0 = np.array([1.0, 2.0, 3.0])
        g = np.array([0.0, 0.0, 9.81])  # NED frame

        system = self._make_diagram(
            initial_position=x0,
            initial_velocity=v0,
            initial_orientation=np.zeros(3),
            initial_angular_velocity=np.zeros(3),
            mass=m,
            gravity_vector=g,
            name="rb",
        )
        context = system.create_context()

        # Simulate for 1 second
        rb = system["rb"]
        recorded_signals = {
            "position": rb.position_output,
            "velocity": rb.velocity_output,
            "orientation": rb.orientation_output,
            "angular_velocity": rb.angular_velocity_output,
        }
        results = collimator.simulate(
            system,
            context,
            (0.0, 1.0),
            recorded_signals=recorded_signals,
        )

        t_sim = results.time
        x_sim = results.outputs["position"]
        v_sim = results.outputs["velocity"]
        rpy_sim = results.outputs["orientation"]
        omega_sim = results.outputs["angular_velocity"]

        # Analytic solution: quadratic projectile motion
        x_sol = x0[:, None] + v0[:, None] * t_sim + 0.5 * g[:, None] * t_sim**2
        v_sol = v0[:, None] + g[:, None] * t_sim

        assert np.allclose(x_sim.T, x_sol)
        assert np.allclose(v_sim.T, v_sol)

        assert np.allclose(rpy_sim, 0.0)
        assert np.allclose(omega_sim, 0.0)

    def test_pure_rotation(self):
        # Constant angular velocity applied to a body with no translation (6 DOF)
        # The initial state is an orientation 45 degrees between the x and z axes
        # in the body frame (45 degree pure pitch).  There is a constant angular
        # velocity about the orld-frame Z-axis.

        theta0 = np.pi / 4
        rpy0 = np.array([0.0, theta0, 0.0])

        # Rotate a constant inertial-frame yaw to the body frame
        yaw_rate = 1.0
        R0_BI = Rotation.from_euler(EULER_SEQ, rpy0).as_matrix()
        omega0_B = R0_BI @ np.array([0.0, 0.0, yaw_rate])

        # Rotating body with no translation (6 DOF)
        system = self._make_diagram(
            initial_position=np.zeros(3),
            initial_orientation=rpy0,
            initial_velocity=np.zeros(3),
            initial_angular_velocity=omega0_B,
            name="rb",
        )
        context = system.create_context()

        # Simulate
        rb = system["rb"]
        recorded_signals = {
            "position": rb.position_output,
            "velocity": rb.velocity_output,
            "orientation": rb.orientation_output,
            "angular_velocity": rb.angular_velocity_output,
        }
        tf = 1.0
        results = collimator.simulate(
            system,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        t_sim = results.time
        assert np.allclose(t_sim[-1], tf)

        x_sim = results.outputs["position"]
        v_sim = results.outputs["velocity"]
        omega_sim = results.outputs["angular_velocity"]

        assert np.allclose(x_sim, 0.0)
        assert np.allclose(v_sim, 0.0)
        assert np.allclose(omega_sim, omega0_B)

        rpy_sim = results.outputs["orientation"]

        # The analytic solution for the orientation is just precession
        # about the world-frame z-axis.  This is equivalent to a linearly-
        # increasing yaw and then a constant 45-degree pitch.
        # The easiest way to test this is to check that the two rpy angles
        # transform a vector in the same way.
        rpy_sol = np.array([0.0, theta0, yaw_rate * tf]).T

        v1 = np.array([0.0, 0.0, 1.0])  # World-frame z-axis
        v2_sol = Rotation.from_euler(EULER_SEQ, rpy_sol).apply(v1)
        v2_sim = Rotation.from_euler(EULER_SEQ, rpy_sim[-1]).apply(v1)

        assert np.allclose(v2_sim, v2_sol)

    def test_12dof(self):
        # Projectile motion with rotation (12 DOF)

        theta0 = np.pi / 4
        rpy0 = np.array([0.0, theta0, 0.0])
        # Rotate a constant inertial-frame yaw to the body frame
        yaw_rate = 1.0
        rot_body_inertial = Rotation.from_euler(EULER_SEQ, np.flip(rpy0))
        omega0_B = rot_body_inertial.apply(np.array([0.0, 0.0, yaw_rate]))
        print(f"{omega0_B=}")

        m = 1.5
        x0_I = np.array([0.0, 0.0, 7.0])  # Inertial frame position
        # Pure z-motion, rotated to the body frame
        v0_I = np.array([0.0, 0.0, -3.0])
        v0_B = rot_body_inertial.apply(v0_I)
        g = np.array([0.0, 0.0, 9.81])

        system = self._make_diagram(
            initial_position=x0_I,
            initial_velocity=v0_B,
            initial_orientation=rpy0,
            initial_angular_velocity=omega0_B,
            mass=m,
            gravity_vector=g,
            name="rb",
        )
        context = system.create_context()

        # Simulate
        rb = system["rb"]
        recorded_signals = {
            "position": rb.position_output,
            "velocity": rb.velocity_output,
            "orientation": rb.orientation_output,
            "angular_velocity": rb.angular_velocity_output,
        }
        options = collimator.SimulatorOptions(
            max_minor_step_size=1e-2,
        )
        tf = 4.0
        results = collimator.simulate(
            system,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
            options=options,
        )

        # Check that the simulation finished
        t_sim = results.time
        assert np.allclose(t_sim[-1], tf)

        # The analytic solution for the orientation is the same as the
        # `pure_rotation` test case
        rpy_sim = results.outputs["orientation"]
        rpy_sol = np.array([0.0 * t_sim, theta0 + 0.0 * t_sim, yaw_rate * t_sim]).T

        v1 = np.array([0.0, 0.0, 1.0])  # World-frame z-axis
        v2_sol = Rotation.from_euler(EULER_SEQ, rpy_sol[-1]).apply(v1)
        v2_sim = Rotation.from_euler(EULER_SEQ, rpy_sim[-1]).apply(v1)
        assert np.allclose(v2_sim, v2_sol)

        # The body-frame angular velocity is constant
        omega_sim = results.outputs["angular_velocity"]
        assert np.allclose(omega_sim, omega0_B)

        # The analytic solution for the position in the inertial frame
        # is just the same as the `pure_translation` test case (parabolic)
        x_sim = results.outputs["position"]
        x_sol = x0_I[:, None] + v0_I[:, None] * t_sim + 0.5 * g[:, None] * t_sim**2
        assert np.allclose(x_sim.T, x_sol)

        # The body-frame velocity is just the inertial-frame projectile
        # solution rotated to the body frame.
        v_B_sim = results.outputs["velocity"]
        v_I_sol = (v0_I[:, None] + g[:, None] * t_sim).T
        v_B_sol = Rotation.from_euler(EULER_SEQ, rpy_sol).apply(v_I_sol)
        assert np.allclose(v_B_sim[-1], v_B_sol[-1])
