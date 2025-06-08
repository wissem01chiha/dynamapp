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

from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
from functools import partial

from ..backend import Rotation, numpy_api as cnp

from ..framework import (
    LeafSystem,
    DependencyTicket,
    ShapeMismatchError,
    parameters,
)
from ..framework.error import BlockParameterError, ErrorCollector

if TYPE_CHECKING:
    from ..backend.typing import Array

__all__ = [
    "CoordinateRotation",
    "CoordinateRotationConversion",
    "RigidBody",
]


# The convention for the roll-pitch-yaw Euler angle sequence, also called 1-2-3.
# This is the intrinsic (body-fixed) rotation sequence identifier for SciPy.
EULER_SEQ = "XYZ"


#
# Common rotations math functions
#
def q_inv(q):
    # UNUSED  Will we want this for RigidBody quaternion kinematics?
    """Computes the inverse of a quaternion.

    The inverse of a quaternion `q` is given by `q⁻¹ = q* / |q|²`, where `q*` is the
    conjugate of `q` and `|q|²` is the squared magnitude of `q`.

    Args:
        q: A 4-element array representing a quaternion.

    Returns:
        The inverse of `q`.
    """
    w, x, y, z = q
    return cnp.array([w, -x, -y, -z]) / cnp.dot(q, q)


def q_mult(p, q):
    # UNUSED.  Will we want this for RigidBody quaternion kinematics?
    """Quaternion product p * q"""
    P = cnp.array(
        [
            [p[0], -p[1], -p[2], -p[3]],
            [p[1], p[0], -p[3], p[2]],
            [p[2], p[3], p[0], -p[1]],
            [p[3], -p[2], p[1], p[0]],
        ]
    )
    return cnp.dot(P, q)


def euler_to_quat(roll_pitch_yaw):
    return Rotation.from_euler(EULER_SEQ, roll_pitch_yaw).as_quat()


def euler_to_dcm(roll_pitch_yaw):
    return Rotation.from_euler(EULER_SEQ, roll_pitch_yaw).as_matrix()


def dcm_to_quat(R):
    return Rotation.from_matrix(R).as_quat()


def dcm_to_euler(R):
    return Rotation.from_matrix(R).as_euler(EULER_SEQ)


def quat_to_dcm(q):
    return Rotation.from_quat(q).as_matrix()


def quat_to_euler(q):
    return Rotation.from_quat(q).as_euler(EULER_SEQ)


def euler_kinematics(roll_pitch_yaw):
    """Matrix that maps angular velocity in the body-fixed frame to Euler rates."""
    phi, theta, psi = roll_pitch_yaw
    cphi = cnp.cos(phi)
    sphi = cnp.sin(phi)
    ctheta = cnp.cos(theta)
    ttheta = cnp.tan(theta)

    # See Lewis, Johnson, Stevens Eq. 1.4-4.
    # That equation was derived for the 3-2-1 rotation sequence, but here
    # we use the 1-2-3 sequence.  Since a yaw-roll-pitch (3-2-1) sequence with Euler
    # angles (ϕ, θ, ψ) is equivalent to a roll-pitch-yaw (1-2-3) sequence with Euler
    # angles (-ϕ, -θ, -ψ), we can use the same kinematics matrix but with the signs
    # reversed compared to the book definition
    return cnp.array(
        [
            [1, sphi * ttheta, -cphi * ttheta],
            [0, cphi, sphi],
            [0, -sphi / ctheta, cphi / ctheta],
        ]
    )


class CoordinateRotation(LeafSystem):
    """Computes the rotation of a 3D vector between coordinate systems.

    Given sufficient information to construct a rotation matrix `C_AB` from orthogonal
    coordinate system `B` to orthogonal coordinate system `A`, along with an input
    vector `x_B` expressed in `B`-axes, this block will compute the matrix-vector
    product `x_A = C_AB @ x_B`.

    Note that depending on the type of rotation representation, this matrix may not be
    explicitly computed.  The types of rotations supported are Quaternion, Euler
    Angles, and Direction Cosine Matrix (DCM).

    By default, the rotations have the following convention:

    - __Quaternion:__ The rotation is represented by a 4-component quaternion `q`.
        The rotation is carried out by the product `p_A = q⁻¹ * p_B * q`, where
        `q⁻¹` is the quaternion inverse of `q`, `*` is the quaternion product, and
        `p_A` and `p_B` are the quaternion extensions of the vectors `x_A` and `x_B`,
        i.e. `p_A = [0, x_A]` and `p_B = [0, x_B]`.

    - __Roll-Pitch-Yaw (Euler Angles):__ The rotation is represented by the set of Euler angles
        ϕ (roll), θ (pitch), and ψ (yaw), in the "1-2-3" convention for intrinsic
        rotations. The resulting rotation matrix `C_AB(ϕ, θ, ψ)` is the same as the product of
        the three  single-axis rotation matrices `C_AB = Cz(ψ) * Cy(θ) * Cx(ϕ)`.

        For example, if `B` represents a fixed "world" frame with axes `xyz` and `A`
        is a body-fixed frame with axes `XYZ`, then `C_AB` represents a rotation from
        the world frame to the body frame, in the following sequence:

        1. Right-hand rotation about the world frame `x`-axis by `ϕ` (roll), resulting
            in the intermediate frame `x'y'z'` with `x' = x`.
        2. Right-hand rotation about the intermediate frame `y'`-axis by `θ` (pitch),
            resulting in the intermediate frame `x''y''z''` with `y'' = y'`.
        3. Right-hand rotation about the intermediate frame `z''`-axis by `ψ` (yaw),
            resulting in the body frame `XYZ` with `z = z''`.

    - __Direction Cosine Matrix:__ The rotation is directly represented as a
        3x3 matrix `C_AB`. The rotation is carried out by the matrix-vector product
        `x_A = C_AB @ x_B`.

    Input ports:
        (0): The input vector `x_B` expressed in the `B`-axes.

        (1): (if `enable_external_rotation_definition=True`) The rotation
            representation (quaternion, Euler angles, or cosine matrix) that defines
            the rotation from `B` to `A` (or `A` to `B` if `inverse=True`).

    Output ports:
        (0): The output vector `x_A` expressed in the `A`-axes.

    Parameters:
        rotation_type (str): The type of rotation representation to use. Must be one of
            ("quaternion", "roll_pitch_yaw", "dcm").
        enable_external_rotation_definition: If `True`, the block will have one
            input port for the rotation representation (quaternion, Euler angles, or
            cosine matrix).  Otherwise the rotation must be provided as a block
            parameter.
        inverse: If `True`, the block will compute the inverse transformation, i.e.
            if the matrix representation of the rotation is `C_AB` from frame `B` to
            frame `A`, the block will compute the inverse transformation
            `C_BA = C_AB⁻¹ = C_AB.T`
        quaternion (Array, optional): The quaternion representation of the rotation
            if `enable_external_rotation_definition=False`.
        roll_pitch_yaw (Array, optional): The Euler angles representation of the
            rotation if `enable_external_rotation_definition=False`.
        direction_cosine_matrix (Array, optional): The direction cosine matrix
            representation of the rotation if `enable_external_rotation_definition=False`.
    """

    @parameters(
        static=[
            "quaternion",
            "roll_pitch_yaw",
            "direction_cosine_matrix",
            "rotation_type",
            "enable_external_rotation_definition",
            "inverse",
        ]
    )
    def __init__(
        self,
        rotation_type,
        enable_external_rotation_definition=True,
        quaternion=None,
        roll_pitch_yaw=None,
        direction_cosine_matrix=None,
        inverse=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.external_rotation = enable_external_rotation_definition
        self.rotation_type = rotation_type
        self.inverse = inverse

        self.vector_input_index = self.declare_input_port()

        # Note: all of the possible rotation specifications are passed as parameters
        # to make the serialization work, but only one is valid at a time. This makes
        # sense from the UI, but is a bit strange when working directly with the code.
        # In any case, the typical use case is to have the external rotation port
        # enabled, so all of these should usually be None.  If more than one is
        # provided (which can happen for instance via hidden parameters in the JSON)
        # then only the rotation corresponding to the `rotation_type` will be used, and
        # the rest will be ignored.
        rotation = self._check_config(
            rotation_type,
            quaternion,
            roll_pitch_yaw,
            direction_cosine_matrix,
        )

        if enable_external_rotation_definition:
            self.rotation_input_index = self.declare_input_port()

        else:
            # Store the static rotation as a parameter (will be None if external
            # rotation is enabled)
            self.declare_dynamic_parameter("rotation", rotation)

        self._output_port_idx = self.declare_output_port(
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    def initialize(
        self,
        rotation_type,
        enable_external_rotation_definition,
        quaternion,
        roll_pitch_yaw,
        direction_cosine_matrix,
        inverse,
        rotation=None,
    ):
        if enable_external_rotation_definition != self.external_rotation:
            raise ValueError("Cannot change external rotation definition.")

        self.rotation_type = rotation_type
        self.inverse = inverse
        if not self.external_rotation:
            rotation = self._check_config(
                rotation_type,
                quaternion,
                roll_pitch_yaw,
                direction_cosine_matrix,
            )

            def _output_func(_time, _state, *inputs, **parameters):
                vector = inputs[self.vector_input_index]
                return self._apply(rotation, vector)

        else:

            def _output_func(_time, _state, *inputs, **parameters):
                vector = inputs[self.vector_input_index]
                rotation = inputs[self.rotation_input_index]
                return self._apply(rotation, vector)

        self.configure_output_port(
            self._output_port_idx,
            _output_func,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    def _check_config(
        self, rotation_type, quaternion, roll_pitch_yaw, direction_cosine_matrix
    ):
        if rotation_type not in ("quaternion", "roll_pitch_yaw", "DCM"):
            message = f"Invalid rotation type: {rotation_type}."
            raise BlockParameterError(
                message=message, system=self, parameter_name="rotation_type"
            )

        if self.external_rotation:
            # Input type checking will be done by `check_types`
            return

        if rotation_type == "quaternion":
            if quaternion is None:
                message = (
                    "A static quaternion must be provided if external rotation "
                    + "definition is disabled."
                )
                raise BlockParameterError(
                    message=message, system=self, parameter_name="quaternion"
                )
            rotation = cnp.asarray(quaternion)
            if rotation.shape != (4,):
                message = (
                    "The quaternion must have shape (4,), but has shape "
                    + f"{rotation.shape}."
                )
                raise BlockParameterError(
                    message=message, system=self, parameter_name="quaternion"
                )

        elif rotation_type == "roll_pitch_yaw":
            if roll_pitch_yaw is None:
                message = (
                    "A static roll-pitch-yaw sequence must be provided if external "
                    + "rotation definition is disabled."
                )
                raise BlockParameterError(
                    message=message, system=self, parameter_name="roll_pitch_yaw"
                )
            rotation = cnp.asarray(roll_pitch_yaw)
            if rotation.shape != (3,):
                message = (
                    "The Euler angles must have shape (3,), but has shape "
                    + f"{rotation.shape}."
                )
                raise BlockParameterError(
                    message=message, system=self, parameter_name="roll_pitch_yaw"
                )

        elif rotation_type == "DCM":
            if direction_cosine_matrix is None:
                message = (
                    "A static direction cosine matrix must be provided if external "
                    + "rotation definition is disabled."
                )
                raise BlockParameterError(
                    message=message,
                    system=self,
                    parameter_name="direction_cosine_matrix",
                )
            rotation = cnp.asarray(direction_cosine_matrix)
            if rotation.shape != (3, 3):
                message = (
                    "The direction cosine matrix must have shape (3, 3), but has shape "
                    + f"{rotation.shape}."
                )
                raise BlockParameterError(
                    message=message,
                    system=self,
                    parameter_name="direction_cosine_matrix",
                )

        return rotation

    def _apply(self, rotation: Rotation, vector: Array) -> Array:
        rot = {
            "quaternion": Rotation.from_quat,
            "roll_pitch_yaw": partial(Rotation.from_euler, EULER_SEQ),
            "DCM": Rotation.from_matrix,
        }[self.rotation_type](rotation)

        if self.inverse:
            rot = rot.inv()

        return rot.apply(vector)

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        vec = self.input_ports[self.vector_input_index].eval(context)

        with ErrorCollector.context(error_collector):
            if vec.shape != (3,):
                raise ShapeMismatchError(
                    system=self,
                    expected_shape=(3,),
                    actual_shape=vec.shape,
                )

        if self.external_rotation:
            rot = self.input_ports[self.rotation_input_index].eval(context)

            with ErrorCollector.context(error_collector):
                if self.rotation_type == "quaternion" and rot.shape != (4,):
                    raise ShapeMismatchError(
                        system=self,
                        expected_shape=(4,),
                        actual_shape=rot.shape,
                    )
                elif self.rotation_type == "roll_pitch_yaw" and rot.shape != (3,):
                    raise ShapeMismatchError(
                        system=self,
                        expected_shape=(3,),
                        actual_shape=rot.shape,
                    )
                elif self.rotation_type == "DCM" and rot.shape != (3, 3):
                    raise ShapeMismatchError(
                        system=self,
                        expected_shape=(3, 3),
                        actual_shape=rot.shape,
                    )


class CoordinateRotationConversion(LeafSystem):
    """Converts between different representations of rotations.

    See CoordinateRotation block documentation for descriptions of the different
    rotation representations supported. This block supports conversion between
    quaternion, roll-pitch-yaw (Euler angles), and direction cosine matrix (DCM).

    Note that conversions are reversible in terms of the abstract rotation, although
    creating a quaternion from a direction cosine matrix (and therefore creating a
    quaternion from roll-pitch-yaw sequence) results in an arbitrary sign assignment.

    Input ports:
        (0): The input rotation representation.

    Output ports:
        (1): The output rotation representation.

    Parameters:
        conversion_type (str): The type of rotation conversion to perform.
            Must be one of ("quaternion_to_euler", "quaternion_to_dcm",
            "euler_to_quaternion", "euler_to_dcm", "dcm_to_quaternion", "dcm_to_euler")
    """

    @parameters(static=["conversion_type"])
    def __init__(self, conversion_type, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port(requires_inputs=True)

    def initialize(self, conversion_type):
        if conversion_type not in (
            "quaternion_to_RPY",
            "quaternion_to_DCM",
            "RPY_to_quaternion",
            "RPY_to_DCM",
            "DCM_to_quaternion",
            "DCM_to_RPY",
        ):
            message = f"Invalid rotation conversion type: {conversion_type}."
            raise BlockParameterError(
                message=message, system=self, parameter_name="conversion_type"
            )

        _func = {
            "quaternion_to_RPY": quat_to_euler,
            "quaternion_to_DCM": quat_to_dcm,
            "RPY_to_quaternion": euler_to_quat,
            "RPY_to_DCM": euler_to_dcm,
            "DCM_to_quaternion": dcm_to_quat,
            "DCM_to_RPY": dcm_to_euler,
        }[conversion_type]

        def _output(_time, _state, *inputs, **_parameters):
            (u,) = inputs
            return _func(u)

        self.configure_output_port(
            self._output_port_idx,
            _output,
            requires_inputs=True,
        )

        # Serialization
        self.conversion_type = conversion_type

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        rot = self.input_ports[0].eval(context)

        with ErrorCollector.context(error_collector):
            if self.conversion_type in (
                "quaternion_to_RPY",
                "quaternion_to_DCM",
            ) and rot.shape != (4,):
                raise ShapeMismatchError(
                    system=self,
                    expected_shape=(4,),
                    actual_shape=rot.shape,
                )
            elif self.conversion_type in (
                "RPY_to_quaternion",
                "RPY_to_DCM",
            ) and rot.shape != (3,):
                raise ShapeMismatchError(
                    system=self,
                    expected_shape=(3,),
                    actual_shape=rot.shape,
                )
            elif self.conversion_type in (
                "DCM_to_quaternion",
                "DCM_to_RPY",
            ) and rot.shape != (3, 3):
                raise ShapeMismatchError(
                    system=self,
                    expected_shape=(3, 3),
                    actual_shape=rot.shape,
                )


class RigidBody(LeafSystem):
    """Implements dynamics of a single three-dimensional body.

    The block models both translational and rotational degrees of freedom, for a
    total of 6 degrees of freedom.  With second-order equations, the block has
    12 state variables, 6 for the position/orientation and 6 for the velocities/rates.

    Currently only a roll-pitch-yaw (Euler angle) representation is supported for
    the orientation.

    The full 12-dof state vector is `x = [p_i, Φ, vᵇ, ωᵇ]`, where `pⁱ` is the
    position in an inertial "world" frame `i`, `Φ` is the (roll, pitch, and yaw)
    Euler angle sequence defining the rotation from the inertial "world" frame to
    the body frame, `vᵇ` is the translational velocity with respect to body-fixed
    axes `b`, and `ωᵇ` is the angular velocity about the body-fixed axes.

    The mass and inertia properties of the block can independently be defined
    statically as parameters, or dynamically as inputs to the block.

    Input ports:
        (0) force_vector: 3D force vector, defined in the _body-fixed_ coordinate
        frame.  For example, if gravity is acting on the body, the gravity vector
        should be pre-rotated using CoordinateRotation.

        (1) torque_vector: 3D torque vector, be defined in the _body-fixed_
        coordinate frame.

        (2) inertia: If `enable_external_inertia_matrix=True`, this input provides
        the time-varying body-fixed inertia matrix.

    Output ports:
        (0): The position in the inertial "world" frame `pⁱ`.

        (1): The orientation of the body, represented as a roll-pitch-yaw Euler
        angle sequence.

        (2): The translational velocity with respect to body-fixed axes `vᵇ`.

        (3): The angular velocity about the body-fixed axes `ωᵇ`.

        (4): (if `enable_output_state_derivatives=True`) The time derivatives of the
        position variables in the world frame `ṗⁱ`. Not generally equal to the state
        `vᵇ`, defining time derivatives in the body frame.

        (5): (if `enable_output_state_derivatives=True`) The "Euler rates" `Φ̇`,
        which are the time derivatives of the Euler angles. Not generally equal to
        the angular velocity `ωᵇ`.

        (6): (if `enable_output_state_derivatives=True`) The body-fixed acceleration
        vector `aᵇ`.

        (7): (if `enable_output_state_derivatives=True`) The angular acceleration in
        body-fixed axes `ω̇ᵇ`.

    Parameters:
        initial_position (Array): The initial position in the inertial frame.

        initial_orientation (Array): The initial orientation of the body, represented
            as a roll-pitch-yaw Euler angle sequence.

        initial_velocity (Array): The initial translational velocity with respect to
            body-fixed axes.

        initial_angular_velocity (Array): The initial angular velocity about the
            body-fixed axes.

        enable_external_mass (bool, optional): If `True`, the block will have one
            input port for the mass. Otherwise the mass must be provided as a block
            parameter.

        mass (float, optional): The constant value for the body mass when
            `enable_external_mass=False`. If `None`, will default to 1.0.

        enable_external_inertia_matrix (bool, optional):  If `True`, the block will
            have one input port for a (3x3) inertia matrix. Otherwise the inertia
            matrix must be provided as a block parameter.

        inertia_matrix: The constant value for the body inertia matrix when
            `enable_external_inertia_matrix=False`. If `None`, will default to
            the 3x3 identity matrix.

        enable_output_state_derivatives (bool, optional): If `True`, the block will
            output the time derivatives of the state variables.

        gravity_vector (Array, optional): The constant gravitational acceleration vector
            acting on the body, defined in the _inertial_ frame. If `None`, will default
            to the zero vector.

    Notes:
        Assumes that the inertia matrix is computed at the center of mass.

        Assumes that the mass and inertia matrix are quasi-steady.  This means that
        if one or both is specified as "dynamic" inputs their time derivative is
        neglected in the dynamics.  For instance, for pure translation (`w_b=0`) the
        approximation to Newton's law is `F_net = (d/dt)(m * v) ≈ m * (dv/dt)`.
    """

    class RigidBodyState(NamedTuple):
        position: Array
        orientation: Array
        velocity: Array
        angular_velocity: Array

        def asarray(self):
            return cnp.concatenate(
                [self.position, self.orientation, self.velocity, self.angular_velocity]
            )

    @parameters(
        static=[
            "initial_position",
            "initial_orientation",
            "initial_velocity",
            "initial_angular_velocity",
            "enable_external_mass",
            "enable_external_inertia_matrix",
            "enable_output_state_derivatives",
        ],
        dynamic=["mass", "inertia_matrix", "gravity_vector"],
    )
    def __init__(
        self,
        initial_position,
        initial_orientation,
        initial_velocity,
        initial_angular_velocity,
        enable_external_mass=False,
        mass=1.0,
        enable_external_inertia_matrix=False,
        inertia_matrix=cnp.eye(3),
        enable_output_state_derivatives=False,
        gravity_vector=cnp.zeros(3),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._enable_external_mass = enable_external_mass
        self._enable_external_inertia_matrix = enable_external_inertia_matrix
        self._enable_output_state_derivatives = enable_output_state_derivatives

        initial_state = self._make_initial_state(
            initial_position,
            initial_orientation,
            initial_velocity,
            initial_angular_velocity,
        )

        self._continuous_state_idx = self.declare_continuous_state(
            default_value=initial_state,
            as_array=False,
            ode=self._state_derivative,
        )

        self._configure_ports(
            initial_state,
            enable_external_mass,
            enable_external_inertia_matrix,
            enable_output_state_derivatives,
        )

    def initialize(
        self,
        initial_position,
        initial_orientation,
        initial_velocity,
        initial_angular_velocity,
        enable_external_mass,
        enable_external_inertia_matrix,
        enable_output_state_derivatives,
        mass,
        inertia_matrix,
        gravity_vector,
    ):
        if enable_external_mass != self._enable_external_mass:
            raise ValueError("Cannot change external mass definition.")
        if enable_external_inertia_matrix != self._enable_external_inertia_matrix:
            raise ValueError("Cannot change external inertia matrix definition.")
        if enable_output_state_derivatives != self._enable_output_state_derivatives:
            raise ValueError("Cannot change output state derivatives definition.")

        gravity_vector = cnp.asarray(gravity_vector)
        if gravity_vector.shape != (3,):
            message = (
                "Gravity vector must have shape (3,), but has shape "
                + f"{gravity_vector.shape}."
            )
            raise BlockParameterError(
                message=message, system=self, parameter_name="gravity_vector"
            )

        initial_state = self._make_initial_state(
            initial_position,
            initial_orientation,
            initial_velocity,
            initial_angular_velocity,
        )

        self.configure_continuous_state(
            self._continuous_state_idx,
            default_value=initial_state,
            as_array=False,
            ode=self._state_derivative,
        )

        self.configure_output_port(
            self.pos_output_index,
            self._pos_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            default_value=initial_state.position,
        )

        self.configure_output_port(
            self.orientation_output_index,
            self._orientation_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            default_value=initial_state.orientation,
        )

        self.configure_output_port(
            self.vel_output_index,
            self._vel_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            default_value=initial_state.velocity,
        )

        self.configure_output_port(
            self.ang_vel_output_index,
            self._ang_vel_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            default_value=initial_state.angular_velocity,
        )

    def _make_initial_state(
        self,
        initial_position,
        initial_orientation,
        initial_velocity,
        initial_angular_velocity,
    ):
        # Validate initial state arrays and create named tuple for initial state.
        initial_position = cnp.asarray(initial_position)
        if initial_position.shape != (3,):
            message = (
                "Initial position must have shape (3,), but has shape "
                + f"{initial_position.shape}."
            )
            raise BlockParameterError(
                message=message, system=self, parameter_name="initial_position"
            )

        initial_orientation = cnp.asarray(initial_orientation)
        if initial_orientation.shape != (3,):
            message = (
                "Initial orientation must have shape (3,), but has shape "
                + f"{initial_orientation.shape}."
            )
            raise BlockParameterError(
                message=message, system=self, parameter_name="initial_orientation"
            )

        initial_velocity = cnp.asarray(initial_velocity)
        if initial_velocity.shape != (3,):
            message = (
                "Initial velocity must have shape (3,), but has shape "
                + f"{initial_velocity.shape}."
            )
            raise BlockParameterError(
                message=message, system=self, parameter_name="initial_velocity"
            )

        initial_angular_velocity = cnp.asarray(initial_angular_velocity)
        if initial_angular_velocity.shape != (3,):
            message = (
                "Initial angular velocity must have shape (3,), but has shape "
                + f"{initial_angular_velocity.shape}."
            )
            raise BlockParameterError(
                message=message, system=self, parameter_name="initial_angular_velocity"
            )

        return self.RigidBodyState(
            position=initial_position,
            orientation=initial_orientation,
            velocity=initial_velocity,
            angular_velocity=initial_angular_velocity,
        )

    @property
    def force_input(self):
        return self.input_ports[self.force_index]

    @property
    def torque_input(self):
        return self.input_ports[self.torque_index]

    @property
    def mass_input(self):
        if self.mass_index is None:
            return None
        return self.input_ports[self.mass_index]

    @property
    def inertia_input(self):
        if self.inertia_index is None:
            return None
        return self.input_ports[self.inertia_index]

    @property
    def position_output(self):
        return self.output_ports[self.pos_output_index]

    @property
    def orientation_output(self):
        return self.output_ports[self.orientation_output_index]

    @property
    def velocity_output(self):
        return self.output_ports[self.vel_output_index]

    @property
    def angular_velocity_output(self):
        return self.output_ports[self.ang_vel_output_index]

    def _configure_ports(
        self,
        initial_state,
        enable_external_mass,
        enable_external_inertia_matrix,
        enable_output_state_derivatives,
    ):
        # External force vector input
        self.force_index = self.declare_input_port(name="force_vector")

        # External torque vector input
        self.torque_index = self.declare_input_port(name="torque_vector")

        # External mass input
        self.mass_index = None
        if enable_external_mass:
            self.mass_index = self.declare_input_port(name="mass")

        # External inertia matrix input
        self.inertia_index = None
        if enable_external_inertia_matrix:
            self.inertia_index = self.declare_input_port(name="inertia_matrix")

        # Position output
        self.pos_output_index = self.declare_output_port(
            self._pos_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            default_value=initial_state.position,
            name=f"{self.name}:position",
        )

        # Orientation output
        self.orientation_output_index = self.declare_output_port(
            self._orientation_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            default_value=initial_state.orientation,
            name=f"{self.name}:orientation",
        )

        # Velocity output
        self.vel_output_index = self.declare_output_port(
            self._vel_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            default_value=initial_state.velocity,
            name=f"{self.name}:velocity",
        )

        # Angular velocity output
        self.ang_vel_output_index = self.declare_output_port(
            self._ang_vel_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            default_value=initial_state.angular_velocity,
            name=f"{self.name}:angular_velocity",
        )

        if enable_output_state_derivatives:
            self.pos_deriv_output_index = self.declare_output_port(
                self._pos_derivative,
                prerequisites_of_calc=[DependencyTicket.xc],
                requires_inputs=False,
                default_value=cnp.zeros(3),
                name=f"{self.name}:position_dot",
            )

            self.orientation_deriv_output_index = self.declare_output_port(
                self._orientation_derivative,
                prerequisites_of_calc=[DependencyTicket.xc],
                requires_inputs=False,
                default_value=cnp.zeros(3),
                name=f"{self.name}:orientation_dot",
            )

            force_ticket = self.input_ports[self.force_index].ticket
            self.vel_deriv_output_index = self.declare_output_port(
                self._vel_derivative,
                prerequisites_of_calc=[force_ticket, DependencyTicket.xc],
                requires_inputs=True,
                default_value=cnp.zeros(3),
                name=f"{self.name}:velocity_dot",
            )

            torque_ticket = self.input_ports[self.torque_index].ticket
            self.ang_vel_deriv_output_index = self.declare_output_port(
                self._ang_vel_derivative,
                prerequisites_of_calc=[torque_ticket, DependencyTicket.xc],
                requires_inputs=True,
                default_value=cnp.zeros(3),
                name=f"{self.name}:angular_velocity_dot",
            )

    def _pos_output(self, time, state, *inputs, **parameters):
        xc = state.continuous_state
        return xc.position

    def _orientation_output(self, time, state, *inputs, **parameters):
        xc = state.continuous_state
        return xc.orientation

    def _vel_output(self, time, state, *inputs, **parameters):
        xc = state.continuous_state
        return xc.velocity

    def _ang_vel_output(self, time, state, *inputs, **parameters):
        xc = state.continuous_state
        return xc.angular_velocity

    def _pos_derivative(self, time, state, *inputs, **parameters):
        # This function produces the inertial -> body rotation.  What we
        # want is to rotate the body-fixed velocity into the inertial frame,
        # so we need the transpose of this rotation matrix.
        xc = state.continuous_state
        C_BI = euler_to_dcm(xc.orientation)
        return C_BI.T @ xc.velocity

    def _orientation_derivative(self, time, state, *inputs, **parameters):
        # Matrix mapping angular velocity in the body-fixed frame to Euler rates
        xc = state.continuous_state
        H = euler_kinematics(xc.orientation)
        return H @ xc.angular_velocity

    def _vel_derivative(self, time, state, *inputs, **parameters):
        xc = state.continuous_state

        if self.mass_index is not None:
            m = inputs[self.mass_index]
        else:
            m = parameters["mass"]

        # Gravity vector in the inertial frame
        g_I = parameters["gravity_vector"]

        # Acceleration in body-fixed frame
        F_B = inputs[self.force_index]
        C_BI = euler_to_dcm(xc.orientation)
        a_B = F_B / m + C_BI @ g_I

        # Body-fixed acceleration is the inertial plus Coriolis terms
        return a_B - cnp.cross(xc.angular_velocity, xc.velocity)

    def _ang_vel_derivative(self, time, state, *inputs, **parameters):
        xc = state.continuous_state

        if self.inertia_index is not None:
            J_B = inputs[self.inertia_index]
        else:
            J_B = parameters["inertia_matrix"]

        # Torque in body-fixed frame
        tau_B = inputs[self.torque_index]

        wJw = cnp.cross(xc.angular_velocity, J_B @ xc.angular_velocity)
        return cnp.linalg.solve(J_B, tau_B - wJw)

    def _state_derivative(self, time, state, *inputs, **parameters):
        # See Eq. (1.7-18) in Lewis, Johnson, Stevens
        args = (time, state, *inputs)
        return self.RigidBodyState(
            position=self._pos_derivative(*args, **parameters),
            orientation=self._orientation_derivative(*args, **parameters),
            velocity=self._vel_derivative(*args, **parameters),
            angular_velocity=self._ang_vel_derivative(*args, **parameters),
        )

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        force = self.input_ports[self.force_index].eval(context)
        torque = self.input_ports[self.torque_index].eval(context)

        with ErrorCollector.context(error_collector):
            if force.shape != (3,):
                raise ShapeMismatchError(
                    system=self,
                    expected_shape=(3,),
                    actual_shape=force.shape,
                )

            if torque.shape != (3,):
                raise ShapeMismatchError(
                    system=self,
                    expected_shape=(3,),
                    actual_shape=torque.shape,
                )

        if self.mass_index is not None:
            mass = self.input_ports[self.mass_index].eval(context)

            with ErrorCollector.context(error_collector):
                if mass.shape != ():
                    raise ShapeMismatchError(
                        system=self,
                        expected_shape=(),
                        actual_shape=mass.shape,
                    )

        if self.inertia_index is not None:
            inertia = self.input_ports[self.inertia_index].eval(context)

            with ErrorCollector.context(error_collector):
                if inertia.shape != (3, 3):
                    raise ShapeMismatchError(
                        system=self,
                        expected_shape=(3, 3),
                        actual_shape=inertia.shape,
                    )
