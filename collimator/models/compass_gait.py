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

from functools import partial
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from ..framework import LeafSystem


class CompassGait(LeafSystem):
    """
    https://underactuated.csail.mit.edu/simple_legs.html#section2
    https://github.com/RobotLocomotion/drake/blob/master/examples/compass_gait/compass_gait.h
    """

    class DiscreteStateType(NamedTuple):
        toe: float
        left_leg_is_stance: bool

    def __init__(
        self,
        *args,
        mass_hip=10.0,
        mass_leg=5.0,
        length_leg=1.0,
        center_of_mass_leg=0.5,
        gravity=9.81,
        slope=0.0525,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.declare_dynamic_parameter("gamma", slope)  # Ramp angle
        self.declare_dynamic_parameter("g", gravity)  # Gravity
        self.declare_dynamic_parameter("L", length_leg)  # Leg length
        self.declare_dynamic_parameter("mh", mass_hip)  # Hip mass
        self.declare_dynamic_parameter("m", mass_leg)  # Leg mass
        self.declare_dynamic_parameter("a", length_leg - center_of_mass_leg)
        self.declare_dynamic_parameter("b", center_of_mass_leg)

        # Continuous state: q = [θ_st, θ_sw], dq = [dθ_st, dθ_sw]
        self.declare_continuous_state(shape=(4,), ode=self.ode)

        # # Add a torque input (by default set to zero)
        # self.declare_input_port(name="u")
        # self.input_ports[0].fix_value(0.0)

        # Extra states for "toe position" and "left leg is stance" for visualization
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(toe=0.0, left_leg_is_stance=True),
            as_array=False,
        )

        self.declare_continuous_state_output()

        def _toe_output(time, state, *inputs, **parameters):
            return state.discrete_state.toe

        def _stance_output(time, state, *inputs, **parameters):
            return state.discrete_state.left_leg_is_stance

        self.declare_output_port(_toe_output, requires_inputs=False)
        self.declare_output_port(_stance_output, requires_inputs=False)

        self.declare_zero_crossing(
            guard=self.foot_collision,
            reset_map=self.reset,
            name="foot_collision",
            direction="positive_then_non_positive",
        )

    def ode(self, time, state, *inputs, **parameters):
        xc = state.continuous_state
        q, dq = xc[:2], xc[2:]  # q[0] ~ stance leg, q[1] ~ swing leg

        g = parameters["g"]
        mh = parameters["mh"]
        m = parameters["m"]
        a = parameters["a"]
        b = parameters["b"]
        L = parameters["L"]

        # Mass matrix
        M = jnp.array(
            [
                [(mh + m) * L**2 + m * a**2, -m * L * b * jnp.cos(q[1] - q[0])],
                [-m * L * b * jnp.cos(q[1] - q[0]), m * b**2],
            ]
        )

        # Coriolis matrix
        C = jnp.array(
            [
                [0, -m * L * b * jnp.sin(q[1] - q[0]) * dq[1]],
                [m * L * b * jnp.sin(q[1] - q[0]) * dq[0], 0],
            ]
        )

        # Gravity vector
        tau_g = jnp.array(
            [(mh * L + m * a + m * L) * jnp.sin(q[0]), -m * b * g * jnp.sin(q[1])]
        )

        # Control matrix
        # B = jnp.array([[-1.], [1.]])

        # # Compute the acceleration
        # ddq = cs.inv(M) @ (tau_g + B @ u - C @ dq)
        ddq = jnp.linalg.inv(M) @ (tau_g - C @ dq)

        return jnp.array([dq[0], dq[1], ddq[0], ddq[1]])

    def foot_collision(self, time, state, *inputs, **parameters):
        """Guard function for foot collision

        See CompassGait<T>::FootCollision
        in drake/examples/compass_gait/compass_gait.cc
        """
        θ_st, θ_sw = state.continuous_state[:2]
        gamma = parameters["gamma"]
        collision = gamma - θ_st - θ_sw
        return jnp.fmax(collision, θ_sw - θ_st)

    def reset(self, time, state, *inputs, **parameters):
        """Reset map for foot collision

        See CompassGait<T>::CollisionDynamics
        in drake/examples/compass_gait/compass_gait.cc
        """
        xc = state.continuous_state  # "Minus" continuous state
        m = parameters["m"]
        mh = parameters["mh"]
        a = parameters["a"]
        b = parameters["b"]
        L = parameters["L"]

        cst = jnp.cos(xc[0])
        csw = jnp.cos(xc[1])
        hip_angle = xc[1] - xc[0]
        c = jnp.cos(hip_angle)
        sst = jnp.sin(xc[0])
        ssw = jnp.sin(xc[1])

        M_floating_base = jnp.array(
            [
                [2 * m + mh, 0.0, (m * a + m * L + mh * L) * cst, -m * b * csw],
                [0.0, 2 * m + mh, -(m * a + m * L + mh * L) * sst, m * b * ssw],
                [
                    (m * a + m * L + mh * L) * cst,
                    -(m * a + m * L + mh * L) * sst,
                    m * a * a + (m + mh) * L * L,
                    -m * L * b * c,
                ],
                [-m * b * csw, m * b * ssw, -m * L * b * c, m * b * b],
            ]
        )

        # The kinematic Jacobian for the location of the pre-collision swing toe
        # about the origin (at the pre-collision stance toe).
        J = jnp.array([[1.0, 0.0, L * cst, -L * csw], [0.0, 0.0, -L * sst, L * ssw]])

        # Floating-base velocity before contact
        v_pre = jnp.array([0.0, 0.0, xc[2], xc[3]])

        # Floating-base velocity after contact.
        Minv = jnp.linalg.inv(M_floating_base)
        v_post = v_pre - Minv @ J.T @ jnp.linalg.inv(J @ Minv @ J.T) @ J @ v_pre

        state = state.with_continuous_state(
            jnp.array([xc[1], xc[0], v_post[3], v_post[2]])
        )

        # Switch stance and swing legs
        left_leg_is_stance = state.discrete_state.left_leg_is_stance
        toe = state.discrete_state.toe
        xd = self.DiscreteStateType(
            toe=toe - 2 * L * jnp.sin(hip_angle / 2),
            left_leg_is_stance=jnp.logical_not(left_leg_is_stance),
        )
        return state.with_discrete_state(xd)

    def plot_trajectory(self, xc, left_leg_is_stance, **kwargs):
        """Plot theta vs theta_dot for left leg"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=kwargs.get("figsize", (4, 4)))

        θ_st, θ_sw = xc[:, 0], xc[:, 1]
        dθ_st, dθ_sw = xc[:, 2], xc[:, 3]

        θL = jnp.where(left_leg_is_stance, θ_st, θ_sw)
        dθL = jnp.where(left_leg_is_stance, dθ_st, dθ_sw)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(θL, dθL, lw=2)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\dot{\theta}$")
        ax.grid()
        plt.show()

    def animate(self, t, xc, toe, left_leg_is_stance, p, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from IPython import display

        gamma, L = p["gamma"], p["L"]

        if kwargs.get("interpolation", True):
            dt = kwargs.get("dt", 0.01)
            t_interp = np.arange(t[0], t[-1], dt)
            xc = jax.vmap(partial(jnp.interp, t_interp, t))(xc.T).T
            toe = jnp.interp(t_interp, t, toe)
            left_leg_is_stance = jnp.interp(t_interp, t, left_leg_is_stance)

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 3)))

        xlim = kwargs.get("xlim", [-1.0, 10.0])
        ylim = kwargs.get("ylim", [-1.0, 1.0])

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")

        θ_st, θ_sw = xc[:, 0], xc[:, 1]
        xt, yt = toe * jnp.cos(gamma), -toe * jnp.sin(gamma)

        def rotate(x, y, θ):
            return x * jnp.cos(θ) - y * jnp.sin(θ), x * jnp.sin(θ) + y * jnp.cos(θ)

        # Plot ramp based on starting toe location
        m = jnp.tan(-gamma)
        b = yt[0] - m * xt[0]
        xr = jnp.linspace(0.9 * xlim[0], 1.1 * xlim[1], 100)
        yr = m * xr + b

        ramp_color = kwargs.get("ramp_color", None)
        (l_ramp,) = ax.plot(xr, yr, "-", c=ramp_color)

        walker_color = kwargs.get("walker_color", None)
        (l_sw,) = ax.plot([], "-", c=walker_color)
        walker_color = l_sw.get_color()
        (l_st,) = ax.plot([], "-", c=walker_color)

        def _animate(i):
            # Pivot point
            xp = xt[i] + L * jnp.sin(θ_st[i])
            yp = yt[i] + L * np.cos(θ_st[i])

            # Free toe
            xf = xp - L * jnp.sin(θ_sw[i])
            yf = yp - L * jnp.cos(θ_sw[i])

            l_st.set_data([xt[i], xp], [yt[i], yp])
            l_sw.set_data([xp, xf], [yp, yf])
            return l_st, l_sw

        anim = FuncAnimation(
            fig,
            _animate,
            frames=xc.shape[0],
            interval=kwargs.get("interval", 50),
            blit=True,
        )

        if not kwargs.get("notebook", True):
            return anim

        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()  # avoid plotting a spare static plot
