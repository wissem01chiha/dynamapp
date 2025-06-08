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

"""HAL for Quanser Qube Servo hardware."""

import sys
import types
import signal

import numpy as np
import jax.numpy as jnp
from collimator.backend import numpy_api as cnp, io_callback
from collimator.framework import LeafSystem

__all__ = ["QuanserHAL", "QubeServoModel", "animate_qube"]


class QubeServoModel(LeafSystem):
    """Plant model for the Quanser Qube Servo Furuta Pendulum.

    The Quanser Qube Servo is a pendulum controlled by a rotary arm. The
    rotary arm is actuated by a DC motor. The pendulum is free to rotate about
    the rotary arm.

    The state of the system is given by the rotor angle (theta), pendulum angle
    (alpha), rotor angular velocity, and pendulum angular velocity. The input to
    the system is the voltage applied to the motor, which is converted to torque
    by a simple linear model.

    Input ports:
        (0) The motor voltage signal

    Output ports:
        (0) If `full_state_output` is False, the rotor angle and pendulum angle.
            Otherwise, will return the entire continuous state vector.

    Parameters:
        x0: Initial state of the system [theta, alpha, theta_dot, alpha_dot]
        Rm: Motor resistance (Ohms)
        km: Back-emf constant (V-s/rad)
        mr: Rotary arm mass (kg)
        Lr: Rotor arm length (m)
        br: Rotor arm damping coefficient (N-m-s/rad)
        mp: Pendulum mass (kg)
        Lp: Pendulum arm length (m)
        bp: Pendulum damping coefficient (N-m-s/rad)
        g: Gravitational constant (m/s^2)
        kr: Feedback control to send the rotor back to zero
        full_state_output: If True, output the full state vector. Otherwise,
            only output the rotor and pendulum angles.

    """

    def __init__(
        self,
        x0=[0.0, 0.0, 0.0, 0.0],
        Rm=8.4,
        km=0.042,
        mr=0.095,
        Lr=0.085,
        br=5e-4,
        mp=0.024,
        Lp=0.129,
        bp=2.5e-5,
        g=9.81,
        kr=0.0,
        full_state_output=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("Rm", Rm)
        self.declare_dynamic_parameter("km", km)
        self.declare_dynamic_parameter("mr", mr)
        self.declare_dynamic_parameter("Lr", Lr)
        self.declare_dynamic_parameter("br", br)
        self.declare_dynamic_parameter("mp", mp)
        self.declare_dynamic_parameter("Lp", Lp)
        self.declare_dynamic_parameter("bp", bp)
        self.declare_dynamic_parameter("g", g)
        self.declare_dynamic_parameter("kr", kr)

        # One input: motor voltage
        self.declare_input_port()

        self.declare_continuous_state(default_value=x0, ode=self._ode)

        if full_state_output:
            self.declare_continuous_state_output()

        else:
            # Only measure (alpha, theta)
            def _obs_callback(time, state, *inputs, **parameters):
                return state.continuous_state[:2]

            self.declare_output_port(_obs_callback, requires_inputs=False)

    def _ode(self, time, state, *inputs, **parameters):
        # Unpack state
        q, dq = state.continuous_state[:2], state.continuous_state[2:]
        theta, alpha = q  # Rotor angle, pendulum angle
        theta_dot, alpha_dot = dq

        # Unpack parameters
        Rm = parameters["Rm"]
        km = parameters["km"]
        mr = parameters["mr"]
        Lr = parameters["Lr"]
        br = parameters["br"]
        mp = parameters["mp"]
        Lp = parameters["Lp"]
        bp = parameters["bp"]
        kr = parameters["kr"]
        g = parameters["g"]

        lp = Lp / 2  # Pendulum center of mass

        # Moment of inertia of the rotor arm about the motor
        Jr = mr * Lr**2 / 3

        # Moment of inertia of the pendulum about the pivot point
        Jp = mp * Lp**2 / 3

        # Unpack inputs
        (u,) = inputs
        u = cnp.atleast_1d(u)

        # Feedback control to send the rotor back to zero
        u -= kr * theta

        # Mass matrix
        M = cnp.array(
            [
                [Jr + Jp * cnp.sin(alpha) ** 2, -mp * lp * Lr * cnp.cos(alpha)],
                [-mp * lp * Lr * cnp.cos(alpha), Jp],
            ]
        )

        # Coriolis matrix
        C = cnp.array(
            [
                [
                    Jp * cnp.sin(2 * alpha) * alpha_dot + br + 0 * km**2 / Rm,
                    mp * lp * Lr * cnp.sin(alpha) * alpha_dot,
                ],
                [-0.5 * Jp * cnp.sin(2 * alpha) * theta_dot, bp],
            ]
        )

        # Gravity vector
        tau_g = cnp.array([0, mp * g * lp * cnp.sin(alpha)])

        # Input matrix
        B = cnp.array(
            [
                [km / Rm],
                [0],
            ]
        )

        # State space representation
        ddq = cnp.linalg.solve(M, B @ u - (C @ dq + tau_g))
        return cnp.concatenate([dq, ddq])


class QuanserHAL(LeafSystem):
    """Hardware Abstraction Layer for Quanser hardware.

    This block provides an interface to virtual or physical Quanser hardware.
    It requires that the Quanser hardware or QLabs simulator be properly configured
    and that the Quanser python library is available on the system path.  See the
    Quanser documentation for more information.

    To use an idealized model of the Qube Servo hardware, see the
    `collimator.library.QubeServoModel` block, which may be run without hardware
    or in the cloud-based simulation UI.

    Input ports:
        (0) Control signal to the motor in volts

    Output ports:
        (0) The observed rotor and pendulum angles in radians

    Parameters:
        dt: The time step of the simulation.
        version: The version of the Qube hardware (2 or 3).  By default, version 2 is
            used with `hardware=False`, or version 3 is used with `hardware=True`.
        hardware:
            If True, connect to the physical hardware. If False, connect to the
            QLabs simulator.
        name: The name of the system in the Collimator model.

    """

    def __init__(self, dt, version=2, hardware=False, name="QuanserHAL", ui_id=None):
        super().__init__(name=name, ui_id=ui_id)

        if version is None:
            version = 2 if not hardware else 3

        # Init QLabs
        # HACK for macOS - Quanser's code only works on Windows
        # Insert asyncio.windows_events fake module
        if sys.platform != "win32":
            _windows_events = types.ModuleType("windows_events")
            _windows_events.INFINITE = np.iinfo(np.uint32).max
            sys.modules["asyncio.windows_events"] = _windows_events

        try:
            from pal.products.qube import QubeServo2, QubeServo3
        except Exception:
            raise ImportError(
                "Could not import QubeServo2 or QubeServo3 from pal.products.qube. "
                "Check that the hardware drivers are available on the system path."
            )

        if version not in [2, 3]:
            raise ValueError("version must be 2 or 3")

        if version == 2:
            QubeClass = QubeServo2
        else:
            QubeClass = QubeServo3

        self.qube = QubeClass(hardware=hardware, pendulum=1, frequency=1 / dt)
        self._setup_siginthandler()

        print("Initialized Qube")
        if self.qube.card is None:
            raise RuntimeError(
                "Could not find hardware. Try power-cycling and check connections."
            )
        self.qube.write_led(color=[0, 1, 0])

        self.declare_input_port()  # Inputs are the control signals to the motor

        # Periodically send the control signals to the motor
        self.declare_periodic_update(
            self.step,
            period=dt,
            offset=0.0,
        )

        # Periodically read the sensor outputs
        self.declare_output_port(
            self.output,
            period=dt,
            offset=0.0,
            requires_inputs=False,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()
        super().__exit__(exc_type, exc_value, traceback)

    def _setup_siginthandler(self):
        self.prev_sigint_handler = signal.signal(signal.SIGINT, self._interrupt)

    def _restore_siginthandler(self):
        print("Restoring sigint handler")
        if self.prev_sigint_handler is not None:
            signal.signal(signal.SIGINT, self.prev_sigint_handler)
            self.prev_sigint_handler = None

    # This custom handler makes it possible to Interrupt the jupyter kernel
    # and still connect again to the Qube environment.
    def _interrupt(self, signum, frame):
        prev_handler = self.prev_sigint_handler
        self.terminate()
        if prev_handler is not None:
            prev_handler(signum, frame)

    def terminate(self):
        self._restore_siginthandler()
        if self.qube is not None:
            self.qube.write_led(color=[1, 1, 0])
            self.qube.terminate()
            self.qube = None

    def _impure_step(self, voltage):
        # Write the voltage to the Qube
        self.qube.write_voltage(voltage)

    def step(self, time, state, *inputs, **parameters):
        return io_callback(self._impure_step, None, *inputs)

    def _impure_output(self):
        # Read the sensor outputs
        self.qube.read_outputs()
        theta, alpha = self.qube.motorPosition, self.qube.pendulumPosition
        return jnp.array([theta, alpha])

    def output(self, time, state, *inputs, **parameters):
        return io_callback(self._impure_output, jnp.zeros(2))


def animate_qube(
    t,
    x,
    parameters,
    interval=50,
    stride=1,
    figsize=(4, 4),
    elev=30,
    azim=-30,
    filename=None,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from IPython import display

    # Cube util function
    def cuboid_data2(pos, size=(1, 1, 1), rotation=None):
        X = [
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        ]
        X = np.array(X).astype(float)
        for i in range(3):
            X[:, :, i] *= size[i]
        if rotation is not None:
            for i in range(4):
                X[:, i, :] = np.dot(rotation, X[:, i, :].T).T
        X += pos
        return X

    # Plot cube for drone body
    def plot_cube(position, size=None, rotation=None, color=None, **kwargs):
        if not isinstance(color, (list, np.ndarray)):
            color = ["C0"] * len(position)
        if not isinstance(size, (list, np.ndarray)):
            size = [(1, 1, 1)] * len(position)
        g = cuboid_data2(position, size=size, rotation=rotation)
        return Poly3DCollection(g, facecolor=np.repeat(color, 6), **kwargs)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")

    Lr = parameters["Lr"]
    Lp = parameters["Lp"]

    lim = (-1.5 * Lr, 1.5 * Lr)
    ax.set_xlim3d(*lim)
    ax.set_ylim3d(*lim)
    ax.set_zlim3d(*lim)
    ax.set_xlabel("x[m]")
    ax.set_ylabel("y[m]")
    ax.set_zlabel("z[m]")

    x = x[::stride, :]
    t = t[::stride]

    theta = -x[:, 0]
    alpha = x[:, 1]

    # Rotor arm base position
    xb = np.zeros(3)

    # Rotor arm end position
    xr = np.repeat(xb.reshape(1, 3), len(t), axis=0)
    xr[:, 0] += Lr * np.cos(theta)
    xr[:, 1] += Lr * np.sin(theta)

    # Pendulum end position
    xp = xr.copy()
    xp[:, 0] -= Lp * np.sin(alpha) * np.sin(theta)
    xp[:, 1] += Lp * np.sin(alpha) * np.cos(theta)
    xp[:, 2] -= Lp * np.cos(alpha)

    # Motor base
    Lm = 0.8 * Lr
    pos = np.array([-Lm / 2, -Lm / 2, -Lm])
    body = plot_cube(pos, size=[Lm, Lm, Lm], edgecolor="k", color="grey")
    ax.add_collection3d(body)

    (l1,) = ax.plot([], [], [], lw=2, color="k", zorder=10)
    (l2,) = ax.plot([], [], [], lw=2, color="k", zorder=10)

    # Single frame plotting
    def _animate(i):
        # Rotor
        rotor_points = np.stack((xb, xr[i]))
        l1.set_data(rotor_points[:, 0], rotor_points[:, 1])
        l1.set_3d_properties(rotor_points[:, 2])

        # Pendulum
        pendulum_points = np.stack((xr[i], xp[i]))
        l2.set_data(pendulum_points[:, 0], pendulum_points[:, 1])
        l2.set_3d_properties(pendulum_points[:, 2])

        return l1, l2

    # Rotate the axes and update
    ax.view_init(elev, azim)
    ax.set_aspect("equal")

    # Create the animation
    anim = FuncAnimation(fig, _animate, frames=len(t), blit=True, interval=interval)

    # Check if the code is running in a Jupyter notebook
    try:
        cfg = get_ipython().config  # noqa: F841
        in_notebook = True
    except NameError:
        in_notebook = False

    if in_notebook:
        plt.close(fig)
        html_content = display.HTML(anim.to_html5_video())
        return html_content

    try:
        writer = FFMpegWriter(fps=60, metadata={"artist": "Me"}, bitrate=1800)
        savefilename = filename or "qube.mp4"
        anim.save(savefilename, writer=writer)
        print(f"Animation saved as {savefilename}")
    except RuntimeError as e:
        print(f"Error saving animation: {e}")
        plt.close()
