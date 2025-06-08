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

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
from ..framework import LeafSystem


class DummyBlock(LeafSystem):
    def __init__(
        self,
        name="dummy",
    ):
        super().__init__(name=name)
        self.declare_dynamic_parameter("dummy_param", 1.0)


class CompactEV(LeafSystem):
    """
    Leaf equivalent of the UI model.
    """

    class DiscreteStateType(NamedTuple):
        # the motor voltage based enable unit delay block.
        mot_volt_based_enable: float = 0.0
        # the app and bpp ZOH.
        app_zoh: float = 0.0
        bpp_zoh: float = 0.0
        # the veh_spd and coolant_temp ZOH.
        veh_spd_zoh: float = 0.0
        coolant_zoh: float = 0.0

    def __init__(
        self,
        *args,
        dt=0.01,  # 0.01 matches the UI model
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.declare_dynamic_parameter("mass", 2e3)
        for i in range(100):
            self.declare_dynamic_parameter(f"dummy{str(i)}", float(i))
        # self.declare_dynamic_parameter("m2km", 0.001)

        # Continuous states:
        self.velocity_state_idx = 0
        self.position_state_idx = 1
        self.accel_loop_breaker_state_idx = 2
        self.driver_pid_iterm_state_idx = 3
        self.coolant_temp_state_idx = 4
        self.battery_soc_state_idx = 5
        self.current_loop_breaker_state_idx = 6
        self.driver_pid_iterm2_state_idx = 7

        velocity_init = 0.0
        pos_init = 0.0
        accel_loop_breaker_init = 0.0
        driver_pid_iterm_init = 0.0
        coolant_temp_init = 80
        battery_soc_init = 0.99
        current_loop_breaker_init = 0.0
        driver_pid_iterm2_init = 0.0
        default_value = jnp.array(
            [
                velocity_init,
                pos_init,
                accel_loop_breaker_init,
                driver_pid_iterm_init,
                coolant_temp_init,
                battery_soc_init,
                current_loop_breaker_init,
                driver_pid_iterm2_init,
            ]
        )
        self.declare_continuous_state(default_value=default_value, ode=self.ode)

        # outputting the state only is sufficient to verify the model is erasonable correct.
        self.declare_continuous_state_output()

        self.declare_input_port()

        self.declare_discrete_state(
            default_value=self.DiscreteStateType(), as_array=False
        )

        self.declare_periodic_update(
            self._discrete_update,
            period=dt,
            offset=dt,
        )

    def _update_app_zoh(self, time, state, e, **params):
        veh_spd_mps = state.continuous_state[self.velocity_state_idx]
        pid_Istate = state.continuous_state[self.driver_pid_iterm_state_idx]
        pid_Istate2 = state.continuous_state[self.driver_pid_iterm2_state_idx]
        drive_cycle_spd = 100
        app_norm, bpp_norm = self.driver_pedal_outputs(
            drive_cycle_spd, veh_spd_mps, pid_Istate, pid_Istate2
        )
        return app_norm[0]

    def _update_bpp_zoh(self, time, state, e, **params):
        veh_spd_mps = state.continuous_state[self.velocity_state_idx]
        pid_Istate = state.continuous_state[self.driver_pid_iterm_state_idx]
        pid_Istate2 = state.continuous_state[self.driver_pid_iterm2_state_idx]
        drive_cycle_spd = 100
        app_norm, bpp_norm = self.driver_pedal_outputs(
            drive_cycle_spd, veh_spd_mps, pid_Istate, pid_Istate2
        )
        return bpp_norm[0]

    def _update_veh_spd_zoh(self, time, state, e, **params):
        veh_spd_mps = state.continuous_state[self.velocity_state_idx]
        return veh_spd_mps

    def _update_coolant_zoh(self, time, state, e, **params):
        coolant_temp_state = state.continuous_state[self.coolant_temp_state_idx]
        return coolant_temp_state

    def _discrete_update(self, time, state, e, **params):
        mot_volt_based_enable = self._update_mot_volt_enable_state(
            time, state, e, **params
        )
        app_zoh = self._update_app_zoh(time, state, e, **params)
        bpp_zoh = self._update_bpp_zoh(time, state, e, **params)
        veh_spd_zoh = self._update_veh_spd_zoh(time, state, e, **params)
        coolant_zoh = self._update_coolant_zoh(time, state, e, **params)
        return self.DiscreteStateType(
            mot_volt_based_enable=mot_volt_based_enable,
            app_zoh=app_zoh,
            bpp_zoh=bpp_zoh,
            veh_spd_zoh=veh_spd_zoh,
            coolant_zoh=coolant_zoh,
        )

    def driver_pedal_outputs(
        self, drive_cycle_spd, veh_spd_mps, pid_Istate, pid_Istate2
    ):
        Kp = 1.0
        Ki = 0.0
        Kd = 0.0
        N = 100.0
        C = jnp.array([(Ki * N), ((Kp * N + Ki) - (Kp + Kd * N) * N)])
        D = jnp.array([(Kp + Kd * N)])
        x = jnp.array([pid_Istate, pid_Istate2])
        spd_error = drive_cycle_spd - veh_spd_mps

        driver_pid = jnp.matmul(C, x) + D * spd_error
        app_norm = jnp.clip(driver_pid, 0.0, 1.0)
        bpp_norm = jnp.clip(driver_pid * -1, 0.0, 1.0)

        return app_norm, bpp_norm

    def merge_and_sat(self, p0, p1, inp):
        presat = (inp - p0) / (p1 - p0)
        return jnp.clip(presat, 0.0, 1.0)

    def axel(
        self,
        axel_vertical_load_N,
        axel_linear_spd_mps,
        axel_trq_Nm,
        axel_linear_acceleration,
        whl_radius_m,
    ):
        """
        in the UI model, the axel_linear_accel is computed locally using
        a derivative block.
        we could do this with local memory, e.g.:
        der = (inp - self.inp_prev)/(time - self.time_prev)
        i dont know how this will afect wildcat.
        """
        Crr = 0.01
        intertia_kgm2 = 0.1
        merge2zero = self.merge_and_sat(0.01, 0.1, axel_linear_spd_mps)
        rolling_drag_force_N = axel_vertical_load_N * Crr * merge2zero

        net_trq_Nm = axel_trq_Nm - axel_linear_acceleration * intertia_kgm2

        force_N = net_trq_Nm / whl_radius_m - rolling_drag_force_N
        whl_rotational_spd_radps = (
            axel_linear_spd_mps / whl_radius_m
        )  # not used. had to move else where due to topology.

        return force_N, whl_rotational_spd_radps

    def front_rear_axel_loads(self, axel_linear_acceleration, mass, g):
        veh_cg_rear_ratio = 0.4
        veh_whl_base = 1.8
        veh_cg_height = 0.8
        fg = mass * g
        Fnr_term1 = veh_cg_rear_ratio * fg
        Fnr_term2 = mass * axel_linear_acceleration * veh_cg_height / veh_whl_base
        rear_axel_vertical_load = -Fnr_term1 - Fnr_term2
        front_axel_vertical_load = -fg - rear_axel_vertical_load
        return front_axel_vertical_load, rear_axel_vertical_load

    def chassis_BBW(self, norm_brake_req, brake_ang_vel):
        friction_brake_max_torque_Nm = 3e3

        max_brake_Nm = lax.cond(
            brake_ang_vel >= 0,
            lambda: friction_brake_max_torque_Nm,
            lambda: friction_brake_max_torque_Nm * -1.0,
        )

        merge2zero = self.merge_and_sat(0.01, 0.1, brake_ang_vel)

        brake_toqrue_Nm = max_brake_Nm * merge2zero
        return brake_toqrue_Nm

    def voltage_limits(self, mot_dc_voltage, mot_volt_based_enable_state):
        min_dc_voltage = 200
        max_dc_voltage = 500
        reenable_voltage_offset = 20

        cmp3 = max_dc_voltage - reenable_voltage_offset >= mot_dc_voltage
        cmp2 = mot_dc_voltage >= reenable_voltage_offset + min_dc_voltage
        voltage_based_enable_req = cmp3 & cmp2

        cmp1 = max_dc_voltage < mot_dc_voltage
        cmp0 = mot_dc_voltage < min_dc_voltage
        voltage_based_disable_req = cmp1 | cmp0

        enable = lax.cond(
            voltage_based_enable_req,
            lambda: 1.0,
            lambda: lax.cond(
                voltage_based_disable_req,
                lambda: 0.0,
                lambda: mot_volt_based_enable_state,
            ),
        )

        return enable

    def speed_based_torque_derating(self, mot_spd_radps):
        peak_spd1 = 1500
        peak_spd2 = 1600
        presat = 1.0 - (jnp.abs(mot_spd_radps) - peak_spd1) * (peak_spd2 - peak_spd1)
        spd_based_derate = jnp.clip(presat, 0.0, 1.0)
        return spd_based_derate

    def mot_eff(self, mot_spd_radps, mot_generated_trq, mot_dc_voltage):
        # no easy way to make 2d interp.
        # just do a 1D interp for now.
        # eff_spd_vector =
        # eff_trq_vector =
        # eff_table =
        eff_pwr_vector = jnp.array([0, 1e6])
        eff_table = jnp.array([0.92, 0.92])

        mech_pwr = mot_spd_radps * mot_generated_trq
        mot_eff = jnp.interp(jnp.abs(mech_pwr), eff_pwr_vector, eff_table)
        mot_eff_sat = jnp.clip(mot_eff, 0.01, 1.0)

        mot_heat_gen_pwr = (1.0 - mot_eff_sat) * jnp.abs(mech_pwr)

        mot_volt_sat = jnp.clip(mot_dc_voltage, 1.0, 1e9)
        mot_elec_pwr = lax.cond(
            mech_pwr >= 0.0,
            lambda: mech_pwr / mot_eff_sat,
            lambda: mech_pwr * mot_eff_sat,
        )

        mot_current_A = mot_elec_pwr / mot_volt_sat

        return mot_current_A, mot_heat_gen_pwr

    def traction_motor(
        self,
        mot_spd_radps,
        mot_trq_req_norm,
        mot_dc_voltage,
        mot_rotational_accel,
        mot_volt_based_enable_state,
    ):
        peak_trq_Nm = 280
        peak_pwr = 80e3
        inertia_kgm2 = 0.01

        # controller software features
        mot_trq_req_norm_ = jnp.clip(mot_trq_req_norm, -1.0, 1.0)
        enable = self.voltage_limits(mot_dc_voltage, mot_volt_based_enable_state)
        spd_based_derate = self.speed_based_torque_derating(mot_spd_radps)

        # mech power and torque
        spd_den = jnp.clip(mot_spd_radps, 0.1, 1e30)
        peak_trq = jnp.min(jnp.array([peak_trq_Nm, peak_pwr / spd_den]))
        mot_max_trq = peak_trq * spd_based_derate
        mot_generated_trq = enable * mot_trq_req_norm_ * mot_max_trq
        inertia_trq_Nm = mot_rotational_accel * inertia_kgm2 * -1.0
        mot_trq_Nm = mot_generated_trq + inertia_trq_Nm

        # efficiency and heat gen
        mot_current_A, mot_heat_gen_pwr = self.mot_eff(
            mot_spd_radps, mot_generated_trq, mot_dc_voltage
        )

        return mot_trq_Nm, mot_current_A, mot_heat_gen_pwr

    def thermal_system(
        self, veh_spd_mps, mot_cool_fan_req, mot_heat_gen_pwr, coolant_temp_degC
    ):
        ht_coeff = 3.0
        ht_fan = 40.0
        ambient_temp_degC = 30.0
        dcdc_heat_gen_pwr = 500.0
        coolant_Cp = 4e3
        coolant_mass_kg = 1.0

        fan_based_conv = lax.cond(mot_cool_fan_req, lambda: ht_fan, lambda: 0.0)
        mm0 = jnp.max(jnp.array([veh_spd_mps * ht_coeff, fan_based_conv]))

        net_heat_loss_pwr = mm0 * (ambient_temp_degC - coolant_temp_degC)
        net_heat_flux_pwr = net_heat_loss_pwr + dcdc_heat_gen_pwr + mot_heat_gen_pwr

        der_coolant_temp_degC = net_heat_flux_pwr / (coolant_Cp * coolant_mass_kg)

        return der_coolant_temp_degC

    def batt_voltage(self, current_loop_breaker_state, batt_soc):
        battery_internal_resistance = 0.01
        soc2ocv_soc_vector = jnp.array([0.0, 1.0])
        soc2ocv_ocv_vector = jnp.array([300, 450])
        batt_resistance_vdrop = (
            current_loop_breaker_state * battery_internal_resistance * -1.0
        )
        ocv = jnp.interp(batt_soc, soc2ocv_soc_vector, soc2ocv_ocv_vector)

        batt_ccv = batt_resistance_vdrop + ocv
        return batt_ccv

    def batt_soc_der(self, pack_current_A):
        battery_capacity_kWh = 25.0
        batt_cap = battery_capacity_kWh * 3.6e6
        dSoc_dt = pack_current_A / batt_cap
        return dSoc_dt

    def quantizer(self, x, interval):
        return interval * jnp.round(x / interval)

    def one_pedal_drv_map(self, spd_kph, app_norm_):
        v1_kph = 5
        v2_kph = 20
        app1 = 0.12
        app2 = 0.15

        regen_ratio = self.merge_and_sat(v1_kph, v2_kph, jnp.abs(spd_kph))
        acc_req = self.merge_and_sat(app2, 1.0, app_norm_)
        decel_req = jnp.clip((app1 - app_norm_) / app1, 0.0, 1.0) * -1.0

        brake_req_norm = (1.0 - regen_ratio) * decel_req
        mot_req_norm = acc_req + (regen_ratio * decel_req)

        return mot_req_norm, brake_req_norm

    def thermal_management(self, coolant_temp_degC_, veh_spd_kph):
        fan_enable_thr_degC = 60
        fan_disable_spd_thr = 60
        mot_cool_fan_req = (coolant_temp_degC_ > fan_enable_thr_degC) | (
            veh_spd_kph <= fan_disable_spd_thr
        )

        return mot_cool_fan_req

    def brake_by_wire_controller(
        self, app_brake_req_norm, bpp_brake_req_norm, veh_acceleration
    ):
        gravity = -9.81
        front_bias_safety_factor = 1.05
        app_brake_max = 0.2
        veh_mass = 2e3
        veh_cg_rear_ratio = 0.4
        veh_cg_height = 0.5
        veh_whl_base = 2.8

        fg = veh_mass * gravity
        Fnr_term1 = veh_cg_rear_ratio * fg
        Fnr_term2 = veh_mass * veh_acceleration * veh_cg_height / veh_whl_base
        rear_axel_vertical_load = -Fnr_term1 - Fnr_term2
        front_axel_vertical_load = -fg - rear_axel_vertical_load

        front_bias_dynamic = (
            front_axel_vertical_load * front_bias_safety_factor / jnp.abs(fg)
        )
        front_bias_dynamic_sat = jnp.clip(front_bias_dynamic, 0.4, 0.9)

        max_ = jnp.max(
            jnp.array([app_brake_req_norm * app_brake_max, bpp_brake_req_norm])
        )

        brake_req_front_norm = front_bias_dynamic_sat * max_
        brake_req_rear_norm = (1.0 - front_bias_dynamic_sat) * max_
        return brake_req_front_norm, brake_req_rear_norm

    def controller(
        self, mot_spd_radps_, app_norm_, bpp_norm_, coolant_temp_degC_, veh_acceleration
    ):
        overall_gear_ratio = 8.0
        driven_whl_radius = 0.31
        veh_spd_mps_ = mot_spd_radps_ * driven_whl_radius / overall_gear_ratio
        veh_spd_kph = veh_spd_mps_ * 3.6

        mot_req_norm, app_brake_req_norm = self.one_pedal_drv_map(
            veh_spd_kph, app_norm_
        )
        mot_cool_fan_req = self.thermal_management(coolant_temp_degC_, veh_spd_kph)
        brake_req_front_norm, brake_req_rear_norm = self.brake_by_wire_controller(
            app_brake_req_norm, bpp_norm_, veh_acceleration
        )
        mot_req_norm_ = self.quantizer(mot_req_norm, 0.002)
        brake_req_front_norm_ = self.quantizer(brake_req_front_norm, 0.002)
        brake_req_rear_norm_ = self.quantizer(brake_req_rear_norm, 0.002)

        return (
            mot_req_norm_,
            brake_req_front_norm_,
            brake_req_rear_norm_,
            mot_cool_fan_req,
        )

    def _update_mot_volt_enable_state(self, time, state, e, **params):
        enable_prev = state.discrete_state.mot_volt_based_enable
        batt_soc = state.continuous_state[self.battery_soc_state_idx]
        current_loop_breaker_state = state.continuous_state[
            self.current_loop_breaker_state_idx
        ]
        mot_dc_voltage = self.batt_voltage(current_loop_breaker_state, batt_soc)

        # this is lame. this function gets called each time continuous derivative are needed,
        # and again when we need to update the state.
        enable = self.voltage_limits(mot_dc_voltage, enable_prev)
        return enable

    def driver_pid_state_der(
        self, drive_cycle_spd, veh_spd_mps, pid_Istate, pid_Istate2
    ):
        N = 100
        A = jnp.array([[0, 1], [0, -N]])
        B = jnp.array([0, 1])
        x = jnp.array([pid_Istate, pid_Istate2])

        spd_error = drive_cycle_spd - veh_spd_mps

        der = jnp.matmul(A, x) + B * spd_error
        # print(f"[driver_pid_state_der] der.shape={der.shape} der={der}")

        return der[0], der[1]  # split to get scalars

    def ode(self, time, state, *inputs, **parameters):
        # params
        g = -9.81
        rho = 1.2
        Cd = 0.8
        frontal_area = 1.2
        whl_radius_m = 0.31
        gear_ratio = 8.0
        mass = parameters["mass"]

        # extract continuious states
        veh_spd_mps = state.continuous_state[self.velocity_state_idx]
        # veh_pos = state.continuous_state[self.position_state_idx]
        accel_loop_breaker_state = state.continuous_state[
            self.accel_loop_breaker_state_idx
        ]
        pid_Istate = state.continuous_state[self.driver_pid_iterm_state_idx]
        pid_Istate2 = state.continuous_state[self.driver_pid_iterm2_state_idx]
        coolant_temp_degC = state.continuous_state[self.coolant_temp_state_idx]
        batt_soc = state.continuous_state[self.battery_soc_state_idx]
        current_loop_breaker_state = state.continuous_state[
            self.current_loop_breaker_state_idx
        ]

        # extract discrete states
        mot_volt_based_enable_state = state.discrete_state.mot_volt_based_enable_state
        app_zoh_state = state.discrete_state.app_zoh
        bpp_zoh_state = state.discrete_state.bpp_zoh
        veh_spd_zoh_state = state.discrete_stat.veh_speed_zoh
        coolant_zoh_state = state.discrete_state.coolant_zoh

        # speed/acceleration conversions that needed to happen before anything else due to dependency
        # these have been extracted from axel and gear_reduction submodels
        whl_rotational_spd_radps = veh_spd_mps / whl_radius_m
        mot_rotational_spd_radps = whl_rotational_spd_radps * gear_ratio
        axel_linear_acceleration = accel_loop_breaker_state * 10.0
        whl_rotational_accel = axel_linear_acceleration / whl_radius_m
        mot_rotational_accel = whl_rotational_accel * gear_ratio
        # discrete conversions
        whl_rotational_spd_radps_d = veh_spd_zoh_state / whl_radius_m
        mot_rotational_spd_radps_d = whl_rotational_spd_radps_d * gear_ratio

        # drive cycle speed
        drive_cycle_spd = 100

        # driver
        # note: we dont need to call this here because the outputs
        # are only ever used to update ZOH state. so this function
        # is called for those state updates.
        # app_norm, bpp_norm = self.driver_pedal_outputs(
        #     drive_cycle_spd, veh_spd_mps, pid_Istate, pid_Istate2
        # )
        # note: the driver PID continuous state update is done below.

        # controller
        mot_spd_radps_ = self.quantizer(mot_rotational_spd_radps_d, 0.01)
        app_norm_ = self.quantizer(app_zoh_state, 0.002)
        bpp_norm_ = self.quantizer(bpp_zoh_state, 0.002)
        coolant_temp_degC_ = self.quantizer(coolant_zoh_state, 0.1)

        (
            mot_req_norm,
            brake_req_front_norm,
            brake_req_rear_norm,
            mot_cool_fan_req,
        ) = self.controller(
            mot_spd_radps_,
            app_norm_,
            bpp_norm_,
            coolant_temp_degC_,
            axel_linear_acceleration,
        )

        # battery
        batt_ccv = self.batt_voltage(current_loop_breaker_state, batt_soc)

        # traction motor
        mot_trq_Nm, mot_current_A, mot_heat_gen_pwr = self.traction_motor(
            mot_rotational_spd_radps,
            1.0,
            batt_ccv,
            mot_rotational_accel,
            mot_volt_based_enable_state,
        )

        # Chassis BBW
        front_brake_trq_Nm = self.chassis_BBW(0.0, whl_rotational_spd_radps)
        rear_brake_trq_Nm = self.chassis_BBW(0.0, whl_rotational_spd_radps)

        # Chassis_longitudinal_dynamics
        front_axel_vertical_load, rear_axel_vertical_load = self.front_rear_axel_loads(
            axel_linear_acceleration, mass, g
        )

        front_axel_net_trq_Nm = mot_trq_Nm * gear_ratio + front_brake_trq_Nm
        front_axel_force_N, front_axel_whl_rot_radps = self.axel(
            front_axel_vertical_load,
            veh_spd_mps,
            front_axel_net_trq_Nm,
            axel_linear_acceleration,
            whl_radius_m,
        )
        rear_axel_force_N, front_axel_whl_rot_radps = self.axel(
            rear_axel_vertical_load,
            veh_spd_mps,
            rear_brake_trq_Nm,
            axel_linear_acceleration,
            whl_radius_m,
        )
        areo_drag_N = 0.5 * rho * Cd * frontal_area * veh_spd_mps**2

        net_force = front_axel_force_N + rear_axel_force_N - areo_drag_N

        accel = net_force / mass

        # loop breakers
        accel_loop_breaker_der = accel + accel_loop_breaker_state * (-10.0)
        current_loop_breaker_der = (
            mot_current_A * -1.0 * -1.0
        ) + current_loop_breaker_state * (-1.0)

        # thermal_system
        der_coolant_temp_degC = self.thermal_system(
            veh_spd_mps, mot_cool_fan_req, mot_heat_gen_pwr, coolant_temp_degC
        )

        # battery soc derivative
        dSoc_dt = self.batt_soc_der(mot_current_A * -1.0)

        # driver state update
        pid_Istate_der, pid_Istate2_der = self.driver_pid_state_der(
            drive_cycle_spd, veh_spd_mps, pid_Istate, pid_Istate2
        )

        return jnp.array(
            [
                accel,
                veh_spd_mps,
                accel_loop_breaker_der,
                pid_Istate_der,
                der_coolant_temp_degC,
                dSoc_dt,
                current_loop_breaker_der,
                pid_Istate2_der,
            ]
        )
