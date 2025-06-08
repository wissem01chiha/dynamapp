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

import numpy as np  # needed so black in CI doesn't give error

R = 8.3145  # J/(mol*K)

O2_molar_mass = 0.032  # kg/mol
CH4_molar_mass = 0.016  # kg/mol

R_O2 = R / O2_molar_mass  # J/(kg*K)
R_CH4 = R / CH4_molar_mass  # J/(kg*K)

cp_O2 = 29.4 / O2_molar_mass  # J/(kg*K)
cp_CH4 = 35.7 / CH4_molar_mass  # J/(kg*K)
cv_O2 = cp_O2 - R_O2
cv_CH4 = cp_CH4 - R_CH4

delta_H = -802.51e3  # Enthalpy of combustion for CH4 + 2O2 (J/mol)

gamma_O2 = cp_O2 / cv_O2
gamma_CH4 = cp_CH4 / cv_CH4

ox_mass_frac = 0.78
fuel_mass_frac = 1 - ox_mass_frac
R_avg = R_O2 * ox_mass_frac + R_CH4 * fuel_mass_frac
cp_avg = cp_O2 * ox_mass_frac + cp_CH4 * fuel_mass_frac
cv_avg = cv_O2 * ox_mass_frac + cv_CH4 * fuel_mass_frac

propellant_mass = 1e4  # kg

ox_tank_temperature = 90  # K
fuel_tank_temperature = 110  # K

ox_tank_pressure_nominal = 0.35e6  # Pa
fuel_tank_pressure_nominal = 1.0e6  # Pa

ox_density = 1141  # kg/m^3
fuel_density = 480  # kg/m^3

init_fill_frac = 0.99

# ox_tank_volume = propellant_mass * ox_mass_frac * R * ox_tank_temperature / ox_tank_pressure_nominal  # m^3
# fuel_tank_volume = propellant_mass * fuel_mass_frac * R * fuel_tank_temperature / fuel_tank_pressure_nominal  # m^3
ox_tank_volume = propellant_mass * ox_mass_frac / ox_density / init_fill_frac
fuel_tank_volume = propellant_mass * fuel_mass_frac / fuel_density / init_fill_frac

# Fill gas density (ideal gas law)
init_ox_gas_density = ox_tank_pressure_nominal / (R_O2 * ox_tank_temperature)  # kg/m^3
init_ox_gas_mass = init_ox_gas_density * (1 - init_fill_frac) * ox_tank_volume
init_fuel_gas_density = fuel_tank_pressure_nominal / (R_CH4 * fuel_tank_temperature)
init_fuel_gas_mass = init_fuel_gas_density * (1 - init_fill_frac) * fuel_tank_volume

tank_outlet_cross_section = 0.003  # m^2
preburner_chamber_cross_section = np.pi * 0.4**2  # m^2
preburner_expansion_ratio = 20
chamber_cross_section = np.pi * 0.7**2  # m^2
