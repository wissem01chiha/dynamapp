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

import numpy as np
import control

m = 1.0  # Pole mass
M = 5.0  # Cart mass
L = 2.0  # Pole length
g = 9.8
d = 1.0  # Damping


def make_sys(up=True):
    b = 1 if up else -1
    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, m * g / M, -d / M, 0],
            [0, b * (m + M) * g / (M * L), -b * d / (M * L), 0],
        ]
    )
    B = np.array([[0, 0, 1 / M, b / (M * L)]]).T
    C = np.array([[1, 0, 0, 0]])
    D = [0]

    return control.ss(A, B, C, D)


sys = make_sys(up=True)
(A, B, C, D) = (sys.A, sys.B, sys.C, sys.D)
q_ref = np.array([0, np.pi, 0, 0])

n = A.shape[0]
m = B.shape[1]
p = C.shape[0]

#
# LQG design (inverted configuration)
#
Q = np.eye(n)
R = 1e-6 * np.eye(m)
K = control.lqr(sys, Q, R)[0]

#
# Kalman filter design
#
sigma_d = 0.01  # Disturbance amplitude
sigma_n = 0.01  # Measurement noise
Vd = sigma_d**2 * np.eye(n)  # Disturbance covariance
Vn = sigma_n**2 * np.eye(p)  # Measurement noise covariance

Kf = control.lqr(A.T, C.T, Vd, Vn)[0].T
