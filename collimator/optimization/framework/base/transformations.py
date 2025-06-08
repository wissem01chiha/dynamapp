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

"""
Base clases for transformations and pre-built transformations.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp


class Transform(ABC):
    """Base class for transformations."""

    @abstractmethod
    def transform(self, params: dict) -> dict:
        """
        Take original parameters dict {key:value} and output a dict with identical keys
        but transformed `values`.
        """
        pass

    @abstractmethod
    def inverse_transform(self, params: dict) -> dict:
        """
        Take transformed parameters dict {key:value} and output a dict with identical
        keys but inverse-transformed `values`.
        """
        pass


class CompositeTransform(Transform):
    """
    A composite transformation that applies a list of transformations in sequence.
    """

    def __init__(self, transformations):
        self.transformations = transformations

    def transform(self, params: dict):
        for transformation in self.transformations:
            params = transformation.transform(params)
        return params

    def inverse_transform(self, params: dict):
        for transformation in reversed(self.transformations):
            params = transformation.inverse_transform(params)
        return params


class IdentityTransform(Transform):
    """
    A transformation that does nothing: ``` y = x ```.
    """

    def transform(self, params: dict):
        return params

    def inverse_transform(self, params: dict):
        return params


class LogTransform(Transform):
    """
    A transformation that applies the natural logarithm to the values of the parameters.
    ``` y = log(x) ```.
    """

    def transform(self, params: dict):
        return {k: jnp.log(v) for k, v in params.items()}

    def inverse_transform(self, params: dict):
        return {k: jnp.exp(v) for k, v in params.items()}


class NegativeNegativeLogTransform(Transform):
    """
    A transformation that applies the negative of the natural logarithm of the negative
    of the values of the parameters.
    ``` y = -log(-x) ```
    """

    def transform(self, params: dict):
        return {k: -jnp.log(-v) for k, v in params.items()}

    def inverse_transform(self, params: dict):
        return {k: -jnp.exp(-v) for k, v in params.items()}


class NormalizeTransform(Transform):
    """
    A transformation that normalizes the values of the parameters to the range [0, 1].
    ``` y = (x - min) / (max - min) ```
    Paramteters:
        - params_min: dict with the minimum values for each parameter.
        - params_max: dict with the maximum values for each parameter.
    """

    def __init__(self, params_min: dict, params_max: dict):
        self.params_min = params_min
        self.params_max = params_max

    def transform(self, params: dict):
        return {
            k: (v - self.params_min[k]) / (self.params_max[k] - self.params_min[k])
            for k, v in params.items()
        }

    def inverse_transform(self, params: dict):
        return {
            k: v * (self.params_max[k] - self.params_min[k]) + self.params_min[k]
            for k, v in params.items()
        }


class LogitTransform(Transform):
    """
    The logit transformation, defined as:
    ``` y = log(x / (1 - x)) ```
    """

    def transform(self, params: dict):
        return {k: jnp.log(v / (1.0 - v)) for k, v in params.items()}

    def inverse_transform(self, params: dict):
        return {k: 1.0 / (1.0 + jnp.exp(-v)) for k, v in params.items()}
