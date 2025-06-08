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

"""Blocks for random number generation."""

from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from jax import random
import jax.numpy as jnp
import numpy as np

from ..framework import LeafSystem, parameters

if TYPE_CHECKING:
    from ..backend.typing import Array, DTypeLike, ShapeLike


__all__ = [
    "RandomNumber",
    "WhiteNoise",
]


class RandomNumber(LeafSystem):
    """Discrete-time random number generator.

    Generates independent, identically distributed random numbers at each time step.
    Dispatches to `jax.random` for the actual random number generation.

    Supported distributions include "ball", "cauchy", "choice", "dirichlet",
    "exponential", "gamma", "lognormal", "maxwell", "normal", "orthogonal",
    "poisson", "randint", "truncated_normal", and "uniform".

    See https://jax.readthedocs.io/en/latest/jax.random.html#random-samplers
    for a full list of available distributions and associated parameters.

    Although the JAX random number generator is a deterministic function of the
    key, this block maintains the key as part of the discrete state, making it a
    stateful RNG.  The block can be seeded for reproducibility by passing an integer
    seed; if None, a random seed will be generated using numpy.random.

    Note that this block should typically not be used as a source of randomness for
    continuous-time systems, as it generates a discrete-time signal. For continuous
    systems, use a continuous-time noise source, such as `WhiteNoise`.

    Input ports:
        None

    Output ports:
        (0) The most recently generated random number.

    Parameters:
        dt: The rate at which random numbers are generated.
        distribution: The name of the random distribution to sample from.
        seed: An integer seed for the random number generator. If None, a random 32-bit
            seed will be generated.
        dtype: data type of the random number.  If None, the default data type for the
            specified distribution will be used.  Not all distributions support all
            data types; check the JAX documentation for details.
        distribution_parameters: A dictionary of additional parameters to pass to the
            distribution function.
    """

    class RNGState(NamedTuple):
        key: Array
        val: Array

    @parameters(static=["distribution", "seed", "shape"])
    def __init__(
        self,
        dt: float,
        distribution: str = "normal",  # UI only exposes 'normal' for now
        seed: int = None,
        dtype: DTypeLike = None,
        shape: ShapeLike = (),
        name: str = None,
        ui_id: str = None,
        **distribution_parameters,
    ):
        super().__init__(name=name, ui_id=ui_id)

        # Declare config parameters for serialization
        self.declare_static_parameters(**distribution_parameters)

        # Add to the data type if specified.  Since not all distributions
        # support this parameter (though most do), we don't want to do this
        # unconditionally.
        if dtype is not None:
            distribution_parameters["dtype"] = dtype

        self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
        )

        self.declare_periodic_update(
            self._update,
            period=dt,
            offset=0.0,
        )

    def initialize(
        self,
        distribution: str = "normal",  # UI only exposes 'normal' for now
        seed: int = None,
        shape: ShapeLike = (),
        **distribution_parameters,
    ):
        # Supposedly all distributions support the shape parameter
        if shape is not None and shape != ():
            distribution_parameters["shape"] = shape

        self.rng = partial(getattr(random, distribution), **distribution_parameters)

        key = random.PRNGKey(np.random.randint(0, 2**32) if seed is None else seed)

        # The discrete state is a tuple of (key, val) pairs.  Because of the way that
        # JAX maintains RNG state, we need to keep track of the key as well as the
        # most recently generated value.
        key, subkey = random.split(key)
        default_state = self.RNGState(
            key=key,
            val=self.rng(subkey),  # Initial random number with the right data type
        )
        self.declare_discrete_state(default_value=default_state, as_array=False)

    def _output(self, _time, state, *_inputs, **_parameters):
        return state.discrete_state.val

    def _update(self, _time, state, *_inputs, **_parameters):
        key, subkey = random.split(state.discrete_state.key)
        return self.RNGState(
            key=key,
            val=self.rng(subkey),
        )


class WhiteNoise(LeafSystem):
    """Continuous-time white noise generator.

    Generates a band-limited white noise signal using a sinc-interpolated random
    number generator.  The output signal is a continuous-time signal, but the
    underlying random number generator is discrete-time.  As a result, the signal
    is not truly white, but is band-limited by the sample rate.  The resulting signal
    has the following approximate power spectral density:
    ```
    S(f) = A * fs if |f| < fs else 0,
    ```
    where `A` is the noise power and `fs = 1/dt` is the sample rate.

    See Ch. 10.4 in Baraniuk, "Signal Processing and Modeling" for details:
        https://shorturl.at/floRZ

    The output signal will have variance `A`, zero mean, and will decorrelate at
    the sample rate.

    Input ports:
        None

    Output ports:
        (0) The band-limited white noise signal with variance `noise_power`, zero
            mean, and correlation time `dt`.

    Parameters:
        correlation_time: The correlation time of the output signal and the inverse of
            the bandwidth. It is the sample frequency of the underlying random number
            generator.
        noise_power: The variance of the white noise signal. Also scales the amplitude
            of the power spectral density.
        num_samples: The number of samples to use for sinc interpolation.  More samples
            will result in a more accurate approximation of the ideal power spectrum,
            but will also increase the computational cost.  The default of 10 is
            sufficient for most applications.
        seed: An integer seed for the random number generator. If None, a random 32-bit
            seed will be generated.
        dtype: data type of the random number.  If None, defaults to float.
        shape: The shape of the output signal.  If empty, the output will be a scalar.
    """

    class RNGState(NamedTuple):
        key: Array
        samples: Array
        t_last: float = 0.0

    @parameters(
        static=["num_samples", "shape", "seed"],
        dynamic=["correlation_time", "noise_power"],
    )
    def __init__(
        self,
        correlation_time,
        noise_power: float = 1.0,
        num_samples: int = 10,
        seed: int = None,
        dtype: DTypeLike = None,
        shape: ShapeLike = (),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dtype = dtype

        self.declare_output_port(self._output)
        self.declare_periodic_update(
            self._update,
            period=correlation_time,
            offset=0.0,
        )

    def initialize(
        self,
        correlation_time,
        noise_power: float = 1.0,
        num_samples: int = 10,
        seed: int = None,
        shape: ShapeLike = (),
    ):
        self.shape = tuple(map(int, shape))
        self.N = num_samples

        self.noise_power = noise_power
        self.shift = np.arange(self.N) - (self.N - 1) / 2
        self.rng = partial(random.normal, dtype=self.dtype)

        # The default state is a tuple of (key, samples) pairs.  The continuous-time
        # output signal is reconstructed from the samples using a sinc interpolation.
        seed = np.random.randint(0, 2**32) if seed is None else int(seed)
        key = random.PRNGKey(int(seed))
        key, subkey = random.split(key)
        default_state = self.RNGState(
            key=key,
            samples=self.sample(subkey, shape=(self.N, *self.shape)),
        )
        self.declare_discrete_state(default_value=default_state, as_array=False)

    def sample(self, key, shape):
        return jnp.sqrt(self.noise_power) * self.rng(key, shape)

    def _output(self, time, state, *_inputs, **parameters):
        t_last = state.discrete_state.t_last

        # Time relative to the last discrete sample, in units of
        # samples.  This is the argument to the sinc function.
        w = (time - t_last) / parameters["correlation_time"] - self.shift

        # Clip the time values to limit discontinuities resulting
        # from sample updates.
        w = jnp.clip(w, -self.N // 2, self.N // 2)

        # Shift the axes so that the last axis is the sample index.
        # This is the index that will be contracted over
        samples = jnp.moveaxis(state.discrete_state.samples, 0, -1)

        return jnp.sum(samples * jnp.sinc(w), axis=-1)

    def _update(self, time, state, *_inputs, **_parameters):
        key, subkey = random.split(state.discrete_state.key)

        new_sample = self.sample(subkey, (1, *self.shape))
        samples = jnp.concatenate((state.discrete_state.samples[1:], new_sample))

        return self.RNGState(
            key=key,
            samples=samples,
            t_last=time,
        )
