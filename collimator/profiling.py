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
import logging
import time
import os

from .backend import IS_JAXLITE

if not IS_JAXLITE:
    import jaxlib


class Profiler:
    _profiles = {}
    _counts = {}
    _cumulative_profiles = {}
    _min_profiles = {}
    _max_profiles = {}
    _logger = logging.getLogger("collimator_profiler")
    _disable_profiling = os.environ.get("COLLIMATOR_DISABLE_PROFILING", "0") == "1"
    # JAX JIT profiling must be disabled by default because it breaks various
    # models and advanced scenarios (autodiff, etc.)
    _enable_jaxjit_profiling = os.environ.get("COLLIMATOR_JAXJIT_PROFILING", "0") == "1"
    _jaxjit_or_abort = os.environ.get("COLLIMATOR_JAXJIT_OR_ABORT", "0") == "1"

    class ScopedProfiler:
        """Use as a context manager to profile a block of code:

        with Profiler.ScopedProfiler("my_block"):
            do_something()

        This supports exceptions well.
        """

        def __init__(self, name, show_start=False):
            self.name = name
            self.show_start = show_start

        def __enter__(self):
            Profiler.start(self.name, show_start=self.show_start)

        def __exit__(self, exc_type, exc_val, exc_tb):
            Profiler.stop(self.name, raised=exc_type is not None)

    @classmethod
    def start(cls, name, show_start=False):
        if cls._disable_profiling:
            return

        cls._profiles[name] = time.perf_counter()
        if show_start:
            cls._logger.debug("%s: start", name)

    @classmethod
    def stop(cls, name, raised=False):
        if cls._disable_profiling:
            return

        t1 = time.perf_counter()
        t0 = cls._profiles[name]
        dt = t1 - t0
        count = cls._counts.get(name, 0) + 1
        cumulative = cls._cumulative_profiles.get(name, 0) + dt
        min_profile = cls._min_profiles.get(name, dt)
        max_profile = cls._max_profiles.get(name, dt)
        cls._counts[name] = count
        cls._cumulative_profiles[name] = cumulative
        cls._min_profiles[name] = min(min_profile, dt)
        cls._max_profiles[name] = max(max_profile, dt)
        cls._logger.debug(
            "%s: %.3fms (count: %d, cumulative: %.3fs, min: %.3fms, max: %.3fms, avg: %.3fms)%s",
            name,
            dt * 1000.0,
            count,
            cumulative,
            min_profile * 1000.0,
            max_profile * 1000.0,
            cumulative / count * 1000.0,
            " (raised)" if raised else "",
        )
        del cls._profiles[name]

    @classmethod
    def clear(cls):
        """Clear all profiling data"""
        cls._profiles = {}
        cls._counts = {}
        cls._cumulative_profiles = {}
        cls._min_profiles = {}
        cls._max_profiles = {}

    @classmethod
    def profile(cls, name: str = None):
        if cls._disable_profiling:
            return lambda x: x

        def decorator(func):
            def wrapper(_profiling_name: str, *args, **kwargs):
                with cls.ScopedProfiler(_profiling_name):
                    return func(*args, **kwargs)

            return partial(wrapper, name if name else func.__name__)

        return decorator

    @classmethod
    def profiledfunc(cls, fn, name: str = None):
        if cls._disable_profiling:
            return fn

        name = name if name else fn.__name__

        def wrapper(*args, **kwargs):
            with cls.ScopedProfiler(name):
                return fn(*args, **kwargs)

        return wrapper

    @classmethod
    def jaxjit_profiledfunc(cls, fn, name=None):
        """Profile a JAX JIT function using AOT"""
        if cls._disable_profiling or IS_JAXLITE:
            return fn

        name = name if name else fn.__name__

        if not cls._enable_jaxjit_profiling or not isinstance(
            fn, jaxlib.xla_extension.PjitFunction
        ):
            if cls._enable_jaxjit_profiling:
                if cls._jaxjit_or_abort:
                    raise RuntimeError(
                        f"COLLIMATOR_JAXJIT_OR_ABORT=1 but {name} isn't a jitted function"
                    )
                cls._logger.warning(
                    "JAX JIT profiling enabled but %s isn't jitted",
                    name,
                )
            return cls.profiledfunc(fn, name=name)

        cls._logger.debug("Profiling JAX JIT function %s", name)
        cls._logger.debug(
            "If compilation fails, consider setting COLLIMATOR_JAXJIT_PROFILING=0"
            + " (or unset it))"
        )

        # Splitting jax.jit into lower/compile/execute steps kinda helps
        # with profiling but it's know to:
        # - break various models
        # - actually give inconsistent timings vs. just jitting all at once
        # (sum does not equal parts)
        def _func(_function_to_profile, *args, **kwargs):
            with cls.ScopedProfiler(f"jit:{name}:lower", show_start=True):
                _function_to_profile = _function_to_profile.lower(*args, **kwargs)
            with cls.ScopedProfiler(f"jit:{name}:compile", show_start=True):
                _function_to_profile = _function_to_profile.compile()
            with cls.ScopedProfiler(f"jit:{name}:execute", show_start=True):
                return _function_to_profile(*args, **kwargs)

        return partial(_func, fn)
