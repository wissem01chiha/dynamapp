# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Definitions copied (almost) verbatim from JAX source code

from typing import Any, Callable

def io_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: None,
    ordered: bool = False,
    **kwargs: Any,
):
    """Calls an impure Python callback.

    For more explanation, see `External Callbacks`_.

    Args:
      callback: function to execute on the host. It is assumet to be an impure function.
        If ``callback`` is pure, using :func:`jax.pure_callback` instead may lead to
        more efficient execution.
      result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
        whose structure matches the expected output of the callback function at runtime.
        :class:`jax.ShapeDtypeStruct` is often used to define leaf values.
      *args: arguments to be passed to the callback function
      sharding: optional sharding that specifies the device from which the callback should
        be invoked.
      ordered: boolean specifying whether sequential calls to callback must be ordered.
      **kwargs: keyword arguments to be passed to the callback function

    Returns:
      result: a pytree of :class:`jax.Array` objects whose structure matches that of
        ``result_shape_dtypes``.

    See Also:
      - :func:`jax.pure_callback`: callback designed for pure functions.
      - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
      - :func:`jax.debug.print`: callback designed for printing.

    .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
    """
