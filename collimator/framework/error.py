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

import warnings
from typing import TYPE_CHECKING, Hashable, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from .port import DirectedPortLocator


@dataclass
class LoopItem:
    name_path: list[str] = None
    ui_id_path: list[str] = None
    port_direction: str = None
    port_index: int = None
    # port_name: str


class CollimatorError(Exception):
    """Base class for all custom collimator errors."""

    # Ideally we'd always have a system to pass but there are at least 2 cases
    # where we may not have one:
    # 1. parsing from json, the block hasn't been built yet
    # 2. other errors not specific to a block
    # In case 1, we should pass all name, path & ui_id info to the error

    def __init__(
        self,
        message=None,
        *,
        system: "SystemBase" = None,  # noqa
        system_id: Hashable = None,
        name_path: list[str] = None,
        ui_id_path: list[str] = None,
        port_index: int = None,
        port_name: str = None,
        port_direction: str = None,  # 'in' or 'out'
        parameter_name: str = None,
        loop: list["DirectedPortLocator"] = None,
    ):
        """Create a new CollimatorError.

        Only `message` is a positional argument, all others are keyword arguments.

        Args:
            message: A custom error message, defaults to the error class name.
            system: The system that the error occurred in, if available.
            system_id: The id of the system that the error occurred in, use if system can't be passed.
            name_path: The name path of the block that the error occurred in, use if system can't be passed.
            ui_id_path: The ui_id (uuid) path of the block that the error occurred in, use if system can't be passed.
            port_index: The index of the port that the error occurred at.
            port_name: The name of the port that the error occurred at.
            port_direction: The direction of the port that the error occurred at.
            parameter_name: The name of the parameter that the error occurred at.
            loop: A list of I/O ports where the error occurred (eg. AlgebraicLoopError).
        """
        super().__init__(message)

        if system and system_id:
            warnings.warn(
                "Should not specify both system and system_id when raising exceptions"
            )

        if system:
            self.system_id = system.system_id
            self.name_path = name_path or system.name_path
            self.ui_id_path = ui_id_path or system.ui_id_path
        else:
            self.system_id = system_id
            self.name_path = name_path
            self.ui_id_path = ui_id_path

        self.message = message
        self.port_index = port_index
        self.port_name = port_name
        self.port_direction = port_direction
        self.parameter_name = parameter_name

        # Extract serializable info from loop
        # NOTE: we could compact it a bit if the JSON becomes too large...
        self.loop: list[LoopItem] = None
        if loop is not None:
            self.loop = [
                LoopItem(
                    name_path=loc[0].name_path,
                    ui_id_path=loc[0].ui_id_path,
                    port_direction=loc[1],
                    port_index=loc[2],
                )
                for loc in loop
            ]

    def __str__(self):
        message = self.message or self.default_message
        return f"{message}{self._context_info()}"

    def _context_info(self) -> str:
        strbuf = []

        if self.name_path:
            # FIXME: this is known to be too verbose when looking at errors from
            # the UI but makes it better when running pytest or from code.
            # For now, be verbose.
            name_path = ".".join(self.name_path)
            strbuf.append(f" in block {name_path}")
        elif self.system_id:  # Unnamed blocks, likely from code
            strbuf.append(f" in system {self.system_id}")

        if self.port_direction:
            strbuf.append(
                f" at {self.port_direction}put port {self.port_name or self.port_index}"
            )
        elif self.port_name:
            strbuf.append(f" at port {self.port_name}")
        elif self.port_index is not None:
            strbuf.append(f" at port {self.port_index}")
        if self.parameter_name:
            strbuf.append(f" with parameter {self.parameter_name}")
        if self.__cause__ is not None:
            strbuf.append(f": {self.__cause__}")

        return "".join(strbuf)

    @property
    def block_name(self):
        if self.name_path is None:
            return None
        if len(self.name_path) == 0:
            return "root"
        return self.name_path[-1]

    @property
    def default_message(self):
        return type(self).__name__

    def caused_by(self, exc_type: type):
        """Check if this error is or was caused by another error type.

        For instance, if a CollimatorError is raised because of a TypeError,
        this method will return True when called with TypeError as exc_type.

        Args:
            exc_type: The type of exception to check for (eg. TypeError)

        Returns:
            bool: True if the error is or was caused by the given exception type.
        """

        def _is_or_caused_by(exc, cause_type) -> bool:
            if not exc or not cause_type:
                return False
            if isinstance(exc, cause_type):
                return True
            if not hasattr(self, "__cause__"):
                return False
            return _is_or_caused_by(exc.__cause__, cause_type)

        return _is_or_caused_by(self, exc_type)


class StaticError(CollimatorError):
    """Wraps a Python exception to record the offending block id. The original
    exception is found in the '__cause__' field.

    See collimator.framework.context_factory._check_types for use.

    This is called 'static' (as opposed to say 'runtime') meaning this is for
    wrapping errors detected prior to running a simulation."""

    pass


class BlockParameterError(StaticError):
    """Block parameters are missing or have invalid values."""

    pass


class ShapeMismatchError(StaticError):
    """Block parameters or input/outputs have mismatched shapes."""

    def __init__(self, expected_shape=None, actual_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape

    def __str__(self):
        if self.expected_shape or self.actual_shape:
            return (
                f"Shape mismatch: expected {self.expected_shape}, "
                f"got {self.actual_shape}" + self._context_info()
            )
        return f"Shape mismatch{self._context_info()}"


class DtypeMismatchError(StaticError):
    """Block parameters or input/outputs have mismatched dtypes."""

    def __init__(self, expected_dtype=None, actual_dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.expected_dtype = expected_dtype
        self.actual_dtype = actual_dtype

    def __str__(self):
        if self.expected_dtype or self.actual_dtype:
            return (
                f"Data type mismatch: "
                f"expected {self.expected_dtype}, got {self.actual_dtype}"
                + self._context_info()
            )
        return f"Dtype mismatch{self._context_info()}"


class ParameterError(StaticError):
    """Error raised during parameter processing."""

    def __init__(self, parameter, **kwargs):
        super().__init__(**kwargs)
        self.parameter = parameter

    def __str__(self):
        return f"Parameter error: {self.message}: {self.parameter.__repr__()}"


class BlockInitializationError(CollimatorError):
    """A generic error to be thrown when a block fails at init time, but
    the full exceptions are known to cause issues, eg. with ray serialization.
    """

    pass


class ModelInitializationError(CollimatorError):
    """A generic error to be thrown when a model fails at init time."""

    pass


class BlockRuntimeError(CollimatorError):
    """A generic error to be thrown when a block fails at runtime, but
    the full exceptions are known to cause issues, eg. with ray serialization.
    """

    pass


class InputNotConnectedError(StaticError):
    """An input port to the system is not connected to a valid signal source."""

    pass


class PythonScriptError(BlockRuntimeError):
    """An error occurred in a PythonScript block."""

    pass


class PythonScriptTimeNotSupportedError(PythonScriptError):
    """PythonScript block does not support implicit time variable"""

    # this is a breaking change from previous versions of collimator, thus
    # the specific error type and message

    @property
    def default_message(self):
        return (
            "The PythonScript block does not support an implicit variable for 'time'. "
            "The block's state and initial outputs should be defined in the block's "
            "'init_script'. If the value of 'time' is needed in the 'step' script, "
            "then a 'Clock' block can be connected as input to this PythonScript block"
        )


class LegacyBlockConfigurationError(StaticError):
    """A block is not supported in the current version of Collimator."""

    # this can be used for blocks that have breaking changes we are not willing
    # to support automatically

    @property
    def default_message(self):
        return (
            "This block has a configuration of parameters or input/output ports "
            "incompatible with the current version of the Collimator simulation "
            "engine. Please re-create the block manually in UI or review the "
            "instantiation code."
        )


class ErrorCollector:
    """
    Tool used to collect errors related to users model specification.
    Errors related to user model specification are identified during
    model static analysis, e.g. context creation, type checking, etc.

    An instance of this tool can be created, and then passed down a
    tree of function calls to collect errors found any where in
    the tree. Locally in the tree it can be determined whether it is
    ok to continue or not. This tool enables collecting errors up until
    the point when continuation is no longer possible.

    Note: this latter behavior, where sometimes there is early exit desired,
    and all other "pipeline" operations are "nullified", might better be
    implemented using pymonad:Either class.
    """

    def __init__(self):
        self._disable_collection = False
        self._parent: Optional["ErrorCollector"] = None
        self.errors: list[BaseException] = []

    def add_error(self, error: BaseException):
        """Add an error to the collection."""

        if self._parent is not None:
            self._parent.add_error(error)
            return

        if not self._disable_collection:
            self.errors.append(error)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Return values: True to suppress the exception, False to propagate it

        if exc_type is not None:
            if self._parent is not None:
                self._parent.add_error(exc_value)
                return True

            self.add_error(exc_value)
            return False

        return True

    @classmethod
    def context(cls, parent: "ErrorCollector" = None) -> "ErrorCollector":
        """A context manager convenience to use when tracing errors.

        Use as:
        ```
        with ErrorCollector.trace(error_context) as ec:
            ...
        ```

        If the parent context is None, then exceptions will pass through without
        being collected. Else, exceptions will be collected in the parent context.
        """

        if parent is None:
            ctx = cls()
            ctx._disable_collection = True
            return ctx

        ctx = cls()
        ctx._parent = parent
        return ctx
