# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A LazyLoader class.

Mostly copied from
https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/python/util/lazy_loader.py
"""

import importlib
import types

from collimator.logging import logger


_LAZY_LOADER_PREFIX = "_ll"


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    `control`, and `sympy` are examples of modules that are large and not always
    needed, and this allows them to only be loaded when they are used.
    """

    # The lint error here is incorrect.
    def __init__(
        self, local_name, parent_module_globals, name, warning=None, error_message=None
    ):  # pylint: disable=super-init-not-called
        self._ll_local_name = local_name
        self._ll_parent_module_globals = parent_module_globals
        self._ll_warning = warning
        self._ll_error_message = error_message

        super().__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        module = None
        try:
            module = importlib.import_module(self.__name__)
        except BaseException as e:
            if self._ll_error_message:
                raise ImportError(self._ll_error_message) from e
            raise ImportError(f"Could not import module {self.__name__}") from e
        self._ll_parent_module_globals[self._ll_local_name] = module

        # Emit a warning if one was specified
        if self._ll_warning:
            logger.warning(self._ll_warning)
            # Make sure to only warn once.
            self._ll_warning = None

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, name):
        module = self._load()
        return getattr(module, name)

    def __setattr__(self, name, value):
        if name.startswith(_LAZY_LOADER_PREFIX):
            super().__setattr__(name, value)
        else:
            module = self._load()
            setattr(module, name, value)
            self.__dict__[name] = value
            try:
                # check if the module has __all__
                if name not in self.__all__ and name != "__all__":
                    self.__all__.append(name)
            except AttributeError:
                pass

    def __delattr__(self, name):
        if name.startswith(_LAZY_LOADER_PREFIX):
            super().__delattr__(name)
        else:
            module = self._load()
            delattr(module, name)
            self.__dict__.pop(name)
            try:
                # check if the module has __all__
                if name in self.__all__:
                    self.__all__.remove(name)
            except AttributeError:
                pass

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)

    def __repr__(self):
        # Carefully to not trigger _load, since repr may be called in very
        # sensitive places.
        return f"<LazyLoader {self.__name__} as {self._ll_local_name}>"

    def __dir__(self):
        module = self._load()
        return dir(module)

    def __reduce__(self):
        return importlib.import_module, (self.__name__,)


class LazyModuleAccessor:
    def __init__(self, module, path):
        self.module = module
        self.parts = path.split(".")

    def __getattr__(self, name):
        mod = self.module
        for part in self.parts:
            mod = getattr(mod, part)
        return getattr(mod, name)

    def __call__(self, *args, **kwargs):
        mod = self.module
        for part in self.parts:
            mod = getattr(mod, part)
        return mod(*args, **kwargs)
