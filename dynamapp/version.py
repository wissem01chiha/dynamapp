"""
Version Information
-------------------
This file defines the version informations of the package.
"""
_MAJOR = "0"
_MINOR = "1"
_PATCH = "0"
_EXTRA = "" 

__version__ = f"{_MAJOR}.{_MINOR}.{_PATCH}{_EXTRA}"

__version_short__ = f"{_MAJOR}.{_MINOR}"

if __name__ == "__main__":
    print(f"version: {__version__}")