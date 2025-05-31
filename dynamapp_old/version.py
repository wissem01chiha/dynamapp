_MAJOR = "1"
_MINOR = "0"
_PATCH = "0"
_EXTRA = "" 

__version__ = f"{_MAJOR}.{_MINOR}.{_PATCH}{_EXTRA}"

__version_short__ = f"{_MAJOR}.{_MINOR}"

if __name__ == "__main__":
    print(f"version: {__version__}")