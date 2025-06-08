## Getting started

### Prerequisites

Python 3.10 or later is required.

`collimator` has been developed and tested on Linux (Ubuntu 22+) and macOS.

Native Windows support has not been tested at this time but may work. We
recommend using the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
with an Ubuntu distribution instead.

### Installation steps

```bash
pip install pycollimator
```

### Optional dependencies

Nonlinear MPC blocks require `IPOPT` to be preinstalled.

- On Ubuntu: `sudo apt install coinor-libipopt-dev`.
- On macOS: `brew install ipopt`.

On macOS with Apple Silicon (M series), `cmake` is also required to build and
install `qdldl` and `osqp` dependencies. Install it with `brew install cmake`.

Install all optional dependencies with:

```bash
pip install pycollimator[all]
```

### Tutorials

Read the [Getting Started Tutorial](https://py.collimator.ai/tutorials/01-getting-started/)
for a more complete example.

## Documentation

Head over to [https://py.collimator.ai](https://py.collimator.ai) for
the API reference documentation as well as examples and tutorials.

## Licensed under AGPLv3

This package is released and licensed under the
[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html) license.

Collimator, Inc reserves all rights to release
under a different license under a different license at any time, as well as all
rights to use the code in any way in their own commercial offerings.
