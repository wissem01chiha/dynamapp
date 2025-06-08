# Installation

## Prerequisites

Python 3.10 or later is required.

`collimator` has been developed and tested on Linux (Ubuntu 22+) and macOS.

Native Windows support has not been tested at this time but may work. We
recommend using the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
with an Ubuntu distribution instead.

## Installation steps

It is highly recommended to use a virtual environment to install `pycollimator`.

```bash
pip install pycollimator
```

- On Windows: `set JAX_ENABLE_X64=True`.

## Optional dependencies

More advanced features require additional dependencies that can be installed with:

```bash
pip install pycollimator[safe]
```

This will not include support for NMPC blocks.

### Nonlinear MPC

Nonlinear MPC blocks require `IPOPT` to be preinstalled.

- On Ubuntu: `sudo apt install coinor-libipopt-dev`.
- On macOS: `brew install ipopt` and `brew install cmake`.

Install all optional dependencies with `pip install pycollimator[all]` or
just the NMPC dependencies with `pip install pycollimator[nmpc]`.

<details>
<summary>Licensed under AGPLv3</summary>
This `pycollimator` package is released and licensed under the
<a href="https://www.gnu.org/licenses/agpl-3.0.en.html">AGPLv3</a> license.

<br />

Collimator, Inc reserves all rights to release under a different license
at any time, as well as all rights to use the code in any
way in their own commercial offerings.

Contact us to obtain a commercial license. Head over to
<a href="https://www.collimator.ai/contact-us">collimator.ai/contact-us</a>
for contact information.

</details>
