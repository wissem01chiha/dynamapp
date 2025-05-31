# MatlabProjectTemplate documentation

This directory contains documentation and tools for the MatlabProjectTemplate template repo itself. (This is the tool that was used to generate the skeleton for this project.)

You can ignore this stuff if you want, or even delete it: it is not required for the project to function properly. But you probably want to read the stuff in here, and maybe keep it around. It contains useful information!

## Publishing to MathWorks File Exchange

MatlabProjectTemplate _strongly_ recommends that you publish your project to File Exchange by linking your File Exchange post to your GitHub repo, and using their new Release-driven linkage. (This is what File Exchange themselves recommend, too.)

## Logging

MatlabProjectTemplate sets up support for logging by defining a standard `mypackage.logger` package, which contains code to do logging in a manner compatible with [SLF4M](https://github.com/janklab/SLF4M) and SLF4J, without actually taking a dependency on the SLF4M library. We do it this way because managing inter-library dependencies in Matlab is really hard, due to its lack of a package manager or virtualenvs. Doing it this way means your code will be compatible with any other Matlab code that uses SLF4M or does its own logging this way, and you won't have name collisions.

## About MatlabProjectTemplate

MatlabProjectTemplate was written by [Andrew Janke](https://apjanke.net). The project home page is <https://github.com/janklab/MatlabProjectTemplate>. You can get support, or even contribute to the project, there.

MatlabProjectTemplate is part of the [Janklab](https://github.com/janklab) suite of libraries for Matlab.
