Contributing
======  

Hi, the work is in progress, and any contributions to the project are welcome. See the following guidelines for areas where help is needed:

- [ ] Complete the development of the [Trajectory](pyDynaMapp/trajectory/) Engine generation.
- [ ] Fix the friction [Lugre](pyDynaMapp/friction/../viscoelastic/lugre.py) model base class.
- [ ] Add more friction and/or rigid body contact models in the viscoelastic engine.
- [ ] Add exception handlers for zero division and/or NaN/Inf value assertions in friction scripts.
- [ ] Complete the N4SID integration method.
- [ ] Implement a reduced mechanical model.
- [ ] Adjust for GPU support (CUDA).
- [ ] Fix the differentiation problem (finite difference method is error-prone). Consider using automatic differentiation with Autodiff or other tools.
- [ ] Change function and class naming to align with Python coding standards.
- [ ] Fix the inertial matrix symmetry issue (add a custom verification function or similar).
- [ ] Add more test scripts in the `/test` folder to detect software bugs.

