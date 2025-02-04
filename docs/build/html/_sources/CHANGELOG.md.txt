# Changelog

## Release 1.0.0

- Refactor to use automatic differentiation through JAX, removing the dependency on the numpy package.
- Remove dependency on pnnchio library. The model will now be defined by explicit dh-parameters, inertial matrices, simple damping, and a generalized friction unified model.
- Restructure into one Python package with multiple modules inside.
- Remove all complex and non-linear friction models (such as Lugure, Maxwell, and others requiring solving partial differential equations at each step). Adopt a simpler general friction model where the force related to joint position, velocity, and acceleration is described by the relation for a single joint \(i\):

  
  $$f_i = \alpha_{1i} q_i + \alpha_{2i} q_i^2 + \ldots + \alpha_{ni} q_i^n + \beta_{1i} v_i + \beta_{2i} v_i^2 + \ldots + \beta_{ki} v_i^k + \gamma_{1i} a_i + \gamma_{2i} a_i^2 + \ldots + \gamma_{pi} a_i^p $$
 

  Here, the values of $n$, $k$ , and $p$ are dynamically chosen and are generally not equal. Non-linear transformations on $q_i$, $v_i$, and $a_i$ are allowed, while not incorporating any specific parameters (e.g., $ \beta_{1i} \cos(v_i) $).

## Release 0.1.0

- First experimental package.
