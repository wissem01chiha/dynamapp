# Description
This is an experimental implementation of an Acausal (Physical) modeling framework.

## Goal
The goal is to demonstrate:
1. an acausal system modeling language based on symbolic math tool like Sympy
2. how component equations and interfaces can be represented using the language
3. how the components can be connected in a network, and system equations generated from the components of the network
4. how the system equations can be used to create a LeafSystem that produces the system dynamics
5. how such a LeaftSystem can be ingrated with a greater pycollimator model and simulated

## Comparison to Modelica Compilers
The main difference with this implementation relative to implementations like Modelica, is that this implementation does not start with a text language that needs to be parsed in order to extract component equations to symbolic representations. Rather than starting with a text language, this implementation starts with a symbolic language which uses Sympy Symbol and Equation classes to directly encode the symbolic representation of the component equations and network equations. The end results is the same as Modelica compilers, e.g. a set of differential equations for the system, but this implementation avoids the need for the parser/lexer/etc., the early stages of compliation needed by Modelica compilers.

In summary, Modelica compiler does this:
- text = user/developer writes components/models as .mo files
- ast = parse(text)
- symbolic_equations = interpreter(ast)
- differential_equations = processing(symbolic_equations)

Where as this implementaion does this:
- symbolic_equations = user/developer writes components/models as .py files
- differential_equations = processing(symbolic_equations)

## Integration in pycollimator
The high level idea is that for a mixed model of causal/acausal blocks, the acausal blocks will be "loaded" separately from the causal blocks, to create an AcasualDiagram object. This AcasualDiagram will be consumed by a AcasualCompiler to produce an AcasualSystem, a LeafSystem with some specificcallback preparation. An instance of AcasualSystem will have an input for each causal inport amongst the set of compoenents in the AcasualDiagram, and an output for each causal outport amongst the set of components in the AcasualDiagram.
When starting from model.json, the acausal portion is first captured in an AcausalNetwork objects, and later converted to an AcausalDiagram.

Presently, the pipeline starting from model.json is as follows:
1. model.json -> from_model_json.py:identify_acausal_networks() -> AcausalNetwork
2. AcausalNetwork -> from_model_json.py:build_acausal_phleaf() -> AcausalDiagram -> AcausalCompiler -> AcasualSystem
3. AcasualSystem -> from_model_json.py insert and connect phealf back in the wildcat diagram.


## Development Status
This is under heavy development. Expect the odd `print()` statement, and occassional broken or incomplete feature.

# Usage
The basic use case in pycollimator is as follows:
```
# make acausal diagram
ev = base.EqnEnv()
adiagram = AcausalDiagram()
v1 = elec.VoltageSource(ev, name="v1", V=1.0)
r1 = elec.Resistor(ev, name="r1", R=1.0)
c1 = elec.Capacitor(ev, name="c1", C=1.0)
ref1 = elec.Ground(ev, name="ref1")
adiagram.connect(v1, "p", r1, "n")
adiagram.connect(r1, "p", c1, "p")
adiagram.connect(c1, "n", v1, "n")
adiagram.connect(v1, "n", ref1, "p")

# compile the diagram to an AcausalSystem
acompiler = AcausalCompiler(ev, adiagram)
asystem = ac()
...
builder.add(asystem)
builder.connect(..., asystem.input_ports[0])
builder.connect(asystem.output_ports[0], ...)
...
```

# Testing
- pycollimator testing can be found in `test/models/test_acausal.py`
- mode.json testing can be found in `test/app/Acausal/test_acausal_json.py`