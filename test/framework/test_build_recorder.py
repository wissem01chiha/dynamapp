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

import numpy as np

from collimator import DiagramBuilder, Parameter, library, simulate
from collimator.framework import build_recorder


def test_build_recorder_simple():
    build_recorder.start()
    builder = DiagramBuilder()
    c = Parameter(name="c", value=np.array([1.0, 2.0]))
    g = Parameter(name="g", value=2.0)
    p = Parameter(name="p", value=3.0)
    Constant_0 = builder.add(library.Constant(value=c))
    builder.add(library.Constant(value=np.array([1.0, 2.0])))
    Gain_0 = builder.add(library.Gain(gain=g * p, name="Gain_0"))
    builder.connect(Constant_0.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build("root", parameters={"c": c, "g": g, "p": p})
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={"Gain_0.out_0": Gain_0.output_ports[0]},
    )

    np.testing.assert_array_equal(results.outputs["Gain_0.out_0"][0], [6.0, 12.0])

    code = build_recorder.generate_code()
    exec(code, globals())

    new_diagram = globals()["root"]
    Gain_0 = globals()["root_Gain_0"]
    new_context = new_diagram.create_context()
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={"Gain_0.out_0": Gain_0.output_ports[0]},
    )

    np.testing.assert_array_equal(new_results.outputs["Gain_0.out_0"][0], [6.0, 12.0])


def test_build_recorder_groups():
    build_recorder.start()
    builder = DiagramBuilder()
    c = Parameter(name="c", value=np.array([1.0, 2.0]))
    g = Parameter(name="g", value=3.0)

    Gain_0 = builder.add(library.Gain(gain=1 + g, name="Gain_0"))

    def _make_group():
        group_builder = DiagramBuilder()
        Gain_0 = group_builder.add(library.Gain(gain=g, name="Gain_0"))
        group_builder.export_input(Gain_0.input_ports[0], "in_0")
        group_builder.export_output(Gain_0.output_ports[0], "out_0")
        group = group_builder.build("group")
        return group

    group_diagram = _make_group()
    builder.add(group_diagram)
    constant = builder.add(library.Constant(value=c))
    builder.connect(constant.output_ports[0], group_diagram.input_ports[0])
    builder.connect(constant.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build("root", parameters={"c": c, "g": g})
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "group.out_0": group_diagram.output_ports[0],
            "Gain_0.out_0": Gain_0.output_ports[0],
        },
    )

    np.testing.assert_array_equal(results.outputs["group.out_0"][0], [3.0, 6.0])
    np.testing.assert_array_equal(results.outputs["Gain_0.out_0"][0], [4.0, 8.0])

    code = build_recorder.generate_code()

    exec(code, globals())

    new_diagram = globals()["root"]
    group = globals()["root_group"]
    gain = globals()["root_Gain_0"]
    new_context = new_diagram.create_context()
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={
            "group.out_0": group.output_ports[0],
            "gain.out_0": gain.output_ports[0],
        },
    )

    np.testing.assert_array_equal(new_results.outputs["group.out_0"][0], [3.0, 6.0])
    np.testing.assert_array_equal(new_results.outputs["gain.out_0"][0], [4.0, 8.0])


def test_build_recorder_submodels():
    # FIXME: reference submodels are generated as groups.
    build_recorder.start()
    builder = DiagramBuilder()
    c = Parameter(name="c", value=np.array([1.0, 2.0]))
    g = Parameter(name="g", value=3.0)

    Gain_0 = builder.add(library.Gain(gain=g, name="Gain_0"))

    def _make_submodel(instance_name, parameters):
        sub_builder = DiagramBuilder()
        Gain_0 = sub_builder.add(library.Gain(gain=parameters["gain"], name="Gain_0"))
        sub_builder.export_input(Gain_0.input_ports[0], "in_0")
        sub_builder.export_output(Gain_0.output_ports[0], "out_0")
        sub = sub_builder.build(instance_name)
        return sub

    ref_id = library.ReferenceSubdiagram.register(
        _make_submodel, parameter_definitions=[Parameter(name="gain", value=0)]
    )

    sub1 = library.ReferenceSubdiagram.create_diagram(
        ref_id, "sub1", instance_parameters={"gain": g + 1}
    )
    sub2 = library.ReferenceSubdiagram.create_diagram(
        ref_id, "sub2", instance_parameters={"gain": g + 2}
    )

    builder.add(sub1)
    builder.add(sub2)
    constant = builder.add(library.Constant(value=c))

    builder.connect(constant.output_ports[0], sub1.input_ports[0])
    builder.connect(constant.output_ports[0], sub2.input_ports[0])
    builder.connect(constant.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build("root", parameters={"c": c, "g": g})
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "sub1.out_0": sub1.output_ports[0],
            "sub2.out_0": sub2.output_ports[0],
            "Gain_0.out_0": Gain_0.output_ports[0],
        },
    )

    np.testing.assert_array_equal(results.outputs["sub1.out_0"][0], [4.0, 8.0])
    np.testing.assert_array_equal(results.outputs["sub2.out_0"][0], [5.0, 10.0])
    np.testing.assert_array_equal(results.outputs["Gain_0.out_0"][0], [3.0, 6.0])

    code = build_recorder.generate_code()
    exec(code, globals())

    new_diagram = globals()["root"]
    sub1 = globals()["root_sub1"]
    sub2 = globals()["root_sub2"]
    gain = globals()["root_Gain_0"]
    new_context = new_diagram.create_context()
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={
            "sub1.out_0": sub1.output_ports[0],
            "sub2.out_0": sub2.output_ports[0],
            "gain.out_0": gain.output_ports[0],
        },
    )

    np.testing.assert_array_equal(new_results.outputs["sub1.out_0"][0], [4.0, 8.0])
    np.testing.assert_array_equal(new_results.outputs["sub2.out_0"][0], [5.0, 10.0])
    np.testing.assert_array_equal(new_results.outputs["gain.out_0"][0], [3.0, 6.0])


def test_build_recorder_python_script_block():
    build_recorder.start()
    builder = DiagramBuilder()

    PythonScriptBlock = builder.add(
        library.CustomPythonBlock(
            dt=0.1,
            user_statements="import numpy as np\nout_0 = in_0 * 2",
            init_script="out_0 = 0.0",
            inputs=["in_0"],
            outputs=["out_0"],
            name="PythonScriptBlock",
        )
    )
    Constant = builder.add(library.Constant(value=1))
    builder.connect(Constant.output_ports[0], PythonScriptBlock.input_ports[0])
    diagram = builder.build("root")
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={"PythonScriptBlock.out_0": PythonScriptBlock.output_ports[0]},
    )

    assert results.outputs["PythonScriptBlock.out_0"][-1] == 2.0

    code = build_recorder.generate_code()
    print(code)
    exec(code, globals())

    new_diagram = globals()["root"]
    PythonScriptBlock = globals()["root_PythonScriptBlock"]
    new_context = new_diagram.create_context()
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={"PythonScriptBlock.out_0": PythonScriptBlock.output_ports[0]},
    )

    assert new_results.outputs["PythonScriptBlock.out_0"][-1] == 2.0
