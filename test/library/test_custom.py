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

import pytest

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import collimator
from collimator import library
from collimator.library.custom import PythonScriptError
from collimator.backend import numpy_api

# from collimator import logging
# logging.set_file_handler("test.log")

sin_init_code_np = "import numpy as np"
sin_step_code_np = "out_0 = np.sin(10 * in_0)"

sin_init_code_jnp = "import jax.numpy as jnp"
sin_step_code_jnp = "out_0 = jnp.sin(10 * in_0)"

relay_init_code_implicit = """
import jax.numpy as jnp
state = 0.0
"""

relay_init_code_explicit = """
import jax.numpy as jnp
state = 0.0
out_0 = state
"""

relay_step_code = """
state = jnp.where(in_0 > 0.5, 1.0, state)
state = jnp.where(in_0 < -0.5, 0.0, state)
out_0 = state
"""

# Legacy code for PythonScript relay definition
untraceable_step_code = """
if in_0 > 0.5:
    state = 1.0
elif in_0 < -0.5:
    state = 0.0
out_0 = state
"""

xfail_init_code = """
import jax.numpy as jnp
x = jnp.zeros(4)
out_0 = 0.0
"""

xfail_step_code = """
x[0] = 1.0
out_0 = x[0]
"""

counter_init_code_implicit = "count = 0.0"

counter_init_code_explicit = """
count = 0.0
out_0 = count
"""

counter_step_code = """
count = count + 1
out_0 = count
"""


def _test_relay(diagram, tf, dt, backend="jax"):
    numpy_api.set_backend(backend)

    context = diagram.create_context()

    recorded_signals = {
        "sin.out_0": diagram["sin_psb"].output_ports[0],
        "relay.out_0": diagram["relay_psb"].output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
    )

    sin_sim = results.outputs["sin.out_0"]
    sin_ex = jnp.sin(10 * results.time)
    assert jnp.allclose(sin_sim, sin_ex)

    relay_sim = results.outputs["relay.out_0"]
    relay_ex = np.zeros(np.shape(relay_sim))
    for idx in range(sin_ex.size):
        if sin_ex[idx] > 0.5:
            relay_ex[idx] = 1.0
        elif sin_ex[idx] < -0.5:
            relay_ex[idx] = 0.0
        elif idx > 0:
            relay_ex[idx] = relay_ex[idx - 1]
    print("time")
    time = results.time
    print(f"{time=}")
    print("sin block")
    print(f"{sin_ex=}")
    print(f"{sin_sim=}")

    print("relay block")
    print(f"{relay_ex=}")
    print(f"{relay_sim=}")

    assert jnp.allclose(relay_sim, relay_ex)
    return results.time, results.outputs


def _make_jax_relay_diagram(dt, reverse_psb_order=False, implicit=False):
    ramp = library.Ramp(start_time=0.0, name="ramp")

    sin_psb = library.CustomJaxBlock(
        name="sin_psb",
        init_script=sin_init_code_jnp,
        user_statements=sin_step_code_jnp,
        inputs=["in_0"],
        outputs=["out_0"],
        time_mode="agnostic",
    )

    if implicit:
        # Test the case where the output value is not explicitly initialized
        # in the init_script.  This should produce a PythonScriptError
        relay_init_code = relay_init_code_implicit
    else:
        # Test the case where the output value is explicitly initialized
        # in the init_script. This should work properly.
        relay_init_code = relay_init_code_explicit

    relay_psb = library.CustomJaxBlock(
        name="relay_psb",
        dt=dt,
        time_mode="discrete",
        init_script=relay_init_code,
        user_statements=relay_step_code,
        inputs=["in_0"],
        outputs=["out_0"],
    )

    builder = collimator.DiagramBuilder()
    builder.add(ramp)

    if reverse_psb_order:
        builder.add(relay_psb)
        builder.add(sin_psb)

    else:
        builder.add(sin_psb)
        builder.add(relay_psb)

    builder.connect(ramp.output_ports[0], sin_psb.input_ports[0])
    builder.connect(sin_psb.output_ports[0], relay_psb.input_ports[0])
    return builder.build()


@pytest.mark.minimal
def test_custom_relay_traceable():
    dt = 0.1

    diagram = _make_jax_relay_diagram(dt=dt, reverse_psb_order=False)
    t1, sol1 = _test_relay(diagram, tf=2.0, dt=dt, backend="jax")

    # Test adding the blocks in the reverse order
    # see https://collimator.atlassian.net/browse/WC-66 for a description
    # of the bug this tests

    diagram = _make_jax_relay_diagram(dt=dt, reverse_psb_order=True)
    t2, sol2 = _test_relay(diagram, tf=2.0, dt=dt, backend="jax")

    assert jnp.allclose(t1, t2)
    assert jnp.allclose(sol1["sin.out_0"], sol2["sin.out_0"])
    assert jnp.allclose(sol1["relay.out_0"], sol2["relay.out_0"])

    # Check that the implicit initialization fails
    with pytest.raises(PythonScriptError):
        diagram = _make_jax_relay_diagram(dt=dt, implicit=True)


@pytest.mark.minimal
@pytest.mark.parametrize("use_jax", [True, False])
def test_custom_agnostic(use_jax):
    # Connecting a discrete clock to a discrete block will lead to a "data
    # flow" delay of one step in the output of the block. This will not
    # be the case for a block in "agnostic" time mode.
    dt = 0.1

    backend = "jax" if use_jax else "numpy"
    numpy_api.set_backend(backend)
    CustomBlock = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    agnostic_psb = CustomBlock(
        name="agnostic_psb",
        init_script="x = 2.0",
        user_statements="out_0 = x * in_0",
        time_mode="agnostic",
        inputs=["in_0"],
        outputs=["out_0"],
    )

    builder = collimator.DiagramBuilder()
    builder.add(agnostic_psb)
    source = builder.add(library.DiscreteClock(dt=dt))
    builder.connect(source.output_ports[0], agnostic_psb.input_ports[0])

    diagram = builder.build()
    assert not diagram.has_ode_side_effects
    context = diagram.create_context()

    recorded_signals = {
        "psb.out_0": agnostic_psb.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 2.0),
        recorded_signals=recorded_signals,
    )
    assert jnp.allclose(results.outputs["psb.out_0"], 2.0 * results.time)


@pytest.mark.minimal
def test_custom_fail_compile():
    dt = 0.1
    block = library.CustomJaxBlock(
        name="xfail",
        dt=dt,
        time_mode="discrete",
        init_script=xfail_init_code,
        user_statements=xfail_step_code,
        inputs=[],
        outputs=["out_0"],
    )

    with pytest.raises(PythonScriptError):
        block.create_context()


def _make_python_relay_diagram(dt, reverse_psb_order=False, implicit=False):
    ramp = library.Ramp(start_time=0.0, name="ramp")

    sin_psb = library.CustomPythonBlock(
        name="sin_psb",
        dt=dt,
        init_script=sin_init_code_np,
        user_statements=sin_step_code_np,
        inputs=["in_0"],
        outputs=["out_0"],
        time_mode="agnostic",
    )

    if implicit:
        # Test the case where the output value is not explicitly initialized
        # in the init_script.  This should produce a PythonScriptError
        relay_init_code = relay_init_code_implicit
    else:
        # Test the case where the output value is explicitly initialized
        # in the init_script. This should work properly.
        relay_init_code = relay_init_code_explicit

    relay_psb = library.CustomPythonBlock(
        name="relay_psb",
        dt=dt,
        time_mode="discrete",
        init_script=relay_init_code,
        user_statements=untraceable_step_code,
        inputs=["in_0"],
        outputs=["out_0"],
    )

    builder = collimator.DiagramBuilder()

    builder.add(ramp)

    if reverse_psb_order:
        builder.add(relay_psb)
        builder.add(sin_psb)

    else:
        builder.add(sin_psb)
        builder.add(relay_psb)

    builder.connect(ramp.output_ports[0], sin_psb.input_ports[0])
    builder.connect(sin_psb.output_ports[0], relay_psb.input_ports[0])

    return builder.build()


# Repeat using untraceable Python (standard control flow)
@pytest.mark.slow
def test_custom_relay_untraceable():
    dt = 0.1

    diagram = _make_python_relay_diagram(dt=dt, reverse_psb_order=False)
    assert not diagram.has_ode_side_effects
    t1, sol1 = _test_relay(diagram, tf=2.0, dt=dt, backend="numpy")

    # Test adding the blocks in the reverse order
    # see https://collimator.atlassian.net/browse/WC-66 for a description
    # of the bug this tests
    diagram = _make_python_relay_diagram(dt=dt, reverse_psb_order=True)
    assert not diagram.has_ode_side_effects
    t2, sol2 = _test_relay(diagram, tf=2.0, dt=dt, backend="numpy")

    assert jnp.allclose(t1, t2)
    assert jnp.allclose(sol1["sin.out_0"], sol2["sin.out_0"])
    assert jnp.allclose(sol1["relay.out_0"], sol2["relay.out_0"])

    # Check that the implicit initialization fails
    with pytest.raises(PythonScriptError):
        diagram = _make_python_relay_diagram(dt=dt, implicit=True)


def _make_jax_counter_diagram(dt, implicit=False):
    if implicit:
        # Test the case where the output value is not explicitly initialized
        # in the init_script.  This should produce a PythonScriptError
        counter_init_code = counter_init_code_implicit
    else:
        # Test the case where the output value is explicitly initialized
        # in the init_script. This should work properly.
        counter_init_code = counter_init_code_explicit

    counter_psb = library.CustomJaxBlock(
        name="counter_psb",
        dt=dt,
        init_script=counter_init_code,
        user_statements=counter_step_code,
        outputs=["out_0"],
        time_mode="discrete",
    )

    gain_by_2 = library.CustomJaxBlock(
        name="gain_by_2",
        dt=dt,
        user_statements="out_0 = in_0 * 2",
        inputs=["in_0"],
        outputs=["out_0"],
        time_mode="agnostic",
    )

    builder = collimator.DiagramBuilder()
    builder.add(counter_psb)
    builder.add(gain_by_2)

    builder.connect(counter_psb.output_ports[0], gain_by_2.input_ports[0])
    return builder.build()


def _test_counter(diagram, tf, show_plot=False):
    context = diagram.create_context()

    recorded_signals = {
        "counter": diagram["counter_psb"].output_ports[0],
        "gain_by_2": diagram["gain_by_2"].output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
    )

    time = results.time
    counter_res = results.outputs["counter"]
    gain_res = results.outputs["gain_by_2"]

    print(f"time=\n{time}")
    print(f"counter_res=\n{counter_res}")
    print(f"counter_sol=\n{np.arange(len(counter_res))}")
    print(f"gain_res=\n{gain_res}")

    if show_plot:
        fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))

        ax1.plot(time, counter_res, label="counter", marker="x")
        ax1.plot(time, gain_res, label="gain_by_2", marker="o")
        ax1.grid(True)
        ax1.legend()

        plt.show()

    assert np.allclose(counter_res, np.arange(len(counter_res)))
    assert np.allclose(gain_res, 2 * counter_res)


@pytest.mark.minimal
def test_custom_counter(show_plot=False):
    numpy_api.set_backend("jax")

    dt = 0.1

    diagram = _make_jax_counter_diagram(dt)
    _test_counter(diagram, tf=2.0, show_plot=show_plot)

    # Check that the implicit initialization fails
    with pytest.raises(PythonScriptError):
        diagram = _make_jax_counter_diagram(dt, implicit=True)


user_statements = """
out_0 = in_0 * 3.0
out_1 = in_1 * 6.0
"""

init_script = """
import jax.numpy as jnp
out_0 = 0.0
out_1 = jnp.zeros(2)
"""


@pytest.mark.minimal
def test_custom_output_shape(show_plot=False):
    numpy_api.set_backend("jax")
    dt = 0.1

    sclr = library.Constant(value=1.0)
    vec = library.Constant(value=jnp.ones(2))

    my_blk = library.CustomJaxBlock(
        name="my_blk",
        dt=dt,
        init_script=init_script,
        user_statements=user_statements,
        inputs=["in_0", "in_1"],
        outputs=["out_0", "out_1"],
    )

    builder = collimator.DiagramBuilder()
    builder.add(sclr)
    builder.add(vec)
    builder.add(my_blk)

    builder.connect(sclr.output_ports[0], my_blk.input_ports[0])
    builder.connect(vec.output_ports[0], my_blk.input_ports[1])
    diagram = builder.build()

    context = diagram.create_context()

    recorded_signals = {
        "vec": vec.output_ports[0],
        "my_blk.out_0": my_blk.output_ports[0],
        "my_blk.out_1": my_blk.output_ports[1],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 0.3),
        recorded_signals=recorded_signals,
    )

    my_blk_out_1 = results.outputs["my_blk.out_1"]
    vec_out = results.outputs["vec"]
    out_1_sol = vec_out * 6.0

    print(f"vec_out={vec_out}\n")
    print(f"my_blk_out_1={my_blk_out_1}\n")
    print(f"out_1_sol={out_1_sol}\n")

    assert np.allclose(my_blk_out_1[1:, :], out_1_sol[1:, :])


@pytest.mark.parametrize("use_jax", [True, False])
def test_wc159(use_jax):
    # Tests a bug where initializing the values in discrete mode fails if the outputs
    # are not scalars.  Also tests that outputs written as lists will get properly
    # type-converted.
    dt = 0.1
    PythonScript = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    out_0_init = [3, 4]
    out_0_step = [5, 6]
    system = PythonScript(
        name="my_blk",
        dt=dt,
        time_mode="discrete",
        user_statements=f"out_0 = {out_0_step}",
        init_script=f"out_0 = {out_0_init}",
        inputs=[],
        outputs=["out_0"],
    )

    context = system.create_context()
    recorded_signals = {
        "system.out_0": system.output_ports[0],
    }
    results = collimator.simulate(
        system,
        context,
        (0.0, 0.3),
        recorded_signals=recorded_signals,
    )

    out_0 = results.outputs["system.out_0"]

    y_init = np.array(out_0_init)
    y_step = np.array(out_0_step)

    assert np.allclose(out_0[0], y_init)
    assert np.allclose(out_0[1:], y_step)
    assert out_0.dtype == int


def test_wc230():
    # Test a bug where PyTree continuous state did not get properly unraveled
    # when passing `result_shape_dtype` to the jax pure callback. To trigger
    # this, need to use an agnostic-mode PythonScript block fed to an Integrator,
    # with another continuous state elsewhere in the system.
    # See https://collimator.atlassian.net/browse/WC-230 for details.
    builder = collimator.DiagramBuilder()

    Clock_0 = builder.add(library.Clock(name="Clock_0"))
    PythonScript_0 = builder.add(
        library.CustomPythonBlock(
            name="PythonScript_0",
            init_script="",
            user_statements="out_0 = in_0",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
    )
    Integrator_0 = builder.add(library.Integrator(0.0, name="Integrator_0"))
    Integrator_1 = builder.add(library.Integrator(0.0, name="Integrator_1"))

    builder.connect(Clock_0.output_ports[0], PythonScript_0.input_ports[0])
    builder.connect(PythonScript_0.output_ports[0], Integrator_0.input_ports[0])
    builder.connect(Clock_0.output_ports[0], Integrator_1.input_ports[0])

    system = builder.build()
    context = system.create_context()

    recorded_signals = {
        "Integrator_0.out_0": Integrator_0.output_ports[0],
        "Integrator_1.out_0": Integrator_1.output_ports[0],
        "PythonScript_0.out_0": PythonScript_0.output_ports[0],
    }

    options = collimator.SimulatorOptions(enable_tracing=True)

    tf = 10.0
    results = collimator.simulate(
        system,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
        options=options,
    )

    assert results.time[-1] == tf
    assert jnp.allclose(results.outputs["PythonScript_0.out_0"], results.time)
    assert jnp.allclose(results.outputs["Integrator_0.out_0"], 0.5 * results.time**2)
    assert jnp.allclose(results.outputs["Integrator_1.out_0"], 0.5 * results.time**2)


def test_wc235():
    # Test a bug where the type inference calls were not sorted according to callback
    # ordering.  See https://collimator.atlassian.net/browse/WC-235

    builder = collimator.DiagramBuilder()

    # First add the downstream block (Agnostic mode)
    psb_0 = builder.add(
        library.CustomPythonBlock(
            name="psb_0",
            init_script="",
            user_statements="out_0 = in_0 * 2",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
    )

    # Then add the upstream block (Discrete mode)
    psb_1 = builder.add(
        library.CustomPythonBlock(
            name="psb_1",
            dt=0.1,
            init_script="out_0 = 0.0",
            user_statements="out_0 = out_0 + 1",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
    )

    builder.connect(psb_1.output_ports[0], psb_0.input_ports[0])

    system = builder.build()

    # Previously this raised an error because the upstream discrete block did not
    # do type inference before the downstream agnostic block.
    system.create_context()


@pytest.mark.parametrize("use_jax", [True, False])
def test_custom_reuse_outport_state(use_jax):
    # Tests initializing an output port value and then reusing the output
    # port name in the step code.
    dt = 0.1
    backend = "jax" if use_jax else "numpy"
    numpy_api.set_backend(backend)

    PythonScript = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    system = PythonScript(
        name="my_blk",
        dt=dt,
        time_mode="discrete",
        user_statements="out_0 = out_0 + 1",
        init_script="out_0 = 0",
        inputs=[],
        outputs=["out_0"],
    )

    assert not system.has_ode_side_effects

    context = system.create_context()
    recorded_signals = {
        "system.out_0": system.output_ports[0],
    }
    results = collimator.simulate(
        system,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )

    out_0 = results.outputs["system.out_0"]
    assert np.allclose(out_0, np.arange(101))


function_def_init = """
def my_func(x):
    return x * 2
"""

function_def_step = "out_0 = my_func(in_0)"


@pytest.mark.parametrize("use_jax", [True, False])
def test_custom_function_def(use_jax):
    PythonScript = library.CustomJaxBlock if use_jax else library.CustomPythonBlock
    builder = collimator.DiagramBuilder()
    clock = library.Clock(name="clock")
    psb = PythonScript(
        name="my_blk",
        time_mode="agnostic",
        init_script=function_def_init,
        user_statements=function_def_step,
        inputs=["in_0"],
        outputs=["out_0"],
    )

    builder.add(clock, psb)
    builder.connect(clock.output_ports[0], psb.input_ports[0])

    system = builder.build()
    context = system.create_context()

    t = 1.5
    context = context.with_time(t)

    psb_out = psb.output_ports[0].eval(context)
    assert psb_out == 2 * t


van_der_pol_init = """
import numpy as np
mu = 1.0
"""

van_der_pol_step = """
x, y = in_0
x_dot = y
y_dot = mu * (1 - x**2) * y - x
out_0 = np.array([x_dot, y_dot])
"""


@pytest.fixture
def van_der_pol():
    builder = collimator.DiagramBuilder()

    rhs_fun = library.CustomPythonBlock(
        name="rhs",
        init_script=van_der_pol_init,
        user_statements=van_der_pol_step,
        inputs=["in_0"],
        outputs=["out_0"],
        time_mode="agnostic",
    )

    integrator = library.Integrator(
        name="integrator",
        initial_state=[0.0, 1.0],
    )

    builder.add(rhs_fun, integrator)
    builder.connect(rhs_fun.output_ports[0], integrator.input_ports[0])
    builder.connect(integrator.output_ports[0], rhs_fun.input_ports[0])

    builder.export_output(integrator.output_ports[0])

    return builder.build(name="van_der_pol")


class TestFeedthroughSideEffects:
    def test_scipy_fallback(self, van_der_pol):
        # See WC-209 bug report
        # Test that an untraceable feedthrough block can be used as the RHS of an ODE
        integrator = van_der_pol["integrator"]
        van_der_pol.create_context()
        assert van_der_pol.has_ode_side_effects

        context = van_der_pol.create_context()

        recorded_signals = {
            "integrator.out_0": integrator.output_ports[0],
        }
        tf = 10.0
        result = collimator.simulate(
            van_der_pol,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        # Check that the sim made it to the final time
        assert result.time[-1] == tf

    def test_with_zc(self, van_der_pol):
        # If we have both a zero-crossing and ODE side effects, the
        # solver should still work by falling back to untraced execution.

        builder = collimator.DiagramBuilder()
        builder.add(van_der_pol)

        demux = builder.add(library.Demultiplexer(2, name="demux"))

        # Add something with a zero-crossing event
        relay = builder.add(
            library.Relay(
                on_threshold=0.5,
                off_threshold=-0.5,
                on_value=1.0,
                off_value=0.0,
                initial_state=0.0,
                name="relay",
            )
        )

        builder.connect(van_der_pol.output_ports[0], demux.input_ports[0])
        builder.connect(demux.output_ports[0], relay.input_ports[0])

        system = builder.build()
        context = system.create_context()

        assert system.has_ode_side_effects
        assert system.zero_crossing_events.has_events

        recorded_signals = {
            "integrator.out_0": van_der_pol.output_ports[0],
        }
        tf = 10.0
        result = collimator.simulate(
            system,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        assert result.time[-1] == tf


if __name__ == "__main__":
    # test_custom_counter(show_plot=True)
    # test_custom_relay_traceable()
    test_custom_output_shape()
