import numpy as np

from edgar_study.exponents import clip_bounds, exponent_pair, fitted_exponents


def test_exponent_pair_finds_each_runs_naming():
    assert exponent_pair(["a", "shape_power_1", "shape_power_2"]) == ("shape_power_1", "shape_power_2")
    assert exponent_pair(["baseline", "power_1", "w", "power_2"]) == ("power_1", "power_2")
    assert exponent_pair(["exponent_1", "exponent_2"]) == ("exponent_1", "exponent_2")
    assert exponent_pair(["amplitude_1", "amplitude_2"]) is None


def test_clip_bounds_reads_the_programs_own_clip():
    code = "power_1 = float(np.clip(power_1, 0.5, 4.0))\nshape_power_2 = jnp.clip(shape_power_2, 1.0, 3.2)"
    assert clip_bounds(code, "power_1") == (0.5, 4.0)
    assert clip_bounds(code, "shape_power_2") == (1.0, 3.2)
    assert clip_bounds(code, "power_2") == (-np.inf, np.inf)
    assert clip_bounds("power_2 = lax.clamp(0.5, power_2, 4.0)", "power_2") == (0.5, 4.0)


def test_fitted_exponents_clips_to_the_programs_bounds():
    names = ["baseline", "power_1", "power_2"]
    params = np.array([[0.0, 0.2, 5.0], [0.0, 2.0, 1.5]])
    code = "power_1 = np.clip(power_1, 0.5, 4.0)\npower_2 = np.clip(power_2, 0.5, 4.0)"
    first, second = fitted_exponents(names, params, code)
    assert first.tolist() == [0.5, 2.0]
    assert second.tolist() == [4.0, 1.5]
    assert fitted_exponents(["a"], params[:, :1], code) is None
