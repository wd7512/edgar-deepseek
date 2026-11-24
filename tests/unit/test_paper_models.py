import jax.numpy as jnp
import numpy as np
import pytest

from edgar_study import paper_models

THETA = jnp.linspace(0, 2 * jnp.pi, 73)


def double_gaussian(theta, pref, baseline, a1, a2, width):
    def dist(centre):
        return jnp.abs(jnp.arctan2(jnp.sin(theta - centre), jnp.cos(theta - centre)))

    return (
        baseline
        + a1 * jnp.exp(-0.5 * (dist(pref) / width) ** 2)
        + a2 * jnp.exp(-0.5 * (dist(pref + jnp.pi) / width) ** 2)
    )


def test_every_ladder_model_reduces_to_the_double_gaussian():
    expected = double_gaussian(THETA, 1.0, 0.2, 3.0, 1.5, 0.4)
    full = {
        "theta_pref": 1.0,
        "baseline": 0.2,
        "amplitude_1": 3.0,
        "amplitude_2": 1.5,
        "tuning_width_1_left": 0.4,
        "tuning_width_1_right": 0.4,
        "tuning_width_2_left": 0.4,
        "tuning_width_2_right": 0.4,
        "tuning_width_1": 0.4,
        "tuning_width_2": 0.4,
        "exponent_1": 2.0,
        "exponent_2": 2.0,
        "angle_offset_2": 0.0,
    }
    for name, (model, _) in paper_models.LADDER.items():
        names = paper_models.parameter_names(model)
        assert np.allclose(model(THETA, *[full[n] for n in names]), expected, atol=1e-5), name


def test_ladder_sizes_match_the_paper():
    sizes = {name: len(paper_models.parameter_names(m)) for name, (m, _) in paper_models.LADDER.items()}

    assert sizes == {"Gaussian+p": 8, "Gaussian+p+asym": 10, "AI model": 11}


@pytest.mark.parametrize("name", list(paper_models.LADDER))
def test_estimators_return_one_value_per_parameter_near_the_peak(name):
    model, estimator = paper_models.LADDER[name]
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 600)
    counts = np.asarray(double_gaussian(theta, 2.0, 0.5, 4.0, 2.0, 0.35)) + 0.1 * rng.standard_normal(600)

    params = estimator(theta, counts)

    assert params.shape == (len(paper_models.parameter_names(model)),)
    assert abs(params[0] - 2.0) < 0.15


def test_the_wave_model_is_the_ai_model_plus_one_sinusoid():
    params = [1.0, 0.2, 3.0, 1.5, 0.3, 0.5, 0.4, 0.6, 1.2, 1.8, 0.1]

    plain = paper_models.ai_model(THETA, *params)
    waved = paper_models.ai_wave_model(THETA, *params, 0.4, -0.3)

    assert np.allclose(paper_models.ai_wave_model(THETA, *params, 0.0, 0.0), plain, atol=1e-6)
    assert np.allclose(waved - plain, 0.4 * jnp.cos(THETA) - 0.3 * jnp.sin(THETA), atol=1e-5)
    assert len(paper_models.parameter_names(paper_models.ai_wave_model)) == 13


def test_the_wave_estimator_starts_without_a_wave():
    theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    counts = np.asarray(double_gaussian(theta, 2.0, 0.5, 4.0, 2.0, 0.35))

    params = paper_models.ai_wave_estimator(theta, counts)

    assert np.allclose(params[:11], paper_models.ai_estimator(theta, counts))
    assert np.allclose(params[11:], 0)
