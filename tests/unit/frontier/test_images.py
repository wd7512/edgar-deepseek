"""Offline checks for parent-specific discovery images."""

from functools import partial
from io import BytesIO
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from edgar_study.frontier.images import FrontierImages, ParentFit, prepare_images


def test_same_sample_for_each_parent_and_root_fallback():
    angles = np.tile(np.linspace(0, 2 * np.pi, 24, endpoint=False), (15, 1))
    responses = 3 + np.cos(angles) + np.arange(15)[:, None] / 10

    def model(theta, level):
        return jnp.full_like(theta, level)

    def objective(model, estimator, loss, x, y, **kwargs):
        return None, None, 0.0, jnp.full((len(x), 1), estimator())

    up = {
        "hypothesis_engine": SimpleNamespace(objective=objective),
        "loss_functions": SimpleNamespace(quadratic_loss=None),
        "seed_programs": SimpleNamespace(neuron_model_2_jax=model, parameter_estimator_2=lambda: 2.0),
    }
    parents = tuple(ParentFit(f"p{i}", model, lambda i=i: float(i)) for i in range(1, 5))
    opts = {"seed": 31}
    root = prepare_images(up, angles, responses, **opts)
    batch = prepare_images(up, angles, responses, parents=parents, **opts)
    expected = tuple(np.random.default_rng(31).choice(15, size=10, replace=False))

    assert root.cells == batch.cells == expected
    assert root.count == 20 and batch.count == 80
    assert batch.parent_ids == ("p1", "p2", "p3", "p4")
    assert root.parent_ids == ()
    assert [panel() for panel in root.panels[:2]] == [panel() for panel in batch.panels[20:22]]
    assert batch.panels[0]() != batch.panels[20]()
    for index, kind in enumerate(("Discovery response", "Residual by angle")):
        with Image.open(BytesIO(batch.panels[index]())) as image:
            assert image.format == "PNG"
            assert image.info["Title"] == kind


def test_panels_are_drawn_on_first_use_only_once_and_keep_their_order():
    drawn = []

    def panel(i):
        drawn.append(i)
        return bytes([i])

    batch = FrontierImages((1,), (), (b"ready", partial(panel, 1), partial(panel, 2)))

    assert batch.count == 3 and drawn == []
    assert batch.images == (b"ready", b"\x01", b"\x02")
    assert batch.images == (b"ready", b"\x01", b"\x02") and drawn == [1, 2]


def test_a_batch_is_counted_without_drawing_and_each_panel_draws_the_same_bytes_every_time():
    angles = np.tile(np.linspace(0, 2 * np.pi, 24, endpoint=False), (12, 1))
    responses = 2 + np.sin(angles)

    def model(theta, level):
        return jnp.full_like(theta, level)

    parent = ParentFit("p", model, None, lambda cohort, max_iter: (0.0, 0.0, 1, np.full((10, 1), 2.0)))
    batch = prepare_images({}, angles, responses, seed=3, parents=(parent,))
    again = prepare_images({}, angles, responses, seed=3, parents=(parent,))

    assert batch.count == 20
    assert batch.panels[1]() == again.panels[1]()


def _double_gaussian(theta, b, a1, a2, m, s):
    wrap = jnp.arctan2(jnp.sin(theta - m), jnp.cos(theta - m))
    back = jnp.arctan2(jnp.sin(theta - m - jnp.pi), jnp.cos(theta - m - jnp.pi))
    return b + a1 * jnp.exp(-0.5 * (wrap / s) ** 2) + a2 * jnp.exp(-0.5 * (back / s) ** 2)


def _population(extra):
    from edgar_study.frontier.images import population_residuals

    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, (40, 720)).astype(np.float32)
    params = np.stack(
        [
            rng.uniform(0.2, 1, 40),
            rng.uniform(2, 5, 40),
            rng.uniform(0.5, 1, 40),
            rng.uniform(0, 6, 40),
            np.full(40, 0.4),
        ],
        axis=1,
    ).astype(np.float32)
    curve = np.asarray(jax.vmap(_double_gaussian)(jnp.asarray(angles), *params.T))
    aligned = np.arctan2(np.sin(angles - params[:, 3:4]), np.cos(angles - params[:, 3:4]))
    responses = curve + extra(aligned) * params[:, 1:2]
    return population_residuals(_double_gaussian, params, angles, responses)


def test_a_perfect_fit_has_a_flat_population_residual():
    centres, mean, _ = _population(lambda aligned: 0 * aligned)
    assert len(centres) == 36 and np.max(np.abs(mean)) < 1e-4


def test_a_misfit_shared_by_every_neuron_shows_up_aligned_to_the_preferred_angle():
    centres, mean, _ = _population(lambda aligned: 0.2 * np.cos(aligned))
    np.testing.assert_allclose(mean, 0.2 * np.cos(np.deg2rad(centres)), atol=0.03)


def test_the_population_panel_is_a_png():
    from edgar_study.frontier.images import population_panel

    png = population_panel("double_gaussian", *_population(lambda aligned: 0 * aligned))
    assert Image.open(BytesIO(png)).format == "PNG"
