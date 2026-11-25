"""Offline checks for parent-specific discovery images."""

from functools import partial
from io import BytesIO
from types import SimpleNamespace

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
