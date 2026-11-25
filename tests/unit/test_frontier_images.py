"""Offline checks for parent-specific discovery images."""

from io import BytesIO
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from PIL import Image

from edgar_study.frontier_images import ParentFit, prepare_images


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
    assert len(root.images) == 20 and len(batch.images) == 80
    assert batch.parent_ids == ("p1", "p2", "p3", "p4")
    assert root.parent_ids == ()
    assert root.images == batch.images[20:40]
    assert batch.images[:20] != batch.images[20:40]
    for index, kind in enumerate(("Discovery response", "Residual by angle")):
        with Image.open(BytesIO(batch.images[index])) as image:
            assert image.format == "PNG"
            assert image.info["Title"] == kind
