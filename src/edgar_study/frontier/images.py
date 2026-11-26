"""Prepare score-free response and residual images for a frontier prompt."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from io import BytesIO

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.figure import Figure

from edgar_study.datasets import TRIAL_SEED
from edgar_study.go_client import GoClient
from edgar_study.upstream import score_program

NEURONS = 10
IMAGES_PER_NEURON = 2


@dataclass(frozen=True)
class ParentFit:
    node_id: str
    model: Callable
    estimator: Callable
    solve: Callable | None = None


@dataclass(frozen=True)
class FrontierImages:
    """The sampled neurons and their image panels, each either PNG bytes or a call that draws them.

    Panels are drawn on first use of `images`, so a batch that is never sent costs no drawing.
    """

    cells: tuple[int, ...]
    parent_ids: tuple[str, ...]
    panels: tuple[bytes | Callable[[], bytes], ...]

    @property
    def count(self) -> int:
        return len(self.panels)

    @cached_property
    def images(self) -> tuple[bytes, ...]:
        return tuple(panel if isinstance(panel, bytes) else panel() for panel in self.panels)

    async def submit(
        self,
        client: GoClient,
        prompt: str,
        *,
        effort: str | None = None,
        temperature: float | None = None,
        history: Sequence[tuple[str, str]] = (),
    ) -> str | None:
        """Submit the prompt and all parent/neuron image pairs to HY4 Flash Free."""
        if len(self.parent_ids) > 1:
            prompt += "\nParent image blocks, in order: " + ", ".join(self.parent_ids) + "."
        prompt += "\nEach neuron has two images: response and fit, then residuals by angle."
        return await client.chat(
            "hy4-flash-free",
            prompt,
            images=self.images,
            effort=effort,
            temperature=temperature,
            history=history,
        )


def _plot(cell, degrees, values, grid, curve, *, residual):
    figure = Figure(figsize=(6, 3.55), layout="constrained")
    axis = figure.subplots()
    title = "Residual by angle" if residual else "Discovery response"
    axis.set_title(f"Neuron {cell:02d} · {title}", loc="left", weight="bold")
    axis.scatter(degrees, values, s=12, color="#637b9b", alpha=0.32, linewidths=0, label="Held-out trials")
    bins = np.floor(degrees / 30).astype(int)
    active = np.unique(bins)
    axis.scatter(
        active * 30 + 15,
        [values[bins == i].mean() for i in active],
        s=20,
        color="#273d71",
        marker="D",
        label="30° bin mean",
        zorder=4,
    )
    if residual:
        axis.axhline(0, color="#007d79", linewidth=1.8, label="Fit")
    else:
        axis.plot(grid, curve, color="#007d79", linewidth=2.4, label="Fit")
    axis.set(
        xlim=(0, 360),
        xticks=[0, 90, 180, 270, 360],
        xlabel="Stimulus angle (°)",
        ylabel="Residual (observed − fitted)" if residual else "Normalized response",
    )
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best", frameon=False, fontsize=8)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=115, metadata={"Title": title})
    return buffer.getvalue()


def prepare_images(
    upstream,
    angles,
    responses,
    *,
    seed: int,
    parents: Sequence[ParentFit] = (),
    max_iter: int = 2000,
) -> FrontierImages:
    """Fit each parent on the same ten exploration-only rows and render held-out trials."""
    x, y = jnp.asarray(angles), jnp.asarray(responses)
    if x.ndim != 2 or x.shape != y.shape or x.shape[0] < NEURONS or x.shape[1] < 2:
        raise ValueError("Expected matching arrays with at least ten exploration neurons and two trials.")
    chosen = np.random.default_rng(seed).choice(x.shape[0], size=NEURONS, replace=False)
    x, y = x[chosen], y[chosen]
    held = jax.random.permutation(jax.random.PRNGKey(TRIAL_SEED), jnp.arange(x.shape[1]))[x.shape[1] // 2 :]
    grid = jnp.linspace(0, 2 * jnp.pi, 361)
    grid_deg, degrees = np.linspace(0, 360, 361), np.rad2deg(np.asarray(x[:, held])) % 360
    observed = np.asarray(y[:, held])
    references = tuple(parents)
    if not references:
        seeds = upstream["seed_programs"]
        references = (ParentFit("root", seeds.neuron_model_2_jax, seeds.parameter_estimator_2),)
    images = []
    for parent in references:
        if parent.solve is None:
            score, _, _, params = score_program(upstream, parent.model, parent.estimator, x, y, max_iter)
        else:
            score, _, _, params = parent.solve((x, y), max_iter)
        if not np.isfinite(score):
            raise ValueError(f"Could not fit reference {parent.node_id!r}.")
        predict = jax.vmap(lambda row, p, model=parent.model: jnp.broadcast_to(model(row, *p), row.shape))
        curves = np.asarray(predict(jnp.broadcast_to(grid, (NEURONS, grid.size)), params))
        residuals = observed - np.asarray(predict(x[:, held], params))
        if not np.isfinite(curves).all() or not np.isfinite(residuals).all():
            raise ValueError(f"Nonfinite predictions for reference {parent.node_id!r}.")
        for i in range(NEURONS):
            images.append(partial(_plot, i + 1, degrees[i], observed[i], grid_deg, curves[i], residual=False))
            images.append(partial(_plot, i + 1, degrees[i], residuals[i], grid_deg, curves[i], residual=True))
    return FrontierImages(tuple(map(int, chosen)), tuple(p.node_id for p in parents), tuple(images))


POPULATION_BINS = 36
PREFERRED_GRID = 720


def population_residuals(model, params, angles, responses, bins=POPULATION_BINS):
    """The fit's residual over many neurons, each aligned so its fitted preferred direction is at 0.

    Each neuron's residuals are divided by the range of its fitted curve (at least a tenth of the median
    range, so flat neurons do not dominate), then averaged in `bins` bins of aligned angle. Returns the bin
    centres in degrees from -180 to 180, the mean, and its standard error.
    """
    params, angles, responses = np.asarray(params), np.asarray(angles), np.asarray(responses)
    grid = jnp.linspace(0.0, 2 * jnp.pi, PREFERRED_GRID, endpoint=False)
    curve = jax.vmap(lambda row, p: jnp.broadcast_to(model(row, *p), row.shape))
    curves = np.asarray(curve(jnp.broadcast_to(grid, (len(params), grid.size)), jnp.asarray(params)))
    preferred = np.asarray(grid)[np.argmax(curves, axis=1)]
    span = curves.max(axis=1) - curves.min(axis=1)
    span = np.maximum(span, 0.1 * np.median(span) + 1e-9)
    fitted = np.asarray(curve(jnp.asarray(angles), jnp.asarray(params)))
    relative = (responses - fitted) / span[:, None]
    aligned = np.rad2deg(np.arctan2(np.sin(angles - preferred[:, None]), np.cos(angles - preferred[:, None])))
    width = 360 / bins
    index = np.clip(np.floor((aligned + 180) / width).astype(int), 0, bins - 1).ravel()
    values = relative.ravel()
    count = np.bincount(index, minlength=bins)
    mean = np.bincount(index, values, minlength=bins) / np.maximum(count, 1)
    square = np.bincount(index, values**2, minlength=bins) / np.maximum(count, 1)
    sem = np.sqrt(np.maximum(square - mean**2, 0) / np.maximum(count, 1))
    return -180 + width * (np.arange(bins) + 0.5), mean, sem


def population_panel(label, centres, mean, sem):
    """PNG of a parent's population residual, aligned to each neuron's preferred direction."""
    figure = Figure(figsize=(6, 3.55), layout="constrained")
    axis = figure.subplots()
    axis.set_title(f"Population residual · {label}", loc="left", weight="bold")
    axis.fill_between(centres, mean - 2 * sem, mean + 2 * sem, color="#637b9b", alpha=0.25, linewidth=0)
    axis.plot(centres, mean, color="#273d71", marker="D", markersize=3, linewidth=1.6, label="Mean residual")
    axis.axhline(0, color="#007d79", linewidth=1.8, label="Fit")
    axis.set(
        xlim=(-180, 180),
        xticks=[-180, -90, 0, 90, 180],
        xlabel="Direction relative to each neuron's preferred direction (°)",
        ylabel="Residual / fitted range",
    )
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best", frameon=False, fontsize=8)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=115, metadata={"Title": "Population residual"})
    return buffer.getvalue()
