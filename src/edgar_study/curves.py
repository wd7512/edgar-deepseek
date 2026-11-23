"""Ten tuning-curve equations from Tilbury et al. 2025 and a benchmark of how long each takes to fit.

The equations are the human-written seeds and the winner's family tree from the paper's
Supplementary Text 3, plus the two von Mises seeds shipped with upstream. S2, the double Gaussian,
is the paper's baseline and H27 is its winning equation. A12 is left out because the paper lists it
as identical to A9.

Every curve is a JAX model `model(theta, *params)` with a numpy parameter estimator, so the
benchmark can run each one through `hypothesis_engine.objective` and time the same path a search
pays for every proposed program: estimate, validate, fit and score.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from edgar_study import fitting
from edgar_study.upstream import PENALTY, import_upstream

log = logging.getLogger(__name__)

BASELINE = "double-gaussian"
WINNER = "winner"
SIGMA_RANGE = (0.05, 3.0)
EXPONENT_RANGE = (0.5, 8.0)
KAPPA_MIN = 1e-3
ESTIMATOR_BINS = 36


def _wrap(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _width(w):
    return jnp.clip(w, *SIGMA_RANGE)


def _exponent(p):
    return jnp.clip(p, *EXPONENT_RANGE)


def _peak(theta, centre, amplitude, sigma_plus, sigma_minus, exponent, slope=0.0):
    """A·exp(−½(|d|/σ±)^p + k·d) with d the signed angle from the centre in (−π, π]."""
    d = _wrap(theta - centre)
    sigma = jnp.where(d >= 0, _width(sigma_plus), _width(sigma_minus))
    return jnp.clip(amplitude, 0, None) * jnp.exp(
        -0.5 * ((jnp.abs(d) + 1e-9) / sigma) ** _exponent(exponent) + slope * d
    )


def _baseline(b):
    return jnp.clip(b, 0, None)


def single_gaussian(theta, phi, b, a, s):
    return _baseline(b) + _peak(theta, phi, a, s, s, 2.0)


def double_gaussian(theta, phi, b, a1, a2, s):
    return _baseline(b) + _peak(theta, phi, a1, s, s, 2.0) + _peak(theta, phi + jnp.pi, a2, s, s, 2.0)


def asymmetric_widths(theta, phi, b, a1, a2, s1p, s1m, s2p, s2m):
    return _baseline(b) + _peak(theta, phi, a1, s1p, s1m, 2.0) + _peak(theta, phi + jnp.pi, a2, s2p, s2m, 2.0)


def asymmetric_shared_widths(theta, phi, b, a1, a2, sp, sm):
    return _baseline(b) + _peak(theta, phi, a1, sp, sm, 2.0) + _peak(theta, phi + jnp.pi, a2, sp, sm, 2.0)


def free_peaks(theta, phi1, phi2, b, a1, a2, s1, s2):
    return _baseline(b) + _peak(theta, phi1, a1, s1, s1, 2.0) + _peak(theta, phi2, a2, s2, s2, 2.0)


def generalised(theta, phi, b, a1, a2, s1p, s1m, s2p, s2m, p1, p2):
    return _baseline(b) + _peak(theta, phi, a1, s1p, s1m, p1) + _peak(theta, phi + jnp.pi, a2, s2p, s2m, p2)


def linear_term(theta, phi, b, a1, a2, s1p, s1m, s2p, s2m, m1, m2):
    return (
        _baseline(b)
        + _peak(theta, phi, a1, s1p, s1m, 2.0, m1)
        + _peak(theta, phi + jnp.pi, a2, s2p, s2m, 2.0, m2)
    )


def winner(theta, phi1, phi2, b, a1, a2, s1p, s1m, s2p, s2m, p1, p2):
    """H27: two asymmetric generalised Gaussian peaks at free positions, 11 parameters."""
    return _baseline(b) + _peak(theta, phi1, a1, s1p, s1m, p1) + _peak(theta, phi2, a2, s2p, s2m, p2)


def _von_mises(theta, phi, amplitude, kappa):
    return jnp.clip(amplitude, 0, None) * jnp.exp(
        jnp.clip(kappa, KAPPA_MIN, None) * (jnp.cos(theta - phi) - 1)
    )


def von_mises(theta, phi, b, a, kappa):
    return _baseline(b) + _von_mises(theta, phi, a, kappa)


def double_von_mises(theta, phi, b, a1, a2, k1, k2):
    return _baseline(b) + _von_mises(theta, phi, a1, k1) + _von_mises(theta, phi + jnp.pi, a2, k2)


@dataclass(frozen=True)
class Curve:
    name: str
    paper_id: str
    equation: str
    param_names: tuple[str, ...]
    model: Callable

    @property
    def n_params(self):
        return len(self.param_names)

    def estimator(self, theta, counts):
        """Initial parameters from a binned tuning curve, one cell at a time like upstream's estimators."""
        features = _features(np.asarray(theta), np.asarray(counts))
        return np.array([features[name] for name in self.param_names], dtype=np.float32)


CURVES = (
    Curve("single-gaussian", "S1", "b + A·exp(−½|d/σ|²)", ("phi", "b", "a", "s"), single_gaussian),
    Curve(
        "double-gaussian",
        "S2",
        "b + A1·exp(−½|d/σ|²) + A2·exp(−½|(d−π)/σ|²)",
        ("phi", "b", "a1", "a2", "s"),
        double_gaussian,
    ),
    Curve(
        "asym-widths",
        "A9",
        "S2 with σ1±, σ2± (left and right widths per peak)",
        ("phi", "b", "a1", "a2", "s1p", "s1m", "s2p", "s2m"),
        asymmetric_widths,
    ),
    Curve(
        "asym-shared-widths",
        "A34",
        "S2 with one σ± shared by both peaks",
        ("phi", "b", "a1", "a2", "sp", "sm"),
        asymmetric_shared_widths,
    ),
    Curve(
        "free-peaks",
        "A35",
        "b + A1·exp(−½|(θ−φ1)/σ1|²) + A2·exp(−½|(θ−φ2)/σ2|²)",
        ("phi1", "phi2", "b", "a1", "a2", "s1", "s2"),
        free_peaks,
    ),
    Curve(
        "generalised",
        "B35",
        "A9 with exponents p1, p2: exp(−½|d/σ±|^p)",
        ("phi", "b", "a1", "a2", "s1p", "s1m", "s2p", "s2m", "p1", "p2"),
        generalised,
    ),
    Curve(
        "linear-term",
        "B12",
        "A9 with a linear term k·d added inside each exponent",
        ("phi", "b", "a1", "a2", "s1p", "s1m", "s2p", "s2m", "m1", "m2"),
        linear_term,
    ),
    Curve(
        WINNER,
        "H27",
        "b + A1·exp(−½|(θ−φ1)/σ1±|^p1) + A2·exp(−½|(θ−φ2)/σ2±|^p2)",
        ("phi1", "phi2", "b", "a1", "a2", "s1p", "s1m", "s2p", "s2m", "p1", "p2"),
        winner,
    ),
    Curve("von-mises", "seed 3", "b + A·exp(κ(cos(θ−φ) − 1))", ("phi", "b", "a", "kappa"), von_mises),
    Curve(
        "double-von-mises",
        "seed 4",
        "b + A1·exp(κ1(cos(θ−φ) − 1)) + A2·exp(κ2(cos(θ−φ−π) − 1))",
        ("phi", "b", "a1", "a2", "k1", "k2"),
        double_von_mises,
    ),
)


def pack(name, **named):
    """Order named parameters as the curve's model expects."""
    curve = next(c for c in CURVES if c.name == name)
    return np.array([named[p] for p in curve.param_names], dtype=np.float32)


def _features(theta, counts):
    """Every initial value any curve needs, from a smoothed circular tuning curve."""
    n = ESTIMATOR_BINS
    bins = np.clip((theta * n / (2 * np.pi)).astype(int), 0, n - 1)
    sums = np.bincount(bins, weights=counts, minlength=n)
    seen = np.bincount(bins, minlength=n)
    curve = np.where(seen > 0, sums / np.maximum(seen, 1), np.mean(counts))
    curve = np.convolve(np.pad(curve, 1, mode="wrap"), [0.25, 0.5, 0.25], mode="valid")
    peak = int(np.argmax(curve))
    baseline = float(np.min(curve))
    a1 = float(curve[peak] - baseline)
    a2 = max(float(curve[(peak + n // 2) % n] - baseline), 0.0)
    above = curve[(np.arange(-5, 6) + peak) % n] >= baseline + a1 / 2
    sigma = float(np.clip(2 * np.pi * np.sum(above) / n / 2.355, 0.1, 1.5))
    phi = (peak + 0.5) * 2 * np.pi / n
    return {
        "phi": phi,
        "phi1": phi,
        "phi2": phi + np.pi,
        "b": baseline,
        "a": a1,
        "a1": a1,
        "a2": a2,
        "s": sigma,
        "s1": sigma,
        "s2": sigma,
        "sp": sigma,
        "sm": sigma,
        "s1p": sigma,
        "s1m": sigma,
        "s2p": sigma,
        "s2m": sigma,
        "p1": 2.0,
        "p2": 2.0,
        "m1": 0.0,
        "m2": 0.0,
        "kappa": 1 / sigma**2,
        "k1": 1 / sigma**2,
        "k2": 1 / sigma**2,
    }


def synthetic_cells(n_cells=24, n_trials=144, seed=0):
    """Cells drawn from the winning equation with random parameters, scaled like upstream's data.

    Returns angles and responses of shape (n_cells, n_trials); each cell's responses have norm 100.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, (n_cells, n_trials))
    y = np.empty_like(x)
    for cell in range(n_cells):
        phi1 = rng.uniform(0, 2 * np.pi)
        a1 = rng.uniform(0.6, 1.5)
        truth = pack(
            WINNER,
            phi1=phi1,
            phi2=phi1 + np.pi + rng.normal(0, 0.3),
            b=rng.uniform(0.05, 0.3),
            a1=a1,
            a2=a1 * rng.uniform(0.1, 0.8),
            s1p=rng.uniform(0.2, 0.6),
            s1m=rng.uniform(0.2, 0.6),
            s2p=rng.uniform(0.2, 0.6),
            s2m=rng.uniform(0.2, 0.6),
            p1=rng.uniform(1.0, 2.0),
            p2=rng.uniform(1.0, 2.0),
        )
        rate = np.asarray(winner(x[cell], *truth))
        y[cell] = np.clip(rate + rng.normal(0, 0.15 * a1, n_trials), 0, None)
    y = 100 * y / np.linalg.norm(y, axis=1, keepdims=True)
    return jnp.asarray(x), jnp.asarray(y)


def benchmark(
    fitters=("upstream", "adam", "lm"),
    n_cells=24,
    n_trials=144,
    max_iter=1_000,
    curves=CURVES,
    upstream=Path("."),
):
    """Time `hypothesis_engine.objective` for each curve under each fitter on the same synthetic cells.

    Each fit runs twice. The first call includes XLA compilation, which a search pays for every new
    program; the repeat shows the steady-state cost. Returns one row per (fitter, curve).
    """
    modules = import_upstream(Path(upstream).resolve())
    engine, loss = modules["hypothesis_engine"], modules["loss_functions"].quadratic_loss
    x, y = synthetic_cells(n_cells, n_trials)
    rows = []
    for fitter in fitters:
        objective = (
            engine.objective if fitter == "upstream" else fitting.make_objective(engine.objective, fitter)
        )
        for curve in curves:
            timings = []
            for _ in range(2):
                np.random.seed(0)
                start = time.perf_counter()
                _, _, score, params = objective(
                    curve.model,
                    curve.estimator,
                    loss,
                    x,
                    y,
                    param_penalty_weight=PENALTY,
                    max_iter=max_iter,
                )
                timings.append(time.perf_counter() - start)
            row = {
                "program": curve.name,
                "paper_id": curve.paper_id,
                "fitter": fitter,
                "params": curve.n_params,
                "first_seconds": timings[0],
                "repeat_seconds": timings[1],
                "train_loss": fitting.train_loss(curve.model, loss, params, x, y),
                "test_loss": score - PENALTY * curve.n_params,
                "score": score,
            }
            log.info(
                "%s %s: first %.2fs, repeat %.2fs, test loss %.4f",
                fitter,
                curve.name,
                row["first_seconds"],
                row["repeat_seconds"],
                row["test_loss"],
            )
            rows.append(row)
    return rows


def benchmark_table(rows):
    """Format benchmark rows as a Markdown table."""
    lines = [
        "| Program | Paper ID | Params | Fitter | First call (s) | Repeat (s) | Train loss | Test loss |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['program']} | {r['paper_id']} | {r['params']} | {r['fitter']} | "
            f"{r['first_seconds']:.2f} | {r['repeat_seconds']:.2f} | "
            f"{r['train_loss']:.4f} | {r['test_loss']:.4f} |"
        )
    return "\n".join(lines)
