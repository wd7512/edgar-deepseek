"""Repeat Tilbury et al. (2025)'s population analyses on a finished run.

Fits the double Gaussian seed, the run's winner, the paper's winning model and its Fig. 3e ablations to
every eligible cell, then reports per-cell held-out R², peakiness, rcvPCA of the recording (Fig. 1d),
PCA spectra of the fitted tuning curves (Fig. 3f), and rcvPCA on model curves plus the recording's own
trial-to-trial noise (Fig. S1).

    uv run python scripts/paper_analysis/run.py --run results/runs/<id> --dataset data/<file>.npy
"""

import argparse
import inspect
import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from edgar_study import fitting, geometry, paper_models
from edgar_study.datasets import load_dataset, prepare
from edgar_study.exponents import fitted_exponents
from edgar_study.split_errors import split_errors
from edgar_study.upstream import import_upstream, score_program, searched_programs

log = logging.getLogger("paper_analysis")
ROOT = Path(__file__).resolve().parents[2]
MAX_ITER = 2_000
N_BINS = 256
N_REPEATS = 6
KEPT_DIMENSIONS = 128


def winner(run_dir, utils):
    rescore = json.loads((run_dir / "rescore.json").read_text())
    best = rescore["winner"]
    frame = searched_programs(run_dir)
    ids = (
        frame.iteration_number.astype(str)
        + "_"
        + frame.birth_island.astype(str)
        + "_"
        + frame.batch_index.astype(str)
    )
    estimator_code = frame.loc[ids == best["id"], "parameter_estimator_code_string"].iloc[0]
    model = utils.str_to_func(best["jax_code"], "neuron_model")
    return best["id"], model, utils.str_to_func(estimator_code, "parameter_estimator"), best["jax_code"]


def predict(model, params, theta):
    theta = jnp.asarray(theta)
    if theta.ndim == 1:
        return np.asarray(jax.vmap(lambda p: model(theta, *p))(jnp.asarray(params)))
    return np.asarray(jax.vmap(lambda p, t: model(t, *p))(jnp.asarray(params), theta))


def spectrum_summary(spectrum):
    summary = {"spectrum": [float(v) for v in spectrum[:KEPT_DIMENSIONS]]}
    for space in ("log", "linear"):
        power = geometry.fit_power_law(spectrum, space=space)
        gaussian = geometry.fit_gaussian_decay(spectrum, space=space)
        summary[space] = {
            "alpha": power.slope,
            "power_intercept": power.intercept,
            "power_mse": power.mse,
            "M": gaussian.slope,
            "gaussian_intercept": gaussian.intercept,
            "gaussian_mse": gaussian.mse,
        }
    return summary


def source(model):
    try:
        return inspect.getsource(model)
    except (OSError, TypeError):
        return ""


def fit_population(upstream, model, estimator, x, y, centres, residuals, code=""):
    """Fit every cell, then return the fitted parameters, curves and their spectra (Fig. 3f and S1)."""
    score, mse, count, params = score_program(upstream, model, estimator, x, y, MAX_ITER)
    params = np.asarray(params)
    curves = predict(model, params, centres)
    entry = {
        "parameters": count,
        "score": score,
        "mse": mse,
        "pca": spectrum_summary(geometry.pca_spectrum(curves)),
        "rcvpca_with_noise": spectrum_summary(
            geometry.rcvpca(curves[:, :, None] + geometry.independent_noise(residuals))
        ),
        "rcvpca_with_paper_noise": spectrum_summary(geometry.rcvpca(curves[:, :, None] + residuals)),
    }
    names = [n for n in inspect.signature(model).parameters if n != "theta"]
    exponents = fitted_exponents(names, params, code or source(model))
    if exponents is not None:
        entry["peakiness"] = peakiness(*exponents)
    return entry, params, curves


def describe(name, entry, seconds):
    log.info(
        "%s: %d params, PCA α %.2f (linear %.2f), rcvPCA+noise α %.2f (paper noise %.2f), %.0fs",
        name,
        entry["parameters"],
        entry["pca"]["log"]["alpha"],
        entry["pca"]["linear"]["alpha"],
        entry["rcvpca_with_noise"]["log"]["alpha"],
        entry["rcvpca_with_paper_noise"]["log"]["alpha"],
        seconds,
    )


def peakiness(p1, p2):
    lowest = np.minimum(p1, p2)
    return {
        "median_min_p": float(np.median(lowest)),
        "quartiles": [float(np.percentile(lowest, q)) for q in (25, 75)],
        "theory_alpha": float(2 * (1 + np.median(lowest))),
        "hist": np.histogram(lowest, bins=np.arange(0.0, 4.01, 0.1))[0].tolist(),
    }


def components(params, grid):
    """Baseline, each peak and the wave of the AI model + wave, evaluated on `grid`."""
    theta = jnp.asarray(grid)
    peak_1 = params.copy()
    peak_1[[1, 3]] = 0
    peak_2 = params.copy()
    peak_2[[1, 2]] = 0
    return {
        "baseline": [float(max(params[1], 0))] * len(grid),
        "peak_1": np.asarray(paper_models.ai_model(theta, *peak_1[:11])),
        "peak_2": np.asarray(paper_models.ai_model(theta, *peak_2[:11])),
        "wave": params[11] * np.cos(grid) + params[12] * np.sin(grid),
    }


def wave_analysis(models, fitted, results, winner_id, x, y, n_examples=4):
    """How large the one-cycle wave is, what it adds to the paper's model, and example decompositions."""
    plain, waved = results["AI model"]["r2"], results["AI model + wave"]["r2"]
    gain = waved - plain
    params = fitted["AI model + wave"]
    size = np.hypot(params[:, 11], params[:, 12]) / np.maximum(np.clip(params[:, 2], 0, None), 1e-9)
    winner_name = f"EDGAR winner {winner_id}"
    w = fitted[winner_name]
    names = paper_models.parameter_names(models[winner_name][0])[: w.shape[1]]

    def column(name, low, high):
        return np.clip(w[:, names.index(name)], low, high) if name in names else np.zeros(len(w))

    winner_wave = np.abs(
        column("baseline_modulation", 0, 1.5) * np.exp(1j * column("baseline_phase", 0, 2 * np.pi))
        + column("wrap_asymmetry", -1.5, 1.5) * np.exp(1j * column("theta_pref", 0, 2 * np.pi))
    ) / np.maximum(column("amplitude_1", 0, 20), 1e-9)
    edges = np.arange(0, 1.01, 0.05)
    order = np.argsort(-gain)
    picks = [order[0], order[1], order[len(order) // 10], order[len(order) // 2]][:n_examples]
    grid = np.linspace(0, 2 * np.pi, 181)
    examples = []
    for k in picks:
        theta, response = np.asarray(x[k]), np.asarray(y[k])
        bins = (theta / (2 * np.pi) * 24).astype(int) % 24
        curves = {
            name: np.asarray(models[name][0](jnp.asarray(grid), *fitted[name][k]))
            for name in ("Double Gaussian", "AI model", "AI model + wave", winner_name)
        }
        examples.append(
            {
                "row": int(k),
                "r2": {
                    name: float(results[name]["r2"][k])
                    for name in ("Double Gaussian", "AI model", "AI model + wave", winner_name)
                },
                "wave_size": float(size[k]),
                "means": [
                    [b * 15 + 7.5, float(response[bins == b].mean())] for b in range(24) if np.any(bins == b)
                ],
                "curves": {name: v.round(3).tolist() for name, v in curves.items()},
                "components": {
                    name: np.round(v, 3).tolist() for name, v in components(params[k], grid).items()
                },
            }
        )
    return {
        "r2_gain_over_ai_model": {
            "median": float(np.median(gain)),
            "mean": float(np.mean(gain)),
            "share_better": float(np.mean(gain > 0)),
            "hist": np.histogram(np.clip(gain, -0.02, 0.0599), bins=np.arange(-0.02, 0.0601, 0.005))[
                0
            ].tolist(),
        },
        "size_edges": edges.round(2).tolist(),
        "size_hist": np.histogram(np.clip(size, 0, 0.999), bins=edges)[0].tolist(),
        "winner_size_hist": np.histogram(np.clip(winner_wave, 0, 0.999), bins=edges)[0].tolist(),
        "median_size": float(np.median(size)),
        "winner_fitted_parameters": names,
        "winner_median_size": float(np.median(winner_wave)),
        "share_size_over_10pct": float(np.mean(size > 0.1)),
        "grid_deg": (grid * 180 / np.pi).round(1).tolist(),
        "examples": examples,
    }


def example_cell(models, fitted, results, groups, x, y, fit_trials, held_out):
    """One locked-test cell's trials, split into fit and held-out halves, with three models' fits."""
    locked = groups["locked_test"]
    scores = results["AI model"]["r2"][locked]
    row = int(locked[np.argsort(scores)[int(0.9 * len(scores))]])
    grid = np.linspace(0, 2 * np.pi, 181)
    shown = ["Double Gaussian", "AI model", next(k for k in models if k.startswith("EDGAR winner"))]
    observed, angles = np.asarray(y[row]), np.asarray(x[row])
    return {
        "row": row,
        "grid": grid.round(4).tolist(),
        "fit_trials": {
            "x": angles[fit_trials].round(4).tolist(),
            "y": observed[fit_trials].round(3).tolist(),
        },
        "held_out_trials": {
            "x": angles[held_out].round(4).tolist(),
            "y": observed[held_out].round(3).tolist(),
        },
        "curves": {
            name: predict(models[name][0], fitted[name][row : row + 1], grid)[0].round(3).tolist()
            for name in shown
        },
        "r2": {name: float(results[name]["r2"][row]) for name in shown},
    }


def improved_cells(models, fitted, results, groups, x, y):
    """Four locked-test cells by how much the winner improves held-out R² over the double Gaussian."""
    winner_name = next(k for k in models if k.startswith("EDGAR winner"))
    shown = ["Double Gaussian", "AI model", winner_name]
    locked = groups["locked_test"]
    gain = results[winner_name]["r2"][locked] - results["Double Gaussian"]["r2"][locked]
    order = np.argsort(-gain)
    picks = {
        "largest gain": order[0],
        "4th largest gain": order[3],
        "75th percentile": order[len(order) // 4],
        "median": order[len(order) // 2],
    }
    grid = np.linspace(0, 2 * np.pi, 181)
    cells = []
    for label, k in picks.items():
        row = int(locked[k])
        theta, response = np.asarray(x[row]), np.asarray(y[row])
        bins = np.round(theta / (2 * np.pi) * 12).astype(int) % 12
        cells.append(
            {
                "row": row,
                "label": label,
                "points": np.stack([np.degrees(theta), response], axis=1).round(2).tolist(),
                "means": [
                    [b * 30.0, float(response[bins == b].mean())] for b in range(12) if np.any(bins == b)
                ],
                "curves": {
                    name: predict(models[name][0], fitted[name][row : row + 1], grid)[0].round(3).tolist()
                    for name in shown
                },
                "r2": {name: float(results[name]["r2"][row]) for name in shown},
            }
        )
    return {"grid_deg": np.degrees(grid).round(1).tolist(), "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fitter", default="lm-bounded")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()

    upstream = import_upstream(ROOT)
    fitting.install(upstream["hypothesis_engine"], args.fitter)
    utils, seeds = upstream["utils"], upstream["seed_programs"]
    response, angles, _ = load_dataset(args.dataset, utils.extract_stimulus_related_response)
    x, y, groups, indices = prepare(response, angles, "strict-holdout-v1")
    cells = indices["eligible_cell_ids"]
    scale = 100 / np.linalg.norm(response[cells[:, None], indices["recording_trial_ids"]], axis=1)
    held_out = indices["score_trial_columns"]
    fit_trials = indices["fit_trial_columns"]
    split_groups = {**groups, "all": np.arange(len(cells))}

    winner_id, winner_model, winner_estimator, winner_code = winner(args.run.resolve(), utils)
    codes = {f"EDGAR winner {winner_id}": winner_code}
    models = {
        "Double Gaussian": (seeds.neuron_model_2_jax, seeds.parameter_estimator_2),
        **paper_models.LADDER,
        "AI model + wave": (paper_models.ai_wave_model, paper_models.ai_wave_estimator),
        f"EDGAR winner {winner_id}": (winner_model, winner_estimator),
    }

    log.info("Binning %d trials into %d bins × %d repeats", angles.size, N_BINS, N_REPEATS)
    binned_all = geometry.bin_repeats(response, angles, N_BINS, N_REPEATS)
    centres = geometry.bin_centres(N_BINS)
    binned = binned_all[cells] * scale[:, None, None]
    residuals = binned - binned.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(response, axis=1)
    every = np.flatnonzero(norms > 0)
    binned_every = binned_all[every] * (100 / norms[every])[:, None, None]
    residuals_every = binned_every - binned_every.mean(axis=2, keepdims=True)

    log.info("rcvPCA on %d eligible cells and all %d responsive cells", len(cells), len(every))
    data = {
        "eligible": spectrum_summary(geometry.rcvpca(binned)),
        "all_cells": spectrum_summary(geometry.rcvpca(binned_every)),
    }

    results, fitted = {}, {}
    for name, (model, estimator) in models.items():
        fit_started = time.time()
        entry, params, curves = fit_population(
            upstream, model, estimator, x, y, centres, residuals, codes.get(name, "")
        )
        observed = np.asarray(y[:, held_out])
        residual = observed - predict(model, params, x[:, held_out])
        total = np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2, axis=1)
        entry["r2"] = 1 - np.sum(residual**2, axis=1) / total
        entry["split_errors"] = split_errors(
            np.asarray(y), predict(model, params, x), fit_trials, held_out, split_groups
        )
        if name == "AI model":
            entry["example_curves"] = curves[:: max(1, len(cells) // 40)][:40].round(3).tolist()
        results[name] = entry
        fitted[name] = params
        describe(name, entry, time.time() - fit_started)

    wave = wave_analysis(models, fitted, results, winner_id, x, y)
    example = example_cell(models, fitted, results, groups, x, y, fit_trials, held_out)
    improved = improved_cells(models, fitted, results, groups, x, y)

    baseline_r2 = results["Double Gaussian"]["r2"]
    for entry in results.values():
        gain = entry["r2"] - baseline_r2
        entry["r2"] = {
            "median": float(np.median(entry["r2"])),
            "mean": float(np.mean(entry["r2"])),
            "gain_over_double_gaussian": {
                "median": float(np.median(gain)),
                "mean": float(np.mean(gain)),
                "share_better": float(np.mean(gain > 0)),
            },
        }

    all_cells = {}
    x_every = jnp.asarray(np.broadcast_to(angles, (len(every), angles.size)))
    y_every = jnp.asarray(response[every] * (100 / norms[every])[:, None])
    for name in ("Double Gaussian", "AI model", f"EDGAR winner {winner_id}"):
        fit_started = time.time()
        model, estimator = models[name]
        entry, _, _ = fit_population(
            upstream, model, estimator, x_every, y_every, centres, residuals_every, codes.get(name, "")
        )
        all_cells[name] = entry
        describe(f"{name} (all cells)", entry, time.time() - fit_started)

    output = {
        "run": args.run.name,
        "fitter": args.fitter,
        "cells": int(len(cells)),
        "bins": N_BINS,
        "repeats": N_REPEATS,
        "fit_range": list(geometry.FIT_RANGE),
        "paper": {"data_alpha": 4.26, "ai_model_alpha": 4.34, "gaussian_M": 9.04, "median_min_p": 1.20},
        "data": data,
        "models": results,
        "all_cells_models": all_cells,
        "wave": wave,
        "example_cell": example,
        "improved_cells": improved,
        "all_cells_count": int(len(every)),
        "seconds": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
