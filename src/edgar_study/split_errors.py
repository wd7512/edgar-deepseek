"""Fit-trial and held-out-trial error of fitted tuning curves, broken down by cell group."""

import numpy as np


def split_errors(observed, predicted, fit_columns, score_columns, groups):
    """Mean per-cell squared error on fit and held-out trials for each cell group, with standard errors."""
    if np.intersect1d(fit_columns, score_columns).size:
        raise ValueError("Fit and held-out trial columns overlap.")
    squared = (np.asarray(observed) - np.asarray(predicted)) ** 2
    per_cell = {
        "fit": squared[:, fit_columns].mean(axis=1),
        "held_out": squared[:, score_columns].mean(axis=1),
    }
    errors = {}
    for name, rows in groups.items():
        entry = {"cells": int(len(rows))}
        for half, values in per_cell.items():
            chosen = values[rows]
            entry[half] = float(chosen.mean())
            entry[f"{half}_sem"] = (
                float(chosen.std(ddof=1) / np.sqrt(len(chosen))) if len(chosen) > 1 else 0.0
            )
        errors[name] = entry
    return errors
