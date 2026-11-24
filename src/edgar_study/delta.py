"""How a winning equation's fitted curves differ from the paper's equation's, neuron by neuron."""

import numpy as np

GAIN_EDGES = np.arange(-0.02, 0.0601, 0.005)
SIZE_EDGES = np.arange(0, 1.01, 0.05)


def delta_analysis(paper_r2, winner_r2, paper_curves, winner_curves, n_examples=4):
    """Held-out R² gain of the winner over the paper's equation, the size of the curve difference, and example rows.

    Size is the largest gap between the two fitted curves, as a share of the paper curve's range.
    Example rows are the largest gain, the second largest, the 90th percentile and the median.
    """
    gain = np.asarray(winner_r2) - np.asarray(paper_r2)
    diff = np.asarray(winner_curves) - np.asarray(paper_curves)
    span = np.ptp(np.asarray(paper_curves), axis=1)
    size = np.abs(diff).max(axis=1) / np.maximum(span, 1e-9)
    order = np.argsort(-gain, kind="stable")
    rows = []
    for k in (order[0], order[1 % len(order)], order[len(order) // 10], order[len(order) // 2]):
        if int(k) not in rows:
            rows.append(int(k))
    return {
        "gain": {
            "median": float(np.median(gain)),
            "mean": float(np.mean(gain)),
            "share_better": float(np.mean(gain > 0)),
            "share_worse": float(np.mean(gain < 0)),
            "edges": GAIN_EDGES.round(3).tolist(),
            "hist": np.histogram(np.clip(gain, GAIN_EDGES[0], GAIN_EDGES[-1] - 1e-4), bins=GAIN_EDGES)[
                0
            ].tolist(),
        },
        "size": {
            "median": float(np.median(size)),
            "share_over_10pct": float(np.mean(size > 0.1)),
            "edges": SIZE_EDGES.round(2).tolist(),
            "hist": np.histogram(np.clip(size, 0, 0.999), bins=SIZE_EDGES)[0].tolist(),
        },
        "rows": rows[:n_examples],
    }
