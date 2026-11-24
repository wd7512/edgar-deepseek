Add `edgar neat`: NEAT searches feed-forward tuning-curve networks whose weights are fitted per
cell and scored by the existing `objective` (10 x held-out MSE + 0.01 per parameter), then the
top programs go through the usual selection/locked-cell holdout. First run targets the 25%
discovery subset.
