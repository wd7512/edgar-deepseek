Added the population analyses from Tilbury et al. (2025):
- `edgar_study.geometry`: rcvPCA, power-law and Gaussian-decay fits in log or linear space, and independent noise for the Fig. S1 check. The paper's own noise recipe biases rcvPCA downward, because residuals from the repeat mean are negatively correlated across repeats.
- `edgar_study.paper_models`: the paper's winning model and parameter estimator, its Fig. 3e ablations, and the paper model plus this run's one-cycle wave.
- `scripts/paper_analysis/`: repeats the analyses on a finished run.
- `docs/artifacts/edgar_runs/`: a results page that takes any number of runs, listed in `runs.json` by `edgar_study.run_index`.
- `edgar_study.split_errors`: fit-trial and held-out-trial error for each cell group (discovery, selection, locked test). The paper analysis records it for every model, plus one example cell's trials and fits.
- The results page opens with a single summary table against the paper and a diagram of the data split and search, then shows how one neuron is fitted and the fit vs held-out error of the paper's equation and the winner.
- The summary table gives the double Gaussian its own column, says under each statistic which direction is better, and marks the best value. The across-runs strip is removed, and `runs.json` no longer carries headline numbers.
- The page shows the data split and the search as figures, compares every model on every chart under one set of names, adds the paper's equation to the search chart, and drops the held-out-cells chart that Figure 5 replaces. The winner is counted as 13 parameters, without the one it never uses.
- Leakage audit (docs/LEAKAGE.md): `rescore` and the analysis scripts now take candidates from the island databases, because upstream's combined database deduplicates on selection and locked-test loss. The image-reader prompt no longer hints at cusp-shaped peaks. Rescoring run 219bfc keeps the same winner.
