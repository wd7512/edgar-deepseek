Added the population analyses from Tilbury et al. (2025):
- `edgar_study.geometry`: rcvPCA, power-law and Gaussian-decay fits in log or linear space, and independent noise for the Fig. S1 check. The paper's own noise recipe biases rcvPCA downward, because residuals from the repeat mean are negatively correlated across repeats.
- `edgar_study.paper_models`: the paper's winning model and parameter estimator, its Fig. 3e ablations, and the paper model plus this run's one-cycle wave.
- `scripts/paper_analysis/`: repeats the analyses on a finished run.
- `docs/artifacts/edgar_runs/`: a results page that takes any number of runs, indexed by `edgar_study.run_index`.
