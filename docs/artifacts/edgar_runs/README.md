# EDGAR runs page

`index.html` shows every run on one page: one table with every statistic for the double Gaussian, the paper's equation and each run's winner, then each run's figures in `runs.json` order. `#run=<id>` in the URL jumps to that run. Serve the folder over HTTP (for example `uv run python -m http.server -d docs/artifacts/edgar_runs`); the page loads `runs.json` and then each run's files.

To add a finished, rescored run:

```sh
R=docs/artifacts/edgar_runs/runs/<run id>
uv run python scripts/paper_analysis/page_data.py --run results/runs/<run id> --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --out $R/data.json
uv run python scripts/paper_analysis/run.py --run results/runs/<run id> --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --out $R/paper_analysis.json
uv run python -m edgar_study.run_index docs/artifacts/edgar_runs
```

`paper_analysis.json` (about two minutes per run on 15 CPU cores, five of them for the all-cells fits) is optional; without it the population sections are hidden. An optional `notes.json` holds the run's hand-written equation and caveats (`equation_title`, `equation_caption`, `equation_html`, `caveats`).
