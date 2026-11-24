# EDGAR runs page

`index.html` opens on an overview: one table with every statistic for the double Gaussian, the paper's equation and each run's winner, then the figures that are the same for every run (data split, search, rcvPCA validation). Clicking a run opens its own view at `#run=<id>`: the equation, a table of how it differs from the paper's equation (`differences` in `notes.json`), a figure of where its fitted curves part from the paper's equation (`delta` in `paper_analysis.json`), its charts and caveats. Every chart has a full-screen button. `#run=<id>` in the URL jumps to that run. Serve the folder over HTTP (for example `uv run python -m http.server -d docs/artifacts/edgar_runs`); the page loads `runs.json` and then each run's files.

To add a finished, rescored run:

```sh
R=docs/artifacts/edgar_runs/runs/<run id>
uv run python scripts/paper_analysis/page_data.py --run results/runs/<run id> --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --out $R/data.json
uv run python scripts/paper_analysis/run.py --run results/runs/<run id> --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --out $R/paper_analysis.json
uv run python -m edgar_study.run_index docs/artifacts/edgar_runs
```

`paper_analysis.json` (about two minutes per run on 15 CPU cores, five of them for the all-cells fits) is optional; without it the population sections are hidden. An optional `notes.json` holds the run's hand-written equation and caveats (`equation_title`, `equation_caption`, `equation_html`, `caveats`).

## Is the LLM needed?

The overview's "Is the LLM needed?" section reads `null_test.json`, written by `scripts/null_compare/run.py`. It refits every LLM winner, every no-LLM winner and the double Gaussian on the locked-test neurons and runs a paired bootstrap over neurons for each LLM–no-LLM pair; the section is hidden when the file is absent. No-LLM runs come from `edgar null-search` (genetic programming inside upstream's unchanged search) and are added like any other run; their model shows as "GP (no LLM)", which also drops the LLM-only parts of the search figure. To refresh it:

```sh
uv run python scripts/null_compare/run.py --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --llm results/runs/<llm run>... --null results/runs/<gp run>... --out results/null_test.json
cp results/null_test.json docs/artifacts/edgar_runs/null_test.json
```
