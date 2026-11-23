# EDGAR × OpenCode Go

Runs [EDGAR](https://github.com/reillytilbury/EDGAR)'s equation-discovery loop unchanged, with OpenCode Go models in place of Gemini. The six upstream modules in `src/edgar_upstream/` are byte-identical to upstream commit [`0943922`](https://github.com/reillytilbury/EDGAR/tree/0943922dcb73b12dbee8e440b3e1ff700cb49c10); `src/edgar_study/` swaps only the LLM call layer and the curve fitter at runtime.

- [docs/PHASE_0.md](docs/PHASE_0.md): the like-for-like replication and its results.
- [docs/ROADMAP.md](docs/ROADMAP.md): ideas for later phases.
- [METHOD_DIFFS.md](METHOD_DIFFS.md): every known difference from the paper and from upstream.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --locked
```

Download dataset 1 ([Stringer et al.](https://janelia.figshare.com/articles/dataset/8279387), 859 MB, CC BY-NC 4.0). Every command checks it against a pinned SHA-256.

```bash
mkdir -p data
curl --fail --location https://ndownloader.figshare.com/files/15479246 \
  --output data/gratings_drifting_GT1_2019_04_12_1.npy
```

For search runs, put the OpenCode Go key in a file outside the repository and set `OPENCODE_API_KEY_FILE` to its path (`OPENCODE_API_KEY` also works). Logs never contain the key.

## Usage

Set these once per shell:

```bash
export JAX_PLATFORMS=cpu DATA=data/gratings_drifting_GT1_2019_04_12_1.npy
```

**Baseline** (no key needed; `--synthetic` instead of `--dataset` for a quick check):

```bash
uv run edgar baseline --dataset $DATA --output results/dataset1
```

**Search.** Check the model responds, then run the paper-sized search (8 islands × 6 proposals × 12 generations):

```bash
uv run edgar preflight --model hy4-flash-free
```

```bash
uv run edgar run --model hy4-flash-free --dataset $DATA --iterations 12
```

For a run of about 30 minutes, search a quarter of the discovery cells with a smaller search:

```bash
uv run edgar run --model hy4-flash-free --dataset $DATA \
  --iterations 2 --islands 4 --batch-size 6 --cell-fraction 0.25
```

**Rescore** the run's best programs on the held-out selection and locked-test cells. Writes `rescore.json`:

```bash
uv run edgar rescore --run results/runs/<run-id> --dataset $DATA
```

### Data split

The `strict-holdout-v1` protocol splits the 2,110 eligible cells into 1,055 discovery cells (identical to upstream's training half), 527 model-selection cells and 528 locked test cells. Baseline scores:

| Gaussian seed | Discovery MSE | Selection MSE | Discovery score | Selection score |
| --- | ---: | ---: | ---: | ---: |
| Single | 2.997599 | 2.998546 | 30.015987 | 30.025467 |
| Double | 2.880491 | 2.877630 | 28.854914 | 28.826298 |

### Run outputs

Each `results/runs/<run-id>/` holds `manifest.json` (models, settings, git SHAs, spend), `api_calls.jsonl` (one row per request), `translations.jsonl` (NumPy → JAX translations used by `rescore`) and upstream's `program_databases/`.

### Fitters

Nearly all run time is per-cell curve fitting. `--fitter` picks how it runs; `rescore` uses the run's fitter unless told otherwise.

| `--fitter` | What it does |
| --- | --- |
| `auto`, `lm-bounded` (default) | Levenberg–Marquardt with capped steps and a per-cell tolerance stop. About 8× faster than `upstream`, scores within 0.7%. |
| `upstream` | Upstream's 1,000 Adam steps, unchanged. Use for exact reproduction. |
| `adam` | The same Adam steps compiled and split across CPU cores. Same scores, about 1.9× faster. |
| `lm` | Unbounded Levenberg–Marquardt. Experiments only. |

Timings are in [scripts/fit_timings/results.md](scripts/fit_timings/results.md).

### Spending safeguards

- Only models the Go price list shows as free (e.g. `hy4-flash-free`) can be called without `--allow-paid`.
- Every request's list-equivalent cost goes to `results/spend.jsonl`; requests are refused once it reaches $9.50.

Paid run (DeepSeek V4.1 Flash has no `none` reasoning effort, so use `low`):

```bash
uv run edgar run --allow-paid --model deepseek-v4.1-flash --reasoning-effort low \
  --image-model muse-spark-1.3-contributor --dataset $DATA --iterations 12
```

## Development

```bash
uv run gate
```

```bash
uv run check
```

`gate` runs format, lint, type check, tests and the lock check. `check` enforces the mirrored test layout, a `changes/` fragment per branch, and other repository hygiene. Both come from [wills-agentic-tools](https://github.com/wd7512/wills-agentic-tools) and run in CI. Coding agents: see [AGENTS.md](AGENTS.md).
