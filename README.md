# EDGAR × OpenCode Go

This fork runs [EDGAR](https://github.com/reillytilbury/EDGAR)'s published equation-discovery loop unchanged, with OpenCode Go models in place of Gemini. The six upstream modules live unchanged in `src/edgar_upstream/` and stay byte-identical to commit `0943922`; `src/edgar_study/` swaps only the LLM call layer at runtime.

- [docs/PHASE_0.md](docs/PHASE_0.md): the like-for-like replication, its protocol, and results.
- [docs/ROADMAP.md](docs/ROADMAP.md): Phase 1 (agentic harness) and Phase 2 (custom harness) ideas.
- [METHOD_DIFFS.md](METHOD_DIFFS.md): every known difference from the paper and from upstream.

Use [uv](https://docs.astral.sh/uv/) with Python 3.12. `pyproject.toml` and `uv.lock` are authoritative. The upstream notebooks, tests and `requirements.txt` live at [commit `0943922`](https://github.com/reillytilbury/EDGAR/tree/0943922dcb73b12dbee8e440b3e1ff700cb49c10).

## Setup

```bash
uv sync --locked
```

```bash
mkdir -p data
curl --fail --location https://ndownloader.figshare.com/files/15479246 \
  --output data/gratings_drifting_GT1_2019_04_12_1.npy
```

Dataset 1 is from [Stringer et al. on Janelia Figshare](https://janelia.figshare.com/articles/dataset/8279387) (859 MB, CC BY-NC 4.0). Every command that loads it checks it against a pinned SHA-256.

Store the OpenCode Go key in a file outside the repository and point `OPENCODE_API_KEY_FILE` to it. `OPENCODE_API_KEY` also works. Request logs never contain the key.

## Offline baseline

Fits the single- and double-Gaussian seeds with the upstream JAX evaluator. Needs no key.

```bash
JAX_PLATFORMS=cpu uv run edgar baseline \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --output results/dataset1
```

`--synthetic` replaces `--dataset` for a quick check. The default `strict-holdout-v1` protocol splits the 2,110 eligible cells into 1,055 discovery, 527 model-selection and 528 locked test cells. Discovery cells are identical to upstream's training half.

| Gaussian seed | Discovery MSE | Selection MSE | Discovery penalized score | Selection penalized score |
| --- | ---: | ---: | ---: | ---: |
| Single | 2.997599 | 2.998546 | 30.015987 | 30.025467 |
| Double | 2.880491 | 2.877630 | 28.854914 | 28.826298 |

## Search runs

```bash
uv run edgar preflight --model hy4-flash-free
```

```bash
JAX_PLATFORMS=cpu uv run edgar run --model hy4-flash-free \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --iterations 12
```

For a run under 30 minutes on 4 CPU cores, search on a quarter of the discovery cells with a smaller search. `rescore` still tests on the full selection and locked-test cells:

```bash
JAX_PLATFORMS=cpu uv run edgar run --model hy4-flash-free \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy \
  --iterations 2 --islands 4 --batch-size 6 --cell-fraction 0.25
```

```bash
JAX_PLATFORMS=cpu uv run edgar rescore \
  --run results/runs/<run-id> --dataset data/gratings_drifting_GT1_2019_04_12_1.npy
```

Each run directory holds:
- `manifest.json`: models, settings, git SHAs and spend before and after.
- `api_calls.jsonl`: one row per request attempt with role, status, latency and the provider's `usage` object.
- `translations.jsonl`: NumPy → JAX translations, used by `rescore`.
- upstream's own `program_databases/` output.

`rescore` writes `rescore.json`.

### Fitting speed

Nearly all run time is per-cell curve fitting over 1,055 cells for every program. `--fitter` picks how it runs:

| `--fitter` | What it does | When |
| --- | --- | --- |
| `auto` (default) | `lm-bounded` | always |
| `lm-bounded` | Levenberg–Marquardt with each parameter's step capped at 0.25, stopping each cell at a 1e-4 relative tolerance (at most 100 iterations). About 8× faster than `upstream` over the paper's curves and upstream's seeds, scores within 0.7% | default |
| `upstream` | upstream's Python loop, unchanged: 1,000 Adam steps | exact reproduction |
| `adam` | the same Adam steps compiled into one XLA loop, cells split across CPU cores (same scores to about 1e-4, about 1.9× faster) | reproduction on CPU |
| `lm` | Levenberg–Marquardt without the step cap: can strand parameters past a program's clips | experiments only |

`src/edgar_study/__init__.py` exposes one XLA CPU device per core so cells can be split across cores. `rescore` uses the run's own fitter unless `--fitter` is given.

### Spending safeguards

- Only models that the Go model list prices at zero (for example `hy4-flash-free`) can be called unless you pass `--allow-paid`. The price list is fetched at the start of every run.
- Every request's list-equivalent cost is appended to `results/spend.jsonl`. New requests are refused once that ledger reaches $9.50.

The paid run is the same command with the models swapped. DeepSeek V4.1 Flash has no `none` reasoning effort, so use `low` for its routine tiers:

```bash
JAX_PLATFORMS=cpu uv run edgar run --allow-paid \
  --model deepseek-v4.1-flash --reasoning-effort low --image-model muse-spark-1.3-contributor \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy --iterations 12
```

## Development

Tests, lint, types and repository hygiene run through [wills-agentic-tools](https://github.com/wd7512/wills-agentic-tools), pinned in `pyproject.toml`. CI runs the same two commands.

```bash
uv run gate
```

```bash
uv run check
```

`gate` runs format, lint, type check, tests and the lock check. `check` enforces the mirrored test layout (`src/edgar_study/x.py` needs `tests/unit/test_x.py`), file-name portability, secrets, untracked paths and a `changes/` fragment per branch. The six upstream modules in `src/edgar_upstream/` are excluded from format, lint and type checks because they must stay byte-identical.

Coding agents: see [AGENTS.md](AGENTS.md). `uv run delegate` starts a worker in its own worktree using the `[tool.devtools.delegate]` settings in `pyproject.toml`.
