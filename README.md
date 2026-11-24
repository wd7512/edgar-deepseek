# EDGAR × OpenCode Go

[EDGAR](https://github.com/reillytilbury/EDGAR)'s equation-discovery loop, unchanged, with OpenCode Go models instead of Gemini. Results: [docs/PHASE_0.md](docs/PHASE_0.md). Differences from the paper: [METHOD_DIFFS.md](METHOD_DIFFS.md).

## Setup

```bash
uv sync --locked
```

```bash
mkdir -p data && curl -fL https://ndownloader.figshare.com/files/15479246 -o data/gratings_drifting_GT1_2019_04_12_1.npy
```

Set `OPENCODE_API_KEY_FILE` to a file holding your OpenCode Go key.

## Run

```bash
export JAX_PLATFORMS=cpu DATA=data/gratings_drifting_GT1_2019_04_12_1.npy
```

```bash
uv run edgar run --model hy4-flash-free --dataset $DATA
```

```bash
uv run edgar rescore --run results/runs/<run-id> --dataset $DATA
```

`run` searches the discovery cells; `rescore` picks a winner on the selection cells and scores it on the locked test cells. Add `--iterations 2 --islands 4 --cell-fraction 0.25` for a 30-minute run. Paid models need `--allow-paid`; spend stops at $9.50.

## NEAT baseline

```bash
uv run edgar neat --dataset $DATA --population 96 --generations 6 --cell-fraction 0.25
```

Evolves feed-forward tuning-curve networks whose weights are fitted per cell and scored by the same objective as EDGAR's programs. The top genomes then go through the usual selection and locked-cell holdout. Runs are written to `results/neat/<run-id>/` with the winner's source. Results for the 25% proof-of-concept run are in [docs/NEAT.md](docs/NEAT.md).

## Develop

```bash
uv run gate
```

```bash
uv run check
```

See [AGENTS.md](AGENTS.md).
