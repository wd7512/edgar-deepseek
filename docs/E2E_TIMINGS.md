# End-to-end stage timings

## 2026-11-23, run `20261123-162634-1ed198`

- Command: `uv run edgar run --model hy4-flash-free --dataset $DATA --iterations 2 --islands 2 --batch-size 4 --cell-fraction 0.25`
- 2 generations × 8 proposals on 264 discovery cells, `lm-bounded` fitter, Apple M5 Pro (15 JAX CPU devices). 13 of 16 programs scored.

| Stage | Calls | Busy (s) | Wall (s) | Share of 196 s |
| --- | ---: | ---: | ---: | ---: |
| LLM proposals | 16 | 687.2 | 107.3 | 55% |
| LLM estimators | 16 | 120.6 | 19.9 | 10% |
| LLM translations | 16 | 80.2 | 13.5 | 7% |
| Plots | 49 | 23.8 | 23.8 | 12% |
| Fit (search) | 15 | 16.1 | 16.1 | 8% |
| Final refit | 24 | 15.8 | 15.8 | 8% |
| Evaluation matrix | 15 | 1.0 | 1.0 | 1% |
| Population updates | 11 | 0.1 | 0.1 | 0% |
| Other | | | 3.7 | 2% |

- The three LLM stages take 72% of the run; fitting and the final refit together 16%; upstream's plots 12%.
- Each fit takes about 1.1 s on 264 cells and each final refit about 0.7 s. All 8 proposals of a generation ran at once, so proposal wall time is set by the slowest call: about 54 s per generation against a 43 s mean call.
- At full size (48 proposals on 1,055 cells) fitting grows about 4× per program and the LLM stages need three rounds of 16, so fitting should stay below a third of a generation.
