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
- At full size (48 proposals on 1,055 cells) fitting grows about 4× per program and the LLM stages need three rounds of 16, so on this 15-core host fitting should stay below a third of a generation. Hosts with fewer cores fit much more slowly (see the pipelining section).

## 2026-11-23, run `20261123-163928-3a269b`: per-program pipelining

Same command and settings as above, with each program's translation, estimator and fit started as soon as its own proposal returns (`--no-pipeline` turns this off). 15 of 16 programs scored; every stage made the same number of calls as the unpipelined run (16 translations, 16 estimators, 15 fits), so no work was repeated.

| | Upstream order | Pipelined |
| --- | ---: | ---: |
| Total | 196 s | 180 s |
| Slowest proposal, generation 0 / 1 | 84.6 s / 22.0 s | 93.8 s / 16.4 s |
| Last proposal back → next stage, generation 0 | 31.2 s | 23.6 s |
| Last proposal back → next stage, generation 1 | 28.2 s | 18.0 s |

- LLM latency varies from run to run, so the totals are not comparable on their own; the time after the last proposal returns is what pipelining controls. It fell by 8–10 s per generation (about 30%).
- What remains is the slowest program's own translation and estimator (they cannot start before its proposal returns) and upstream's plots.
- Correction (review of this run): an early fit is blocking work on the event loop. While it runs, requests already sent keep generating on the server, but queued requests are not sent and replies are not read. Fits therefore overlap only with in-flight requests, not with queued ones, and this run's proposal latencies include time the loop spent fitting. Early fits now wait until no request is queued, and each logged request records the blocked time as `blocked_s`.
- These shares are specific to this host and size. On a 4-core machine at full size (1,055 cells), real programs took 10–70 s to fit and fitting was about 65% of the run.
