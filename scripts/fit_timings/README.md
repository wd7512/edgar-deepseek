# Fit timings

Times each curve in `edgar_study.curves` under each fitter through upstream's `objective`. Not part of gate or CI.

```bash
uv run scripts/fit_timings/run.py --size small
```

`small` is 16 cells × 144 trials, 100 steps (about a minute). `real` is 1,055 cells × 428 trials, 1,000 steps, like a search. Run with `--help` for overrides.

Each call is split into estimator, compile and checks + fit time. Every program pays compilation again, so it counts as cost. `lm` ignores the step count and runs `--lm-iterations` (default 100).

Paste the printed section at the top of `results.md` with a line on what it shows.
