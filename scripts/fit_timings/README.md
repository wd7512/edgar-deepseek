# Fit timings

About 90% of an EDGAR run is spent fitting proposed programs. `run.py` times that path so changes to
it can be measured. It is a script, not a test, so gate and CI never run it.

For each fitter (`upstream`, `adam`, `lm`) and each of the ten curves in `edgar_study.curves`, it
calls `hypothesis_engine.objective` on synthetic cells drawn from the paper's winning equation (H27),
the same estimate, check, fit and score path a search pays for every program.

```bash
uv run scripts/fit_timings/run.py --size small
```

```bash
uv run scripts/fit_timings/run.py --size real
```

`small` is 16 cells × 144 trials with 100 Adam steps, a quick check that takes about a minute.
`real` matches a search: 1,055 cells × 428 trials with 1,000 Adam steps, and takes much longer.
`--cells`, `--trials`, `--steps`, `--lm-iterations`, `--fitters`, `--curves` and `--repeats`
override the preset. Progress is logged to stderr; the Markdown section printed to stdout is what
goes into `results.md`.

## Columns

- **Iterations**: Adam steps for `upstream` and `adam`; Levenberg–Marquardt iterations for `lm`,
  which ignores the step count. Production uses 1,000 Adam steps and 100 LM iterations.
- **Total**: wall time of one `objective` call (median over `--repeats`).
- **Estimator**: time in the curve's numpy parameter estimator, summed over cells.
- **Compile**: time JAX reports spending on tracing, lowering and XLA compilation during the call.
  Every fitter builds a new jitted function per call, so a search pays this for every program; it
  is part of the cost, not warm-up.
- **Checks + fit**: the rest: upstream's validity checks, the fit itself and scoring.
- **Train loss** and **Score**: upstream's `10 × MSE` on the training half, and its penalised
  held-out score. These show whether a faster fitter still fits as well.

## Caveats

- The curves' estimator is one shared binned-curve estimator, not each paper program's own, so
  estimator time will not match LLM-written estimators.
- The curves clip amplitudes, widths, exponents and κ, which flattens gradients outside those
  ranges and can change convergence compared with unclipped LLM-written programs.
- Timings depend on the host. Compare rows from the same run, and record the header with them.

## Recording results

Add each run to the top of `results.md`: the printed section as is, then a few lines on what it
shows. Keep full logs out of the repo (`results/` is untracked).
