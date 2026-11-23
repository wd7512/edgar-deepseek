# Fit timings: recorded runs

Newest first. See `README.md` for what each column means.

## 2026-11-23, `small`, commit `f5f7a25`

- Command: `uv run scripts/fit_timings/run.py --size small --repeats 3`
- Data: 16 synthetic cells × 144 trials, drawn from the winner (H27)
- Adam steps: 100; LM iterations: 100; median of 3 call(s)
- Host: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 cores, 4 JAX cpu device(s), JAX 0.6.0

| Program | Paper ID | Params | Fitter | Iterations | Total (s) | Estimator (s) | Compile (s) | Checks + fit (s) | Train loss | Score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-gaussian | S1 | 4 | upstream | 100 | 0.31 | 0.00 | 0.21 | 0.10 | 90.2249 | 104.6181 |
| double-gaussian | S2 | 5 | upstream | 100 | 0.57 | 0.00 | 0.45 | 0.12 | 57.4470 | 71.2577 |
| asym-widths | A9 | 8 | upstream | 100 | 0.53 | 0.00 | 0.37 | 0.15 | 50.4017 | 64.5318 |
| asym-shared-widths | A34 | 6 | upstream | 100 | 0.48 | 0.00 | 0.35 | 0.13 | 56.7428 | 70.2440 |
| free-peaks | A35 | 7 | upstream | 100 | 0.45 | 0.00 | 0.33 | 0.12 | 51.2588 | 66.1965 |
| generalised | B35 | 10 | upstream | 100 | 0.52 | 0.00 | 0.40 | 0.14 | 49.3854 | 63.5498 |
| linear-term | B12 | 10 | upstream | 100 | 0.48 | 0.00 | 0.36 | 0.14 | 48.7939 | 63.0648 |
| winner | H27 | 11 | upstream | 100 | 0.48 | 0.00 | 0.35 | 0.13 | 48.1286 | 63.7525 |
| von-mises | seed 3 | 4 | upstream | 100 | 0.22 | 0.00 | 0.16 | 0.07 | 91.4624 | 104.2919 |
| double-von-mises | seed 4 | 6 | upstream | 100 | 0.29 | 0.00 | 0.22 | 0.08 | 59.8622 | 70.5410 |
| single-gaussian | S1 | 4 | adam | 100 | 0.50 | 0.00 | 0.41 | 0.08 | 90.2249 | 104.6181 |
| double-gaussian | S2 | 5 | adam | 100 | 0.53 | 0.00 | 0.43 | 0.09 | 57.4470 | 71.2577 |
| asym-widths | A9 | 8 | adam | 100 | 0.51 | 0.00 | 0.41 | 0.10 | 50.4017 | 64.5318 |
| asym-shared-widths | A34 | 6 | adam | 100 | 0.51 | 0.00 | 0.41 | 0.09 | 56.7428 | 70.2440 |
| free-peaks | A35 | 7 | adam | 100 | 0.52 | 0.00 | 0.41 | 0.11 | 51.2588 | 66.1965 |
| generalised | B35 | 10 | adam | 100 | 0.62 | 0.00 | 0.49 | 0.13 | 49.3854 | 63.5498 |
| linear-term | B12 | 10 | adam | 100 | 0.55 | 0.00 | 0.44 | 0.10 | 48.7939 | 63.0648 |
| winner | H27 | 11 | adam | 100 | 0.62 | 0.00 | 0.51 | 0.11 | 48.1286 | 63.7525 |
| von-mises | seed 3 | 4 | adam | 100 | 0.28 | 0.00 | 0.22 | 0.06 | 91.4624 | 104.2919 |
| double-von-mises | seed 4 | 6 | adam | 100 | 0.37 | 0.00 | 0.29 | 0.08 | 59.8622 | 70.5410 |
| single-gaussian | S1 | 4 | lm | 100 | 0.41 | 0.00 | 0.33 | 0.06 | 63.3838 | 76.6950 |
| double-gaussian | S2 | 5 | lm | 100 | 0.46 | 0.00 | 0.37 | 0.08 | 50.8700 | 66.4071 |
| asym-widths | A9 | 8 | lm | 100 | 0.56 | 0.00 | 0.46 | 0.10 | 43.0971 | 62.1460 |
| asym-shared-widths | A34 | 6 | lm | 100 | 0.55 | 0.00 | 0.44 | 0.08 | 49.9740 | 66.2803 |
| free-peaks | A35 | 7 | lm | 100 | 0.47 | 0.00 | 0.38 | 0.09 | 43.6030 | 61.2808 |
| generalised | B35 | 10 | lm | 100 | 0.55 | 0.00 | 0.46 | 0.09 | 45.8450 | 63.7980 |
| linear-term | B12 | 10 | lm | 100 | 0.55 | 0.00 | 0.45 | 0.11 | 41.8306 | 62.0271 |
| winner | H27 | 11 | lm | 100 | 0.66 | 0.00 | 0.58 | 0.10 | 44.0972 | 69.0024 |
| von-mises | seed 3 | 4 | lm | 100 | 0.31 | 0.00 | 0.27 | 0.05 | 63.4889 | 76.7396 |
| double-von-mises | seed 4 | 6 | lm | 100 | 0.38 | 0.00 | 0.32 | 0.07 | 49.2535 | 66.6623 |


At this size almost all of every call is JAX compilation (70–85%), which every fitter pays again for each program because each call builds a new jitted function. The fit itself is 0.05–0.15 s, so this size cannot rank the fitters by fitting speed; use `--size real` for that. `adam` and `upstream` give identical losses, as expected. `lm` runs to convergence: lower training loss and a better score here on 100-step Adam budgets, but see `docs/PHASE_0.md` on how that changes held-out scoring at full size.
