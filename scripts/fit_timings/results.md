# Fit timings: recorded runs

Newest first. See `README.md` for what each column means.

## 2026-11-23, `real`, commit `f7ae000`

- Command: `uv run scripts/fit_timings/run.py --size real`
- Data: 1055 synthetic cells × 428 trials, drawn from the winner (H27)
- Adam steps: 1000; LM iterations: 100; median of 1 call(s)
- Host: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 cores, 4 JAX cpu device(s), JAX 0.6.0

| Program | Paper ID | Params | Fitter | Iterations | Total (s) | Estimator (s) | Compile (s) | Checks + fit (s) | Train loss | Score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-gaussian | S1 | 4 | upstream | 1000 | 10.29 | 0.08 | 1.21 | 9.01 | 27.5486 | 28.3705 |
| double-gaussian | S2 | 5 | upstream | 1000 | 18.86 | 0.08 | 0.77 | 18.01 | 21.1730 | 22.3159 |
| asym-widths | A9 | 8 | upstream | 1000 | 19.28 | 0.09 | 0.95 | 18.25 | 17.6953 | 19.2784 |
| asym-shared-widths | A34 | 6 | upstream | 1000 | 19.20 | 0.09 | 0.78 | 18.33 | 20.5472 | 21.9223 |
| free-peaks | A35 | 7 | upstream | 1000 | 18.38 | 0.10 | 0.77 | 17.52 | 17.9783 | 19.3098 |
| generalised | B35 | 10 | upstream | 1000 | 22.55 | 0.12 | 1.02 | 21.41 | 17.1947 | 19.0478 |
| linear-term | B12 | 10 | upstream | 1000 | 18.13 | 0.10 | 0.51 | 17.53 | 17.1074 | 18.8067 |
| winner | H27 | 11 | upstream | 1000 | 23.21 | 0.10 | 0.92 | 22.19 | 16.7513 | 18.6607 |
| von-mises | seed 3 | 4 | upstream | 1000 | 6.18 | 0.11 | 0.32 | 5.75 | 27.8526 | 28.6270 |
| double-von-mises | seed 4 | 6 | upstream | 1000 | 10.19 | 0.12 | 0.35 | 9.72 | 20.8002 | 21.9475 |
| single-gaussian | S1 | 4 | adam | 1000 | 5.61 | 0.13 | 0.71 | 4.77 | 27.5486 | 28.3705 |
| double-gaussian | S2 | 5 | adam | 1000 | 10.17 | 0.09 | 0.57 | 9.51 | 21.1730 | 22.3159 |
| asym-widths | A9 | 8 | adam | 1000 | 10.70 | 0.10 | 0.71 | 9.89 | 17.6953 | 19.2784 |
| asym-shared-widths | A34 | 6 | adam | 1000 | 9.97 | 0.11 | 0.57 | 9.30 | 20.5472 | 21.9222 |
| free-peaks | A35 | 7 | adam | 1000 | 10.69 | 0.12 | 0.67 | 9.91 | 17.9783 | 19.3098 |
| generalised | B35 | 10 | adam | 1000 | 13.69 | 0.11 | 0.64 | 12.94 | 17.1946 | 19.0477 |
| linear-term | B12 | 10 | adam | 1000 | 10.22 | 0.08 | 0.52 | 9.61 | 17.1074 | 18.8067 |
| winner | H27 | 11 | adam | 1000 | 13.28 | 0.12 | 0.67 | 12.49 | 16.7528 | 18.6616 |
| von-mises | seed 3 | 4 | adam | 1000 | 3.29 | 0.09 | 0.42 | 2.79 | 27.8526 | 28.6270 |
| double-von-mises | seed 4 | 6 | adam | 1000 | 5.77 | 0.09 | 0.40 | 5.29 | 20.8002 | 21.9475 |
| single-gaussian | S1 | 4 | lm | 100 | 1.66 | 0.12 | 0.54 | 1.00 | 27.4906 | 28.3335 |
| double-gaussian | S2 | 5 | lm | 100 | 3.06 | 0.08 | 0.63 | 2.35 | 21.1492 | 22.3004 |
| asym-widths | A9 | 8 | lm | 100 | 3.27 | 0.12 | 0.70 | 2.46 | 17.9009 | 19.4851 |
| asym-shared-widths | A34 | 6 | lm | 100 | 2.99 | 0.08 | 0.75 | 2.16 | 20.5825 | 21.9566 |
| free-peaks | A35 | 7 | lm | 100 | 3.16 | 0.11 | 0.70 | 2.34 | 18.2405 | 19.5864 |
| generalised | B35 | 10 | lm | 100 | 4.07 | 0.12 | 0.87 | 3.08 | 17.7349 | 19.6646 |
| linear-term | B12 | 10 | lm | 100 | 3.60 | 0.11 | 0.60 | 2.90 | 17.4025 | 19.2134 |
| winner | H27 | 11 | lm | 100 | 4.22 | 0.09 | 0.88 | 3.26 | 17.4564 | 19.4415 |
| von-mises | seed 3 | 4 | lm | 100 | 0.88 | 0.08 | 0.32 | 0.48 | 27.5276 | 28.3584 |
| double-von-mises | seed 4 | 6 | lm | 100 | 1.52 | 0.09 | 0.38 | 1.06 | 20.6314 | 21.8904 |
| single-gaussian | S1 | 4 | lm-bounded | 35 (1055/1055 converged) | 1.15 | 0.13 | 0.49 | 0.54 | 27.4906 | 28.3328 |
| double-gaussian | S2 | 5 | lm-bounded | 26 (1055/1055 converged) | 1.39 | 0.12 | 0.48 | 0.79 | 21.1492 | 22.2972 |
| asym-widths | A9 | 8 | lm-bounded | 39 (1055/1055 converged) | 1.80 | 0.10 | 0.55 | 1.16 | 17.6612 | 19.2658 |
| asym-shared-widths | A34 | 6 | lm-bounded | 48 (1055/1055 converged) | 2.13 | 0.12 | 0.64 | 1.38 | 20.5745 | 21.9498 |
| free-peaks | A35 | 7 | lm-bounded | 41 (1055/1055 converged) | 1.81 | 0.11 | 0.57 | 1.13 | 17.9502 | 19.3066 |
| generalised | B35 | 10 | lm-bounded | 69 (1055/1055 converged) | 3.49 | 0.09 | 0.69 | 2.71 | 17.1597 | 19.1946 |
| linear-term | B12 | 10 | lm-bounded | 100 (1050/1055 converged) | 4.03 | 0.11 | 0.59 | 3.33 | 17.0375 | 18.7917 |
| winner | H27 | 11 | lm-bounded | 67 (1055/1055 converged) | 3.42 | 0.08 | 0.73 | 2.61 | 16.6403 | 18.7244 |
| von-mises | seed 3 | 4 | lm-bounded | 81 (1055/1055 converged) | 1.00 | 0.09 | 0.41 | 0.50 | 27.5277 | 28.3584 |
| double-von-mises | seed 4 | 6 | lm-bounded | 100 (1045/1055 converged) | 1.67 | 0.09 | 0.43 | 1.16 | 20.5229 | 21.7892 |

`lm-bounded` against upstream's loop, per program:

- **Time:** 4.5–13.6× faster (1.0–4.0 s against 6.2–23.2 s), 2.5–7.3× faster than compiled `adam`. Every cell stops on the 1e-4 tolerance except 5 of 1,055 for linear-term and 10 for double-von-mises, which hit the 100-iteration cap.
- **Fit:** lower training loss on nine of ten curves (asym-shared-widths is 0.03 higher).
- **Score:** within 0.8% (largest gap +0.15 on generalised; winner +0.06). The top five keep upstream's order. Double-von-mises and asym-shared-widths, 0.03 apart under upstream, swap.
- Plain `lm` is as fast but scores up to 3.4% worse (generalised 19.66 against 19.05), because unbounded steps strand parameters past the programs' clips.

Compilation (0.5–0.7 s) is now 20–40% of an `lm-bounded` call, so it is the next cost to cut.

## 2026-11-23, `real`, commit `cc709c0`

- Command: `uv run scripts/fit_timings/run.py --size real --curves double-gaussian winner`
- Data: 1055 synthetic cells × 428 trials, drawn from the winner (H27)
- Adam steps: 1000; LM iterations: 100; median of 1 call(s)
- Host: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 cores, 4 JAX cpu device(s), JAX 0.6.0

| Program | Paper ID | Params | Fitter | Iterations | Total (s) | Estimator (s) | Compile (s) | Checks + fit (s) | Train loss | Score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| double-gaussian | S2 | 5 | upstream | 1000 | 21.31 | 0.08 | 1.42 | 19.82 | 21.1730 | 22.3159 |
| winner | H27 | 11 | upstream | 1000 | 21.70 | 0.09 | 0.94 | 20.67 | 16.7513 | 18.6607 |
| double-gaussian | S2 | 5 | adam | 1000 | 9.53 | 0.08 | 0.90 | 8.55 | 21.1730 | 22.3159 |
| winner | H27 | 11 | adam | 1000 | 12.19 | 0.09 | 0.57 | 11.54 | 16.7528 | 18.6616 |
| double-gaussian | S2 | 5 | lm | 100 | 2.77 | 0.08 | 0.69 | 2.00 | 21.1492 | 22.3004 |
| winner | H27 | 11 | lm | 100 | 3.65 | 0.12 | 0.64 | 2.89 | 17.4564 | 19.4415 |

At search size the fit dominates: compilation is under 1.5 s and the estimator under 0.15 s, against 20 s of fitting under upstream's loop. Compiled `adam` gives the same losses 1.8–2.2× faster. `lm` is 3.3–3.4× faster than `adam` and 5.9–7.7× faster than upstream. It fits the double Gaussian slightly better, but the winner worse (train 17.46 against 16.75) at 100 iterations, so its iteration count needs checking before it is used for speed. Only the baseline and the winner were run; the full ten-curve real run is still to do.

## 2026-11-23, `small`, commit `cc709c0`

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
