# Fit floor: recorded runs

## 2026-11-23, 264 discovery cells, commit before this file

- Command: `uv run scripts/fit_floor/run.py` (defaults: 3 curves, 264 cells, 5 starts, 300 iterations, tolerance 1e-6)
- Host: Apple M5 Pro, 15 JAX CPU devices; total 22 s

| Program | Params | Fit | Seconds | Train loss | Score | Train gap | Score gap | Cells improved >1% | Won by perturbed start |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| double-gaussian | 5 | lm-bounded | 0.35 | 29.8643 | 29.6343 |  |  |  |  |
| double-gaussian | 5 | multistart ×5 | 3.81 | 29.8618 | 29.6433 | 0.0025 | -0.0090 | 0.4% | 22.0% |
| generalised | 10 | lm-bounded | 0.45 | 29.2342 | 29.4761 |  |  |  |  |
| generalised | 10 | multistart ×5 | 5.03 | 29.1620 | 29.5265 | 0.0722 | -0.0504 | 5.3% | 10.6% |
| winner | 11 | lm-bounded | 0.45 | 29.1546 | 29.4857 |  |  |  |  |
| winner | 11 | multistart ×5 | 6.19 | 29.0804 | 29.5132 | 0.0742 | -0.0274 | 5.7% | 7.2% |

- **Training loss:** the floor is 0.0025–0.074 below `lm-bounded` (at most 0.25%). Only 0.4–5.7% of cells improve by more than 1%, so the search's fitter already finds nearly the best fit from the estimator's start.
- **Held-out score:** the floor is *worse* on all three curves (by 0.009–0.050). Fitting the training trials harder overfits them, which matches Phase 0's finding that full convergence costs held-out score.
- **Conclusion:** a faster fitter used to run longer or from more starts would not lower scores. Fitter speed only matters for how many programs a run can score per hour.
