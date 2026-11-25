# Fitting benchmark

We never know the true best fit of an equation to a neuron, so a fitter can't be scored against ground truth. Nesting gives a check that needs none.

**The rule.** Suppose fixing some of a child equation's parameters gives its parent exactly. Then the child's best training error is at most the parent's on every neuron. A fitted child that does worse than its fitted parent shows the fitter missed the child's optimum, and the gap is a lower bound on the miss. The rule covers training error only. It says nothing about held-out error or the penalised score.

## What it measures

`src/edgar_study/frontier/nested.json` holds 16 equations taken verbatim from the frontier runs (run-20 and run-cold-10), with their declared starts, and 20 nestings between them. Each nesting gives every child parameter as a parent parameter, a number, or `[name, scale, shift]`. For example, generalised power ⊃ double Gaussian with `s → √2·s` and `p = 2`. Loading the file checks every map against the parent's predictions, and a map that fails is an error. Two equivalent rewrites are linked both ways, so their errors must match.

For each fitter, on the cohort's training trials:

- **Violations:** cells where a child's error exceeds its parent's by more than 0.1%, per nesting and in total.
- **Regret:** each error relative to the best known for that equation and neuron. The best known is the lowest error any saved fitter reached for that equation, or for any of its ancestors through the nestings.
- **Seconds:** wall time per equation, compilation included.

Fitters receive training trials only. The benchmark computes every error itself from the returned parameters.

## Running it

```
uv run edgar fitbench --cells 64 --fitters protocol,warm
```

The first run reads the dataset and caches the first 64 exploration cells in `results/fitbench/<cohort>/cohort.npz`. Each fitter's errors are saved beside the cache. The report combines every fitter saved for that cohort, so trying a new fitter means fitting only that one. Pass `--fitters ""` to print the report without fitting.

The built-in fitters:

| Fitter | What it is |
| --- | --- |
| `protocol` | The frontier's protocol: 5 cold starts, at most 100 iterations. |
| `warm` | `protocol`, plus a warm start from the fitted parents through the proven maps. It can fail only on a nesting whose parent is fitted after the child (the W6/W6b equivalence), or whose map breaks outside the probed region (see below). |
| `single` | One start only. |
| `reference` | 32 starts, 400 iterations, relative step cap and warm starts. Slow; run it once to tighten the best-known errors. |

## First results (64 cells, `results/fitbench`)

| Fitter | Seconds | Violations | Mean regret | Cells at best |
| --- | ---: | ---: | ---: | ---: |
| `single` | 22.5 | 531 of 1280 | 23.9% | 37% |
| `protocol` | 66.7 | 244 of 1280 | 2.9% | 77% |
| `warm` | 67.4 | 6 of 1280 | 0.06% | 97% |

One equation takes 4.2 ± 1.0 s under `protocol` (16 equations; range 3.2 to 6.4 s). The cold protocol misses a contained equation's fit on 19% of checks. It is worst on the tanh-width (T7), generalised-power (GG7, GS10) and Laplace-power (L8) equations.

Two of `warm`'s six violations come from W6 being fitted before its equivalent W6b. The other four come from GG7 fitting a negative width. GG7 uses `abs(x/s)`, which ignores the sign, but L8's `(abs(x)/s1)**p` is undefined for a negative `s1`. The map is exact only for positive widths, so those four are the map's fault, not the fitter's.

## Adding a fitter

Add a `Fitter` to `FITTERS` in `src/edgar_study/frontier/bench.py`. Its `fit(equation, xt, yt, warm)` returns one row of parameters per cell, in the equation's declared order. `warm` is `None` unless the fitter asks for warm starts. Any optimiser works (LM, Adam, scipy), as long as it uses only `xt` and `yt`.
