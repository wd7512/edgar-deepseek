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
| `protocol` | The frontier's protocol: 5 cold starts, at most 100 iterations, each step capped at 0.25 × (1 + \|parameter\|). |
| `warm` | `protocol`, plus a warm start from the fitted parents through the proven maps. It can fail only on a nesting whose parent is fitted after the child (the W6/W6b equivalence), or whose map breaks outside the probed region (see below). |
| `single` | `protocol` with one start. |
| `absolute`, `absolute-warm` | The earlier protocol, whose steps were capped at 0.25 whatever the parameter's size. |
| `reference` | 32 starts, 400 iterations and warm starts. Slow; run it once to tighten the best-known errors. |

## Results (64 cells, `results/fitbench`)

| Fitter | Seconds | Violations | Mean regret | Cells at best |
| --- | ---: | ---: | ---: | ---: |
| `single` | 19.4 | 526 of 1280 | 22.6% | 34% |
| `absolute` | 66.7 | 244 of 1280 | 3.2% | 67% |
| `protocol` | 64.3 | 189 of 1280 | 2.2% | 78% |
| `absolute-warm` | 67.4 | 6 of 1280 | 0.39% | 81% |
| `warm` | 55.8 | 2 of 1280 | 0.13% | 91% |

With the absolute cap of 0.25, an amplitude could move only about 25 in 100 iterations, while real peak responses reach 87. Making the cap relative to the parameter's size cut violations by a quarter and regret by a third, without slowing the fit. The cold protocol still misses a contained equation's fit on 15% of checks, mostly on the generalised-power (GG7, GS10), Laplace-power (L8) and tanh-width (T7) equations. Warm starts through correct maps remove nearly all of that.

Under the absolute cap, `warm` had six violations. Two come from W6 being fitted before its equivalent W6b. The other four come from GG7 fitting a negative width. GG7 uses `abs(x/s)`, which ignores the sign, but L8's `(abs(x)/s1)**p` is undefined for a negative `s1`. The map is exact only for positive widths, so those four are the map's fault, not the fitter's.

## Fitting speed

Bounded LM runs in chunks of 10 iterations and drops fits that have stopped between chunks, so fits still moving no longer pay for finished ones. Results are identical: the largest parameter difference is 0. One equation on the 791 exploration neurons with 5 starts:

| Equation | Single loop | Chunks of 10 |
| --- | ---: | ---: |
| G5 (5 parameters) | 31.1 s | 12.8 s |
| SK9 (9 parameters) | 52.1 s | 22.4 s |
| L8 (8 parameters) | 39.2 s | 15.6 s |

Pruning (`prune_after`) is available but off by default. It runs every start for a few iterations, then continues only each neuron's leader. It halves the time again, but costs accuracy, mainly on cold fits and the larger equations:

| Fitter (64 cells) | Seconds | Violations | Mean regret |
| --- | ---: | ---: | ---: |
| `warm` | 59.5 | 2 | 0.21% |
| `pruned-20-warm` | 32.8 | 2 | 0.50% |
| `pruned-10-warm` | 24.9 | 4 | 0.74% |
| `protocol` | 60.8 | 189 | 2.23% |
| `pruned-20` | 30.3 | 202 | 2.66% |

The paper's model beats the double Gaussian by only about 0.5% in MSE, so the frontier keeps the unpruned protocol.

## Adding a fitter

Add a `Fitter` to `FITTERS` in `src/edgar_study/frontier/bench.py`. Its `fit(equation, xt, yt, warm)` returns one row of parameters per cell, in the equation's declared order. `warm` is `None` unless the fitter asks for warm starts. Any optimiser works (LM, Adam, scipy), as long as it uses only `xt` and `yt`.

## Binning (rejected)

The `binned-N` fitters fit to averages of N angle-sorted neighbouring training trials. Each point stands for the same number of trials, so no weights are needed. Errors are still measured on the raw trials.

| Fitter (64 cells) | Seconds | Violations | Mean regret |
| --- | ---: | ---: | ---: |
| `warm` | 59.5 | 2 | 0.45% |
| `binned-2-warm` | 53.9 | 13 | 347% |
| `binned-4-warm` | 53.8 | 26 | 9660% |
| `binned-8-warm` | 49.5 | 28 | 2.79% |

At 64 cells, compilation dominates the time, so fewer points save little. Fewer points also leave room between them for spikes, and SKQ11, SKP10 and SK9 blow up when scored on the trials.
