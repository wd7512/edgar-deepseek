# Runtime statistics: rank-bias search and comparison runs

Measured on one cloud container: 4 CPU cores, 15 GB RAM, JAX on CPU, `hy4-flash-free`. Wall times are real elapsed time. Where a time was inferred from file timestamps rather than timed directly, the table says so.

## Search run (`edgar frontier --rank-bias 1 --seed 0`, planned 1000 batches, stopped at 156)

| statistic | value |
| --- | ---: |
| batches completed | 156 of 1000 (stopped by the user) |
| wall time | about 6.8 h (first model call to last write, 02:07 to 08:54) |
| wall time per batch | about 2.6 min |
| batches that asked for a new idea / a revisit | 118 / 38 |
| new-idea modes | single_idea 50, combination 40, anomaly_detour 20, independent_root 8 |
| equations proposed | 950 |
| fitted | 471 (49.6%) |
| invalid | 417 (43.9%) |
| duplicate | 53 (5.6%) |
| fit failed | 9 (0.9%) |
| fitted equations per hour | about 69 |
| output size | `wave.json` 5.1 MB, `api_calls.jsonl` 360 KB |

Best exploration score over the run (lower is better; baseline double Gaussian 28.5916): 28.4914 at batch 1, 28.4865 at batch 125, 28.4863 at batch 148.

### Model calls

| statistic | value |
| --- | ---: |
| calls | 470 (about 3.0 per batch) |
| HTTP status | all 200; every call succeeded on the first attempt |
| time blocked on rate limits | 0 s |
| latency, mean / median / p90 / max | 9.4 s / 9.2 s / 13.2 s / 21.5 s |
| summed latency | 4,419 s (73.7 min) |
| images per call, mean | 28.1 |
| prompt tokens | 5,259,204 |
| completion tokens | 605,678 |
| cost | $0 (free model) |

Summed latency is 18% of wall time. Calls may overlap, so this is an upper bound on the share spent waiting for the model. At least 82% of wall time went to fitting equations, drawing images and the scheduler. An invalid equation is rejected before any fitting, so the 44% invalid rate wastes model calls, not fitting time.

## Comparison runs (`edgar compare`, bounded Levenberg-Marquardt, 5 starts, 3 candidates plus 2 controls)

| run | cells | wall time |
| --- | ---: | ---: |
| common-start, confirmation cohort | 264 | under 90 s |
| common-start, exploration cohort | 791 | about 2.6 min (from the previous run's finish) |
| native, confirmation cohort | 264 | not timed |
| native, exploration cohort, top 1 candidate | 791 | not timed |

These include loading the dataset and importing the upstream modules. The fits themselves are fast because every cell and start is fitted in one batch.

## Population-geometry check (PCA and rcvPCA spectra, 1,055 discovery cells)

Per-model fit times, with the upstream objective and 2,000 iterations: double Gaussian 15 s, paper equation 21 s, idea-1 18 s, idea-148 17 s, idea-125 22 s.

## Adam comparison runs (`--fitter adam --max-iter 10000`)

At the time of writing, the first of four runs (common-start, confirmation cohort, 264 cells) had run for 22 minutes without finishing. Adam with up to 10,000 steps is far slower than bounded Levenberg-Marquardt on this machine, and the other three runs (a native confirmation run and two exploration runs of 791 cells) were still queued behind it. No Adam timings or results are included here.

## Tests

| run | result | time |
| --- | --- | ---: |
| `tests/unit/frontier/test_compare.py` | 14 passed | about 25 s |
| compare and protocol tests together (before the Adam change) | 46 passed | 140 s |
| full unit suite on a shallow clone, before the upstream commit was fetched | 292 passed, 35 failed (all from the missing upstream commit) | 208 s |

The full `uv run gate` has not been run on the final branch. The pull request's CI is the first complete run.

## Not measured

- Per-batch breakdown of fitting, image rendering and model waiting.
- Peak memory.
- Timing of the `native` runs and the Adam runs.
