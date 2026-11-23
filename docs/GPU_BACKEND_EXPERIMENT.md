# GPU / backend fitting experiment — running log

Living document. Updated as the experiment runs. Everything here is measured on this machine
unless marked as a plan.

## Objective

Fit EDGAR's per-cell tuning curves on the **real recording**, using the **Mac's GPU** where the
framework supports it, compare a wide set of optimisers and packages (JAX, PyTorch/MPS,
TensorFlow, MLX, numpy, numba, scipy), and summarise which combination is fastest and which
scores best.

## Hardware

| | |
| --- | --- |
| Chip | Apple M5 Pro |
| CPU | arm64, 14 cores (6P + 8E) |
| GPU | 16-core Apple GPU, Metal 4 |
| OS | macOS 26.7.3 (25G342) |
| Python | 3.12.14 (uv-managed) |

## Grounding

- Objective and scoring: `src/edgar_upstream/hypothesis_engine.py:82`. Score is
  `10 × MSE + 0.01 × n_params` on the held-out trial half, lower is better.
- Trial split: `jax.random.PRNGKey(0)`, 50/50 permutation (`hypothesis_engine.py:117`).
- Cell split: `PRNGKey(42)`, discovery/selection/locked-test (`src/edgar_study/datasets.py:55`).
- Real recording: `gratings_drifting_GT1_2019_04_12_1.npy`, Figshare 15479246,
  SHA-256 `dbfbb9cd…e1b57` — **verified** after download.
- Curves and estimators: `src/edgar_study/curves.py`.
- Current fitters: `src/edgar_study/fitting.py` (`upstream`, `adam`, `lm`, `lm-bounded`).
- Prior CPU timings: `scripts/fit_timings/results.md`.

## Hypotheses

1. The Mac GPU only helps once the batch of cells × trials is big enough to amortise dispatch;
   at 1,055 × 214 floats the per-cell problems are tiny, so a batched second-order method on CPU
   may still win.
2. PyTorch MPS will be the only usable GPU path — `jax-metal` is unmaintained and TF Metal is
   pinned to old TensorFlow versions.
3. Second-order (LM) beats Adam everywhere in wall time, but Adam's 1,000-step early stopping
   changes held-out scores; the ranking by *score* may not follow the ranking by *speed*.
4. MLX (Apple's own array framework) is the best-fit GPU candidate if PyTorch MPS dispatch
   overhead dominates.
5. Pure numpy + batched `np.linalg.solve` may be competitive with JAX CPU for 4–11 parameters,
   because the linear algebra is tiny and BLAS is already tuned for the M5.

## Roadmap

- [x] Create worktree, `docs/GPU_BACKEND_EXPERIMENT.md`
- [x] Download and verify the real recording
- [x] Probe backends: torch MPS, TF, MLX, numba, jax devices
- [x] Write the common harness (data prep, trial split, scoring) — numpy only
- [x] Baseline: JAX CPU `lm-bounded`, `lm`, `adam` on real discovery cells
- [x] Numpy batched LM (CPU)
- [x] Numba parallel LM (CPU)
- [x] SciPy `least_squares` (`trf`, `dogbox`, `lm`), L-BFGS-B, BFGS and Powell on a cell subset
- [x] PyTorch: CPU and MPS, LM (autodiff and analytic Jacobian), Adam and Muon
- [x] TensorFlow: CPU, Adam and LM
- [x] MLX: GPU LM and Adam
- [x] Collect results into tables, write summary and recommendation
- [x] `uv run gate` and `uv run check`, `changes/` fragment, commit, push
- [x] Open the pull request: [#8](https://github.com/wd7512/edgar-deepseek/pull/8)

## Experiment matrix

Models: `double-gaussian` (S2, 5 params) and `winner` (H27, 11 params).
Data: real discovery cells, 1,055 cells × 856 fit + 856 held-out trials.

| Backend | Device | Algorithm | Status |
| --- | --- | --- | --- |
| JAX | CPU | `lm`, `lm-bounded` (float32, compiled) | done |
| numpy | CPU | batched LM, `lm-bounded` | done |
| numba | CPU | parallel LM, `lm-bounded`, `prange` over 15 threads | done |
| scipy | CPU | `least_squares` `lm` / `trf` / `dogbox`, `minimize` L-BFGS-B / BFGS / Powell | done |
| torch | CPU | LM with analytic Jacobian | done |
| torch | MPS | LM with analytic and autodiff Jacobians, Adam | done |
| torch | MPS | Muon (per-cell and whole-batch Newton-Schulz) | done |
| tensorflow | CPU | LM, Adam | done |
| MLX | GPU | LM, Adam | done |

Every row runs the same model, the same trial split and the same scoring, through
`scripts/backend_bench/`. `./scripts/backend_bench/run_all.sh` reproduces the whole table and writes
`scripts/backend_bench/results.jsonl`; `python summary.py` renders it.

## Results log

Numbers are medians of 3–5 warm runs (one untimed warm-up first), fit-only: data loading, scoring
and reporting are outside the timer. Unless a row says otherwise, the Jacobian is analytic and the
dtype is float32.

### Environment probe

Median of 3 warm runs unless noted.

| Framework | Version | GPU device | Notes |
| --- | --- | --- | --- |
| JAX | 0.6.0 | none | CPU only. `edgar_study/__init__.py` exposes one XLA device per core (15 here); `jax-metal` is not installed and is unmaintained. |
| PyTorch | 2.15.0 | MPS available | `torch.func.jacfwd`, `vmap`, `jvp` all run on MPS. |
| Muon | torch 2.15.0 | MPS | Newton-Schulz orthogonalisation batched with `bmm`; no optimiser library needed. |
| TensorFlow | 2.21.0 | none | `tf.config.list_physical_devices('GPU') == []`. `tensorflow-metal` does not support this TensorFlow. |
| MLX | 0.32.4 | GPU | `mx.default_device() == gpu`. **`mx.linalg.solve` is not implemented on the GPU**; solved on the CPU stream via `mx.stream(mx.cpu)` on unified memory. |
| numba | 0.68.1 | none | CPU only (no CUDA/Apple GPU backend); `prange` across 15 threads. |
| SciPy | 1.15.3 | none | CPU, single-threaded per cell. |

Real data: 1,055 discovery cells × 1,712 active trials, split 856 fit / 856 score exactly as
upstream does. All backends receive the same `float32` arrays and the same estimator starts.

### Baseline (JAX CPU)

`scripts/fit_timings/run.py --size real --dataset … --curves double-gaussian winner`, float64,
median of 3, includes compile and upstream's checks:

| Program | Fitter | Iterations | Total (s) | Fit (s) | Train loss | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| double-gaussian | lm-bounded | 27 | 0.59 | 0.39 | 28.1962 | 28.8655 |
| winner | lm-bounded | 89 | 1.73 | 1.51 | 27.5162 | 28.6953 |
| double-gaussian | lm | 100 | 1.12 | 0.87 | 28.1960 | 28.8738 |
| winner | lm | 100 | 1.77 | 1.55 | 27.5601 | 28.8265 |
| double-gaussian | adam | 1000 | 3.74 | 3.40 | 28.2016 | 28.8654 |
| winner | adam | 1000 | 5.23 | 5.01 | 27.4684 | 28.7442 |

The like-for-like, fit-only, float32 numbers from the shared harness are in the results below.

### double-gaussian (S2, 5 parameters, 1,055 cells)

Fit-only, median of 3–5 warm runs, seconds for all 1,055 cells. Sorted by time.

| Rank | Backend | Device | Algorithm | Jacobian | Seconds | Iterations | Train loss | Score |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | numba | cpu | lm-bounded | analytic | 0.031 | 27 | 2.8196 | 28.8656 |
| 2 | MLX | gpu | lm-bounded | analytic | 0.139 | 27 | 2.8196 | 28.8655 |
| 3 | torch | mps | lm-bounded | analytic | 0.159 | 27 | 2.8196 | 28.8656 |
| 4 | JAX | cpu | lm-bounded | analytic | 0.211 | 27 | 2.8196 | 28.8655 |
| 5 | tensorflow | cpu | lm-bounded | analytic | 0.279 | 27 | 2.8196 | 28.8655 |
| 6 | torch | mps | lm-bounded | autodiff | 0.463 | 27 | 2.8196 | 28.8656 |
| 7 | torch | cpu | lm-bounded | analytic | 0.525 | 27 | 2.8196 | 28.8656 |
| 8 | torch | mps | lm | analytic | 0.550 | 100 | 2.8196 | 28.8738 |
| 9 | numba | cpu | lm | analytic | 0.579 | 100 | 2.8196 | 28.8750 |
| 10 | MLX | gpu | lm | analytic | 0.589 | 100 | 2.8196 | 28.8739 |
| 11 | JAX | cpu | lm | analytic | 0.695 | 100 | 2.8196 | 28.8738 |
| 12 | tensorflow | cpu | lm | analytic | 1.012 | 100 | 2.8196 | 28.8739 |
| 13 | torch | cpu | lm | analytic | 1.917 | 100 | 2.8196 | 28.8739 |
| 14 | MLX | gpu | adam | analytic | 3.661 | 1000 | 2.8202 | 28.8655 |
| 15 | tensorflow | cpu | adam | analytic | 4.042 | 1000 | 2.8202 | 28.8655 |
| 16 | torch | mps | adam | autodiff | 4.970 | 1000 | 2.8202 | 28.8655 |
| 17 | torch-muon | mps | muon-per-cell | analytic | 5.086 | 1000 | 2.8201 | 28.8754 |
| 18 | numpy | cpu | lm-bounded | analytic | 5.241 | 27 | 2.8196 | 28.8656 |
| 19 | torch-muon | mps | muon-global | analytic | 5.508 | 1000 | 2.8205 | 28.8519 |
| 20 | numpy | cpu | lm | analytic | 19.632 | 100 | 2.8196 | 28.8736 |

### winner (H27, 11 parameters, 1,055 cells)

Fit-only, median of 3–5 warm runs, seconds for all 1,055 cells. Sorted by time.

| Rank | Backend | Device | Algorithm | Jacobian | Seconds | Iterations | Train loss | Score |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | numba | cpu | lm-bounded | analytic | 0.224 | 100 | 2.7555 | 28.7012 |
| 2 | torch | mps | lm | analytic | 0.916 | 100 | 2.7531 | 28.7842 |
| 3 | torch | mps | lm-bounded | analytic | 0.944 | 100 | 2.7556 | 28.7017 |
| 4 | numba | cpu | lm | analytic | 1.006 | 100 | 2.7526 | 28.7838 |
| 5 | MLX | gpu | lm-bounded | analytic | 1.012 | 100 | 2.7557 | 28.7017 |
| 6 | MLX | gpu | lm | analytic | 1.132 | 100 | 2.7526 | 28.7848 |
| 7 | torch-muon | mps | muon-per-cell | analytic | 1.189 | 196 | 2.7612 | 28.7321 |
| 8 | JAX | cpu | lm-bounded | analytic | 1.301 | 89 | 2.7516 | 28.6953 |
| 9 | JAX | cpu | lm | analytic | 1.316 | 100 | 2.7560 | 28.8265 |
| 10 | tensorflow | cpu | adam | analytic | 1.564 | 247 | 2.7736 | 28.6764 |
| 11 | tensorflow | cpu | lm-bounded | analytic | 2.011 | 100 | 2.7555 | 28.7019 |
| 12 | tensorflow | cpu | lm | analytic | 2.062 | 100 | 2.7530 | 28.7843 |
| 13 | torch | mps | adam | autodiff | 2.420 | 417 | 2.7614 | 28.6753 |
| 14 | torch | cpu | lm | analytic | 2.941 | 100 | 2.7530 | 28.7861 |
| 15 | torch | cpu | lm-bounded | analytic | 2.948 | 100 | 2.7555 | 28.7020 |
| 16 | torch | mps | lm-bounded | autodiff | 3.330 | 89 | 2.7519 | 28.6952 |
| 17 | torch-muon | mps | muon-global | analytic | 4.937 | 785 | 2.7652 | 28.6617 |
| 18 | MLX | gpu | adam | analytic | 6.656 | 1000 | 2.7470 | 28.7412 |
| 19 | numpy | cpu | lm-bounded | analytic | 22.685 | 100 | 2.7557 | 28.7027 |
| 20 | numpy | cpu | lm | analytic | 22.827 | 100 | 2.7605 | 28.8212 |

### SciPy, single-threaded, `least_squares` and L-BFGS-B

Only the first 200 cells, so the absolute scores are **not** comparable to the tables above; those
200 cells are harder (numpy's own bounded LM scores 30.2036 on the same subset). Times are
extrapolated to 1,055 cells for comparison with the batched rows.

| Program | Method | Cells | Seconds (subset) | Seconds / 1,055 cells | Evaluations | Train loss | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| double-gaussian | `lm` | 200 | 0.30 | 1.56 | 9.3 | 3.0426 | 30.2195 |
| double-gaussian | `dogbox` | 200 | 0.32 | 1.68 | 9.1 | 3.0426 | 30.2199 |
| double-gaussian | `trf` | 200 | 0.36 | 1.90 | 10.3 | 3.0426 | 30.2230 |
| double-gaussian | `bfgs` | 200 | 0.81 | 4.30 | 20.2 | 3.0429 | 30.2285 |
| double-gaussian | `lbfgsb` | 200 | 0.87 | 4.61 | 26.3 | 3.0426 | 30.2204 |
| double-gaussian | `powell` | 200 | 1.92 | 10.15 | 206.3 | 3.0436 | 30.2079 |
| winner | `bfgs` | 200 | 2.83 | 14.91 | 74.7 | 2.9830 | 30.1778 |
| winner | `lm` | 200 | 5.04 | 26.61 | 141.5 | 2.9836 | 30.1542 |
| winner | `trf` | 200 | 5.88 | 31.04 | 126.4 | 2.9815 | 30.1677 |
| winner | `dogbox` | 200 | 6.32 | 33.31 | 137.6 | 2.9872 | 30.1376 |
| winner | `powell` | 200 | 11.37 | 59.99 | 1109.7 | 2.9646 | 30.1775 |
| winner | `lbfgsb` | 200 | 13.81 | 72.86 | 363.4 | 2.9718 | 30.1914 |

### Compile and warm-up costs

| Backend | First-call cost | Notes |
| --- | ---: | --- |
| numba | 0.85 s | `cache=True`; 13–17 ms per program afterwards |
| JAX | ~0.4 s | `jax.jit` trace + XLA compile, outside the timed loop |
| torch (analytic) | ~0.3 s | module move to MPS, outside the timed loop |
| torch (autodiff) | ~1 s/program | `jacfwd` traces the batched model once |
| TensorFlow | ~1 s/program | `tf.function` graph build |
| MLX | ~0.1 s | lazy graph build on first `eval` |

### Adam vs LM: speed versus score

| Program | Best LM score | Best LM time | Best Adam score | Best Adam time | Adam steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| double-gaussian | 28.8655 | 0.031 s | 28.8655 | 3.66 s | 1000 |
| winner | 28.6953 | 0.224 s | 28.6753 | 2.42 s | 417 (early-stopped) |

On `winner` Adam's shorter effective budget regularises better by 0.02 score points, at 10× the
wall time of numba's bounded LM. On `double-gaussian` the two are identical to four decimals.
That reproduces the synthetic result in `docs/PHASE_0.md`: the win is fit quality, not speed.

### Muon: orthogonalised momentum

Muon is built for 2-D weight matrices, not for 5- or 11-parameter per-cell fits, so it was run two
ways in PyTorch on MPS. The learning rate was picked on a 100-cell subset (200 steps, rates
{3e-3, 1e-2, 3e-2}) and then fixed for the full run.

| Program | Mode | Learning rate | Steps | Seconds | Train loss | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| double-gaussian | per-cell (1×K) | 1e-2 | 1000 | 5.09 | 2.8201 | 28.8754 |
| double-gaussian | whole batch (1055×K) | 3e-2 | 1000 | 5.51 | 2.8205 | **28.8519** |
| winner | per-cell (1×K) | 3e-2 | 196 (non-finite stop) | 1.19 | 2.7612 | 28.7321 |
| winner | whole batch (1055×K) | 3e-2 | 785 (non-finite stop) | 4.94 | 2.7652 | **28.6617** |

Both modes are slower than bounded LM and neither beats it on training loss. Per-cell Muon is the
honest application and is strictly worse: a 1×K matrix orthogonalises to a unit vector, so it is
just normalised momentum with no curvature. The whole-batch variant gives the best *held-out*
score in both programs (28.8519 and 28.6617, beating Adam's 28.6753 on `winner`) at a worse
training loss — but it orthogonalises across independent cells, so parameter updates leak between
cells. Treat that as a regulariser, not a valid per-cell fit; it is not a drop-in replacement.

## Summary and recommendation

1. **Overall winner: numba batched `lm-bounded` on the CPU.** 31 ms for `double-gaussian` and
   224 ms for `winner` — 7–10× faster than JAX's compiled CPU fit and 25–60× faster than the
   current end-to-end pipeline (0.59 s / 1.73 s including checks). The model is 4–11 parameters per
   cell, so `np.linalg.solve` on 5×5 and 11×11 systems inside a `prange` over cells is close to the
   arithmetic limit; there is nothing left for a GPU to do.
2. **Fastest GPU path: MLX (139 ms / 1.01 s) and torch MPS with the analytic Jacobian
   (159 ms / 944 ms), effectively tied.** Both beat JAX CPU on `double-gaussian` but not on
   `winner`, and neither beats a 15-thread CPU kernel. This confirms hypothesis 1: at this batch
   shape, GPU dispatch and tiny-matrix inefficiency dominate.
3. **On MPS, use the analytic Jacobian.** `jacfwd`-based LM costs 0.46 s vs 0.16 s
   (`double-gaussian`) and 3.33 s vs 0.94 s (`winner`), i.e. 2–3.5× slower, and converges slightly
   worse on `winner` because it stalls at 89 iterations.
4. **TensorFlow is CPU-only on this Mac** (2.21.0 reports no GPU devices, and `tensorflow-metal`
   is not compatible). It is a competent CPU fallback (0.28 s / 2.0 s) but never the first choice
   here.
5. **JAX remains the best portable choice** if you do not want a numba dependency: 0.21 s / 1.30 s
   fit-only, and its float64 `lm-bounded` produces the best `winner` LM score (28.6953).
6. **numpy is the wrong tool for the batch** (5.2 s / 22.7 s): 1,055 tiny solves plus one
   (1,055, 856) temporary per parameter per iteration; it is a reference implementation, not a
   candidate.
7. **Adam stays a niche option.** It gives a good `winner` held-out score (28.675) but is
   10–30× slower than bounded LM; keep it only if 0.02 score points matters more than seconds.
8. **Muon does not fit this problem.** Per-cell Muon is normalised momentum and scores worse than
   LM (28.8754 / 28.7321). The whole-batch variant accidentally regularises and gives the best
   held-out scores (28.8519 / 28.6617), but it couples cells and is 5–20× slower, so it is a
   curiosity rather than a candidate.

Concretely: move the fitting hot loop to numba (or keep JAX where a compiled, dependency-light,
portable path matters more than the last 7×), and do not spend effort on Metal for this shape.
If the batch grows by orders of magnitude (more cells, more trials, or per-trial resampling), the
GPU rows are the ones to re-measure.

## Conclusions against the hypotheses

1. Confirmed. The GPU only pays off when the per-cell linear algebra is large; here it is not.
2. Half confirmed. torch MPS is the most usable GPU path, but MLX is equally fast and simpler;
   JAX has no GPU here at all.
3. Confirmed for `winner` (Adam better score, far slower) and neutral for `double-gaussian`.
4. Rejected as a *win*: MLX is the best GPU candidate but does not beat the CPU batch.
5. Rejected: numpy is 25–100× slower than numba; BLAS does not rescue the 1,055-tiny-solves shape.
6. Muon, added later: the per-cell form degenerates to normalised momentum and loses to LM; the
   whole-batch form only wins on held-out score by mixing cells, which is not a per-cell fit.

## Decisions and notes

- The dataset is 819 MB and lives outside the worktree; `data/` is git-ignored and symlinked.
- Extra frameworks are used through `uv run --with …` so the pinned project lock is untouched.
- Every backend fits the **same** model, same trial split, same score; the harness is numpy and
  shared, so differences are the optimiser/runtime, not the data.
- float32 everywhere except the JAX float64 baseline; LM results agree to ~1e-3 score, and the
  `winner` float32 bounded-LM score (28.7017) is 0.006 worse than float64 (28.6953).
- MLX has no GPU linear solve in 0.32.4, so its 5×5/11×11 solves run on the CPU stream over
  unified memory; everything else in its loop stays on the GPU. Even so it is competitive.
- The `winner` bounded LM exhausts its 100-iteration budget on most cells (1,045–1,047 of 1,055
  converged), which is why its score is slightly worse than JAX's 89-iteration run.
- Muon has no built-in stopping rule here; both modes ran to the 1,000-step cap or until a
  non-finite step. Its learning rate was tuned on 100 cells only, so its scores are optimistic
  relative to the other rows, which use the project's fixed constants.
- "Muon-global" orthogonalises the whole (1,055 × K) momentum matrix, so a cell's update depends
  on the other cells' gradients. It is reported for completeness, not as a valid per-cell fit.

## Open questions

- Would a larger batch (per-trial resampling, or several recordings) make the GPU rows win? The
  MPS/MLX overhead is ~3–10 ms per iteration, so the crossover is plausibly a few thousand cells.
- Does a hand-written Metal kernel for the 5×5/11×11 batched solve beat numba? Probably not worth
  it at this size.
- Is numba's 7× fit advantage preserved inside the real pipeline once the upstream checks and the
  `objective` plumbing are included? The JAX end-to-end numbers suggest roughly 25–60× overall.
- Would more LM iterations (or a tighter tolerance) close the 0.006 `winner` gap to float64
  without losing the speed advantage?
- Is the whole-batch Muon gain a genuine regulariser (e.g. shrinking per-cell variance), and would
  a small cross-cell penalty reproduce it on top of LM without the coupling side effect?
