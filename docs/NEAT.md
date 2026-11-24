# NEAT baseline

Question: can a NEAT-evolved network, fitted per cell and scored by EDGAR's objective, match the
double Gaussian or the paper's form without an LLM's prior?

## Method

- **Candidate.** A genome decodes to a feed-forward network over `sin θ`, `cos θ` and a constant.
  Hidden nodes use NEAT's activation choice from `identity, tanh, sin, cos, gauss, square, abs,
  softplus`; the output is a softplus so the predicted rate is positive without a dead gradient.
- **Parameters.** Every enabled weight and bias is a named argument of `neuron_model`, in a fixed
  order. The estimator returns the genome's values with the output bias shifted to the cell's mean
  response, and `edgar_study.fitting`'s bounded Levenberg–Marquardt specialises them per cell.
- **Score.** Unchanged from EDGAR: `10 × held-out MSE + 0.01 × number of weights`. NEAT maximises
  the negative score on the discovery cells, splitting trials with seed 0 exactly as `objective`
  does.
- **Holdout.** The top 10 unique genomes by discovery score are scored on the 527 selection cells;
  the winner and the two Gaussian seeds are scored once on the 528 locked cells.

## Run

```bash
JAX_PLATFORMS=cpu uv run edgar neat \
  --dataset data/gratings_drifting_GT1_2019_04_12_1.npy \
  --population 96 --generations 20 --cell-fraction 0.25
```

The run directory holds `manifest.json`, `programs.jsonl`, `search.json`, `rescore.json` and the
winner's source. The 6-generation proof of concept is the same command with `--generations 6`.

## Results

Proof of concept on the 25% discovery subset (264 cells), 96 genomes per generation, CPU,
`lm-bounded`. Every genome starts from the smallest possible network: the three inputs connect
directly to one output, with no hidden neurons. NEAT adds nodes and connections from there.

| Run | Best discovery | Winner selection | Winner locked | Δ locked vs double Gaussian |
| --- | ---: | ---: | ---: | ---: |
| NEAT, 6 generations | 30.12 | 29.38 | 29.04 | −0.47 |
| NEAT, 20 generations | 29.57 | 28.77 | 28.40 | +0.18 |
| Double Gaussian seed | — | 28.85 | 28.57 | 0 |

Best discovery score per generation, 20-generation run:

| Generation | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best | 34.17 | 31.61 | 30.42 | 30.42 | 29.94 | 29.67 | 29.67 | 29.67 | 29.67 | 29.67 |

| Generation | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best | 29.67 | 29.67 | 29.64 | 29.59 | 29.59 | 29.59 | 29.59 | 29.57 | 29.57 | 29.57 |

The 6-generation run converged prematurely: its winner had one hidden neuron and the ten best
genomes were within 0.2 of each other. The longer run kept mutating and found a better structure at
generation 17. Its winner has **two hidden neurons**, 5 connections and 8 fitted parameters:

`softplus(b + w·softplus(b′ + w·cos θ + w·cos(b″ + w·cos θ + w·sin θ)))`

The constant input is unused; both hidden nodes are trigonometric, the first `cos` and the second
`softplus`. This is a composition of a harmonic and a rectified unit, still not the paper's pair of
Gaussians, but it fits the cells slightly better than the double Gaussian on held-out data.

NEAT kept adding neurons without improving the best score. By generation 19 most genomes had 3–5
hidden nodes, up from 0 at the start, while the best genome stayed at 2. The plateau from
generation 6 to 11 and the final gain at generation 17 suggest the search is limited by mutation
step size, not by the model class; extra neurons mostly added parameters and noise.

## Caveats

- NEAT searches architecture and initial weights; the per-cell fit is still gradient descent, as
  the score requires. This is not a network trained end to end by neuroevolution.
- A network is not an equation. A better locked-cell score would show that the model class can fit
  the cells, not that it recovers the paper's mechanism.
- One seed, one host. The 0.18 locked-cell margin over the double Gaussian has not been checked
  across seeds, and the search needed 20 generations to find it.
