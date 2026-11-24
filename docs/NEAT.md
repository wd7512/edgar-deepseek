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
  --population 96 --generations 6 --cell-fraction 0.25
```

The run directory holds `manifest.json`, `programs.jsonl`, `search.json`, `rescore.json` and the
winner's source.

## Results

Proof of concept on the 25% discovery subset (264 cells), 96 genomes × 6 generations, CPU,
`lm-bounded`. Every genome starts from the smallest possible network: the three inputs connect
directly to one output, with no hidden neurons. NEAT adds nodes and connections from there.

| Run | Best discovery | Winner selection | Winner locked | Δ locked vs double Gaussian |
| --- | ---: | ---: | ---: | ---: |
| NEAT from nothing | 30.12 | 29.38 | 29.04 | −0.47 |
| Double Gaussian seed | — | 28.85 | 28.57 | 0 |

Best discovery score and hidden-neuron count per generation:

| Generation | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Best score | 34.17 | 31.61 | 31.50 | 30.31 | 30.24 | 30.12 |
| Gain | — | −2.56 | −0.12 | −1.19 | −0.06 | −0.12 |
| Genomes with 0 / 1 / 2 / 3 hidden | 96 / 0 / 0 / 0 | 57 / 39 / 0 / 0 | 31 / 48 / 17 / 0 | 5 / 58 / 33 / 0 | 2 / 55 / 33 / 6 | 6 / 61 / 28 / 1 |

The score improved every generation but the last two gained only 0.18 in total, so the search was
converging rather than opening a new structural direction. The winner has **one hidden neuron**
(`sin`, 7 fitted parameters, 5 connections), and the ten best genomes are all 30.12–30.32, nine of
them with one hidden neuron. NEAT found a harmonic model, not the paper's asymmetric bump:

`softplus(b + w·1 + w·sin θ + w·sin(b′ + w·cos θ + w·sin θ))`

It beats the score it started from but not the 5-parameter double Gaussian. With 576 evaluations
the search spent most of its budget discovering that one hidden unit helps; it did not reach the
paper's shape.

## Caveats

- NEAT searches architecture and initial weights; the per-cell fit is still gradient descent, as
  the score requires. This is not a network trained end to end by neuroevolution.
- A network is not an equation. A better locked-cell score would show that the model class can fit
  the cells, not that it recovers the paper's mechanism.
- One seed, one host. Repeat before drawing conclusions about the search itself.
