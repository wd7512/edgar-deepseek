# Roadmap

Issues are disabled on this repository, so later phases are tracked here. Status and run history are in the [handover doc](https://claude.ai/artifact/UMTbnxaYf8tcoiMSm8HK5w). Each phase is gated by the one before it.

## Phase 0: like-for-like EDGAR replication

Run EDGAR's unchanged discovery loop with OpenCode Go models in place of Gemini. See [PHASE_0.md](PHASE_0.md).

- [x] Free model `hy4-flash-free` ($0): a 1-generation smoke run with locked-cell rescore, and a 12-generation run stopped after 4 generations.
- [x] Faster CPU fitting (compiled, core-sharded Adam) and a 24-minute fast run with locked-cell rescore.
- [ ] Paid run: DeepSeek V4.1 Flash writes all code; Muse Spark 1.3 Contributor reads diagnostic images only. Total budget $10. Run paper-sized searches on a GPU.
- [ ] rcvPCA eigenspectrum check of the selected equations against the paper's slopes (4.26 / 4.34).

## Phase 1: off-the-shelf agentic harness (next)

Question: does EDGAR's evolutionary scaffold add anything over a coding agent that has the same scorer and budget?

- **Design: the harness as the scientist.** One `opencode run` session on DeepSeek V4.1 Flash gets:
  - the discovery cells;
  - the frozen scorer (`hypothesis_engine.objective` + `loss_functions.quadratic_loss`, penalty 0.01) as a CLI tool that returns the penalized discovery loss and a diagnostic PNG;
  - a fixed budget.

  The agent writes, scores and revises `neuron_model` and `parameter_estimator` itself. Running through OpenCode's own harness is also the clearest permitted way to use the Go plan.
- **Smaller alternative: the harness as the LLM call.** Each proposal, translation or estimator request becomes an `opencode run` subprocess. This only adds overhead, so it is worth it only if Go restricts automated traffic to its own harness.
- **Fairness rules:**
  - same model, and the same budget: 576 scorer calls or the same list-equivalent USD;
  - same `strict-holdout-v1` cells: the agent never sees the selection or locked-test cells;
  - same `rescore` rule: the agent nominates its top programs, selection happens on the selection cells, locked cells are used once.
- **Report:** best-so-far discovery loss against scorer calls on the same axes as Phase 0, the locked-test gain over the double Gaussian, and whether p < 2 appears.

First steps:

- [ ] Wrap the scorer as a CLI tool (`src/edgar_study/score_tool.py`): model file and estimator file in, JSON plus PNG out, `--fitter auto`, calls counted against the budget.
- [ ] Write the agent's task prompt: data description, tool usage, budget, the deliverable (top 10 programs as files), no access to held-out cells.
- [ ] Pilot on `hy4-flash-free` with a 48-call budget, comparable to the fast run, for $0.
- [ ] Matched paid run against the Phase 0 DeepSeek run.

## Phase 2: custom equation-discovery harness (later)

Start only once Phase 1 shows whether an agent loop helps at all. Same fairness rules, compared against Phases 0 and 1.

- **Tools:** the scorer; a fit plotter; per-cell residual summaries (peak, flanks, baseline); a parameter-ablation helper like the paper's Fig. 3e.
- **Memory:** the program database as a queryable tool, with scores, lineage and fitted-parameter distributions, rather than parent code pasted into prompts.
- **Parallelism:** optional parallel sub-agents in place of islands, with migration done as shared notes.
- **Regularisation:** decide explicitly between Adam's implicit early stopping and LM with validation stopping, so fit quality and speed stop pulling against each other.
- **Contamination control:** test on synthetic tuning laws the model can't have memorised before claiming discovery.

First steps:

- [ ] Write down Phase 1's failure modes (wasted scorer calls, repeated ideas, overfitting) to decide which tools matter.
- [ ] Build the residual-summary and ablation tools on top of `src/edgar_study/fitting.py`.
- [ ] Generate 2–3 synthetic "unseen law" datasets as a discovery benchmark.
