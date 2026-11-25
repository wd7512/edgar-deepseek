# EDGAR knowledge-frontier pilot

**Question:** Can `hy4-flash-free` learn which equation changes fix recurring errors in neural tuning curves when its next task is chosen by a gap in knowledge rather than a score ranking?

The pilot is separate from the existing EDGAR loop. Start with the double Gaussian, judge fit across the whole curve, and report regional improvements and regressions separately. The pilot tests a workflow; it cannot establish that this search strategy is better than EDGAR.

![A conjunctive prerequisite graph and its exploration rule](knowledge-frontier.svg)

## 1. Represent claims and evidence separately

Use a directed hypergraph: each route stores a set of prerequisite claim IDs and one conclusion ID. All prerequisites in a route must be supported (**AND**); any supported route is sufficient (**OR**). Validate acyclicity on the graph formed by expanding routes into prerequisite-to-conclusion arcs. Unobserved premises are not assumptions.

- **Claims** are falsifiable and scoped: “On this discovery cohort, separate widths correct the main peak's flank error beyond what a free center can explain.” Claims without incoming routes can be tested directly. Derived claims are evaluated from their routes, not marked supported by the model.
- **Tests** are evidence records, not prerequisite edges. Store the equation or controlled variant, the prediction (if any), the discovery split and comparator, the affected angle regions, fit diagnostics, and any provisional outcome. A failed fit is inconclusive; mixed evidence stays unresolved. A contradicted prerequisite blocks that route, not every route to its conclusion.
- **Branches** are stable scheduling groups, initially peak tips, flanks, second-peak placement, and baseline. Admit a branch to the pilot only if an observed residual motivates it. Archive previous EDGAR results as context unless they supply a controlled comparison under the same protocol.

For example, a flank-asymmetry claim could be supported by three separate findings: a recurring signed flank pattern, persistence when the peak center is freed, and a width-split improvement without meaningful regression elsewhere. An alternative route could instead use a controlled factorial fit to establish the isolated contribution of width asymmetry together with replication on a second discovery cohort. The diagram shows hypothetical evidence, not run results.

## 2. Explore gaps and make room for surprises

Measure **evidence depth** per branch as the number of distinct, completed, interpretable contrasts, including negative results. Graph depth is separate. Refits of the same contrast, near-duplicate parameter tweaks, and failed executions do not add depth. Replications can strengthen confidence without counting as a new contrast.

Run in rolling waves. At the start of each exploratory episode, draw a lane: **60% frontier**, **30% detour**, **10% new root**. These are starting probabilities, not quotas; keep them fixed within a wave. Draw only among available lanes, renormalizing if one is empty, and pause if none are available. The scheduler draws once per episode, not between tests within a batch.

- **Frontier:** Draw an eligible branch with weight `1 / (1 + evidence_depth)`. Within it, draw a testable open node with weight `1 / ((1 + prior_attempts_at_node) × (1 + similar_ideas_already_tried))`. Normalize weights separately at each step. In the pilot, count distinct previously tried ideas with the same structural mechanism tag as the node, excluding the node itself. The harness assigns tags from the equation grammar, not from wording or fit. This makes a new mechanism more likely than a small variation on a crowded one. Sampling branches first prevents a branch from gaining priority just because it has more nodes. With depths 6, 2, and 1, the conditional branch probabilities are about 15%, 34%, and 51%. If a route is contradicted, seek a materially different test or route.
- **Detour:** The harness draws from unresolved observations, outliers flagged in discovery, or combinations of ideas from unrelated branches. Prefer untried mechanism tags. Record leads and their lineage separately from the graph. Test the batch before drawing again; an early poor fit does not stop it.
- **New root:** Draw a distinct, directly testable idea from a separate pool of observed patterns and untried mechanism families, seeded by the experimenter or predefined templates. The idea must have no prerequisites in the current map. If it's grounded in an observed pattern and not a duplicate, add it as a provisional standalone node; it can stay disconnected or acquire links later. If no candidate qualifies, treat the lane as unavailable rather than forcing a root.

**HY4's only job is to turn a supplied idea or thread into a batch of equations.** The experimenter authors the fixed templates and initial idea descriptions. The harness resolves the draw to a prompt type and fills in the fields; HY4 does not see the draw probability, graph state, or previous evaluation. A thread is an ordered set of design ideas, not a claim that they have been confirmed. Do not merge incompatible routes as if both were true.

### Prompt templates

**Combination** — use for a frontier route with multiple compatible ideas or for a cross-branch detour. Keep this wording fixed:

```text
Combine the ideas in this thread:

- Idea 1 — Base equation: {base_equation}
{numbered_ideas_2_through_N}

Give {K} structurally distinct equations that combine these ideas
and could fit the data shown in the attached images.
```

**Single-node frontier** — use when the draw targets one change rather than a combination:

```text
Base equation: {base_equation}
Idea to test: {selected_idea}
Keep unchanged: {unchanged_terms}

Give {K} structurally distinct equations that each test this idea
while keeping the listed terms unchanged. Use the attached images
of the data and current fit as context.
```

**Anomaly detour** — use when a surprising pattern motivates a detour and the harness has a specific idea to try:

```text
Pattern to look at: {observable_pattern_and_plot_region}
Idea to try: {detour_idea}

Give {K} structurally distinct equations that express this idea
and could fit the data in the attached images. You may change the
equation family; the displayed fit is a reference, not a constraint.
```

**Independent root** — use for the new-root draw. The harness supplies the seed idea, not predecessors, and the model need not inherit any term from the plotted fit:

```text
New standalone idea: {root_idea}

Give {K} structurally distinct full equations based on this idea
that could fit the data in the attached images. Build from the
idea itself; the plotted reference fit is not a starting equation.
```

Append the same output contract to each template:

```text
{angle_and_output_conventions}
{allowed_equation_grammar}
Return each equation and its free-parameter definitions. Use different
structures, not just different parameter values. Do not fit or rank them.
```

The harness attaches observed discovery responses overlaid with the current fitted comparator, using a fixed or seeded cell sample. The comparator is the double Gaussian or the reference named by the selected node, not a model picked for having the best score. The overlay shows what the thread is meant to address; it does not compare or grade proposals. Do not send scalar or per-cell scores, rankings, win/loss labels, or prose summarizing previous candidate performance, even on follow-up calls.

[Example attachment: four discovery neurons overlaid with the double-Gaussian fit](knowledge-frontier-discovery-example.png).

The harness validates outputs, fits all admissible equations in a batch, compares results, and updates the map. Canonicalize expressions to reject exact duplicates and count relabelings or parameter-only tweaks as the same family; prefer coverage of distinct structures when filling a batch. No lane uses fit scores to set draw weights. Save the prompt-template version, resolved text, panel inputs, random seed, candidate sets, effective probabilities, and draws so scheduling and model inputs can be reconstructed.

## 3. Let results develop before promoting claims

The exploratory loop can record **observations and hypotheses**, but cannot mark claims supported. At a wave boundary, freeze any promising hypothesis, its prediction, the comparison, and the pass/fail rules before checking it on discovery cells reserved for confirmation and never shown in exploratory panels. If the result fails to replicate or trades one regional error for another, leave the claim unresolved. Preserve the distinction between “we noticed this” and “this held up when we checked.” If fresh confirmation data are exhausted, stop promoting claims, not necessarily exploring. Selection and locked-test cells stay outside both loops.

## 4. Run and evaluate the pilot

1. **Build the map and scheduler first.** Use versioned JSON and a testable map/scheduler module under `src/edgar_study/`. Test AND/OR semantics, cycle rejection, blocked routes, canonical duplicate detection, inverse novelty weighting, exact normalized probabilities, seeded replay, empty lanes, and disconnected roots. Verify that eligible deeper branches retain nonzero probability, a negative result does not disqualify a branch, and changing numeric fit rankings alone cannot change scheduling, the prompt text, or the plotted cells and comparator.
2. **Build the discovery-only experiment adapter.** Reuse existing dataset splits and fitting code. Start with a small declarative library of tuning-curve terms and modifiers; compile validated proposals locally rather than executing arbitrary generated Python. Freeze the discovery cohort, trial split, angle windows, fitting budget, and evidence rules before calling the model. Compute paired residual changes on held-out discovery trials for the nominated region and the whole curve. On discovery, judge whether the targeted change helped without moving the error elsewhere.
3. **Exercise the whole loop offline.** Use a synthetic fixture with a planted residual pattern and a stub model. Check that the known contrast produces a lead but only fresh confirmation supports the claim, an irrelevant contrast does not, and confirmation, selection, and locked-test data are unavailable to the exploratory agent.
4. **Then run the free pilot.** Seed the map from a fixed discovery diagnostic panel and log every scheduling decision and update. Review after six four-equation batches, not as a claim deadline or a total cap. Continue while waves produce new contrasts or unresolved leads worth pursuing. The free model removes the token-cost constraint, not compute time, rate limits, or the risk of overfitting repeatedly viewed data. Freeze the shortlist and selection rule before opening selection cells; use the locked test once for the selected equation versus the double Gaussian. Report regional and full-curve fit and complexity separately.

**Deliverables:** the versioned knowledge map, a readable graph of supported and unresolved claims, the test log, and a shortlist of equations with evidence for what each fixes. Pilot success means the map explains why tests were chosen, shallow branches receive attention regardless of early fit results, and the evidence supports or narrows claims. Finding a better equation is a result to report, not a condition for calling the pilot informative.
