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

Run in rolling waves, not a 12-proposal total cap. As a starting cadence, take two coverage turns for each serendipity turn. For coverage, choose an eligible branch with the lowest evidence depth; break ties by oldest last attempt, then stable ID. If a route is contradicted, seek a materially different test or route. If nothing remains testable, log why and move on. Spread failed attempts across branches rather than looping on one.

On serendipity turns, give `hy4-flash-free` a randomly sampled diagnostic panel, including occasional outliers, failed variants, or evidence from two unrelated branches. Let it follow an unexpected pattern, combine ideas, or try a different equation family without first fitting the idea into the graph. Keep these leads and their lineage in an observation log separate from the prerequisite graph. Give a novel lead up to three consecutive discovery contrasts before returning to coverage, even if the first contrast makes the fit worse. Preserve dead ends and revisit unresolved leads in later waves if new evidence makes them testable.

For coverage turns, give the model the relevant subgraph, comparator curves, signed residuals and prior outcomes. Ask it to name one gap, propose an equation contrast, and predict what should change and what should stay unchanged. No global score ranking goes into either lane. The model may suggest new nodes or routes; the harness validates them before adding them.

## 3. Let results develop before promoting claims

The exploratory loop can record **observations and hypotheses**, but cannot mark claims supported. At a wave boundary, freeze any promising hypothesis, its prediction, the comparison, and the pass/fail rules before checking it on discovery cells reserved for confirmation and never shown in exploratory panels. If the result fails to replicate or trades one regional error for another, leave the claim unresolved. Preserve the distinction between “we noticed this” and “this held up when we checked.” If fresh confirmation data are exhausted, stop promoting claims, not necessarily exploring. Selection and locked-test cells stay outside both loops.

## 4. Run and evaluate the pilot

1. **Build the map and scheduler first.** Use versioned JSON and a testable map/scheduler module under `src/edgar_study/`. Write offline tests for AND/OR semantics, cycle rejection, blocked routes, duplicate contrasts, and shallow-branch selection even after a negative result. Verify that changing numeric fit rankings alone cannot change scheduling.
2. **Build the discovery-only experiment adapter.** Reuse existing dataset splits and fitting code. Start with a small declarative library of tuning-curve terms and modifiers; compile validated proposals locally rather than executing arbitrary generated Python. Freeze the discovery cohort, trial split, angle windows, fitting budget, and evidence rules before calling the model. Compute paired residual changes on held-out discovery trials for the nominated region and the whole curve. On discovery, judge whether the targeted change helped without moving the error elsewhere.
3. **Exercise the whole loop offline.** Use a synthetic fixture with a planted residual pattern and a stub model. Check that the known contrast produces a lead but only fresh confirmation supports the claim, an irrelevant contrast does not, and confirmation, selection, and locked-test data are unavailable to the exploratory agent.
4. **Then run the free pilot.** Seed the map from a fixed discovery diagnostic panel and log every scheduling decision and update. Start with 24 proposal turns before a progress review, not a claim deadline or a total cap. Continue while waves produce new contrasts or unresolved leads worth pursuing. The free model removes the token-cost constraint, not compute time, rate limits, or the risk of overfitting repeatedly viewed data. Freeze the shortlist and selection rule before opening selection cells; use the locked test once for the selected equation versus the double Gaussian. Report regional and full-curve fit and complexity separately.

**Deliverables:** the versioned knowledge map, a readable graph of supported and unresolved claims, the test log, and a shortlist of equations with evidence for what each fixes. Pilot success means the map explains why tests were chosen, shallow branches receive attention regardless of early fit results, and the evidence supports or narrows claims. Finding a better equation is a result to report, not a condition for calling the pilot informative.
