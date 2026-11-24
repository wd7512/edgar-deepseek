# Leakage audit

Run 20261123-170837-219bfc fitted an exponent p in generation 0. This audit asks whether the paper's equation, or the held-out neurons, reached the search some other way.

| Path | Finding | Action |
| --- | --- | --- |
| Seed programs | Single and double Gaussian with p fixed at 2. No exponent and no mention of the paper. | None |
| Upstream prompts (frozen, the paper's own) | The image instructions ask whether the models' "shapes (e.g., peak sharpness, width, skewness, amplitude)" match the data. With a plot of real, peaky responses, a shape exponent is a natural first change. The paper's Gemini saw the same text. | None: this is the method, not leakage |
| Our image-reader prompt | Asked for "peak sharpness or cusp shape". "Cusp" points at the paper's result (p < 2). It is only sent when `--image-model` is set; run 219bfc did not set it, and its API log has no image-reader calls. | Reworded to upstream's own list; a test forbids shape hints such as cusp, power or exponent |
| The model's memory | Asked directly at temperature 0, `hy4-flash-free` said it does not know the paper and that its knowledge cutoff predates it. Asked about EDGAR, it guessed a plain von Mises. Asked blind for a tuning equation, it proposed a von Mises with no exponent. | None |
| Images | Upstream plots 9 random cells from its training half, which is exactly the discovery set (same seed-42 permutation). | None |
| Candidate pool | Upstream's combined database removes near-duplicate programs (discovery loss within 0.025, outputs over 0.99 cosine-similar) by keeping the lower `test_loss`, which is scored on the selection and locked-test cells. `rescore` read that database, so locked cells could decide which near-duplicate competed. | `rescore` now reads the island databases, deduplicated on discovery loss only, and never reads `test_loss` |

Re-running `rescore` on run 219bfc with the island pool changes 5 of the 10 candidates, but the winner is the same program (10_2_1, selection score 28.653 against 28.681 for the runner-up) with the same locked-test score, 28.322. Its results stand.

The generation-0 docstring credits the exponent to the image: "lobe-specific generalized Gaussian sharpness … target the visible weaknesses". So the exponent was prompted by upstream's own instructions and the data, not recalled from the paper.

## DeepSeek V4.1 Flash (run 20261124-005036-fa1fb9)

The paid run used no image reader, read candidates from the island databases (rebuilt with `edgar recover` after the spend cap stopped it) and so has none of the leaks fixed above. The rebuilt pool holds every program it had scored, not only the ones that survived on the islands; all of them were scored on discovery cells only. Its winner has exactly the paper's 11-parameter form, so we checked what the model wrote:

| Check | Finding |
| --- | --- |
| Mentions of the paper | None of the 299 scored programs mentions Tilbury, EDGAR, bioRxiv, a paper, a publication or the literature. |
| When the form appeared | 11 of 44 generation-0 programs already had two free peaks, separate left and right widths and an exponent. |
| Stated reasons | Each generation-0 docstring justifies each term from specific plotted cells, e.g. a secondary bump "not exactly opposite the main peak", "a very sharp peak whose tails are not well described by a fixed Gaussian shape". |
| Direct recall probe | Not run: it is a paid call and the run had already spent past its cap. |

So nothing in the transcript shows recall, but unlike `hy4-flash-free` this model was not asked directly, and a model trained after November 2025 could know the paper. Treat the match as unconfirmed until that probe is run.
