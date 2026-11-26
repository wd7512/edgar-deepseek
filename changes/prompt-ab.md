Add a second proposal prompt (`v2`) and `edgar prompt-ab`, which A/B tests prompts on fixed contexts.

The `v2` prompt:
- states the data and the objective without any score: one equation fitted separately to about 800 neurons, judged on held-out trials, with a cost per parameter;
- states the grammar's limits, taken from the grammar's own constants;
- gives stability rules: no denominators that can cross zero, positive widths, `tanh`-bounded changes, no needles;
- adds an image checklist adapted from the paper;
- asks for a short analysis;
- marks earlier equations on a revisit as fitted or could not be fitted;
- includes an example object.

Each parent now also gets a population residual image: residuals averaged over neurons, aligned to each neuron's preferred direction.

`prompt-ab` takes contexts from a finished run, samples every arm on the same contexts and the same neurons, and fits each equation on a subset of exploration neurons against its parent. It reports outcome rates and paired gains.

A revisit context is skipped when its parent route includes an idea with no fitted equation, instead of crashing. The output directory is created only after the free-model check.
