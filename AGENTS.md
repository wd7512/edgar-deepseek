# Agent instructions

- Use `uv` for everything: `uv sync --locked`, `uv run pytest`, `uv run gate`, `uv run check`. No pip, no direct `python`.
- The six upstream modules in `src/edgar_upstream/` (`diagnostic`, `genetic_helpers`, `hypothesis_engine`, `loss_functions`, `seed_programs`, `utils`) are frozen at upstream commit `0943922`. Never edit them or add an `__init__.py` there; `edgar_study.upstream.import_upstream` refuses to run if they differ.
- Never call paid models or pass `--allow-paid`. Tests must not need the dataset, an API key or the network.
- Work only in your worktree. One change, one commit. Do not push and do not use `gh`.
- Tests first: write the test, see it fail for the expected reason, then write the code. Assert on outputs, not on which internal functions were called.
- Add a fragment file under `changes/` for every branch.
- No comments in code unless asked. Docstrings on public functions are fine. Log with `logging`; `print` only for a command's final output.
- Do not add a dependency unless the brief names it.
- Never pipe a gate or test command through `tail` or `head`; report the real exit code.
