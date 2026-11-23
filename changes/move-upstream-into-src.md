Moved the six frozen upstream EDGAR modules from the repo root to `src/edgar_upstream/` (byte-identical, still checked against `0943922`) and `conftest.py` to `tests/`, so the repo follows a standard src layout.
Agents may now push their own branch and open a pull request (AGENTS.md), but never push to `main`.
