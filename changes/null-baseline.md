Added a no-LLM null baseline for the equation search:
- `edgar_study.symbolic`: expression trees over the stimulus angle whose parameter leaves are fitted per neuron, turned into the same NumPy program, JAX program and parameter estimator an LLM writes. It has random trees, mutation, crossover, the two upstream seeds as trees, and a parser for PySR-style formulas.
