#!/bin/sh
# Re-run the backend benchmark and rewrite scripts/backend_bench/results.jsonl.
set -eu
cd "$(dirname "$0")/../.."
OUT=scripts/backend_bench/results.jsonl
: > "$OUT"

uv run scripts/backend_bench/prepare.py
uv run scripts/backend_bench/fit_jax.py --repeats 5 >> "$OUT"
uv run --with numba scripts/backend_bench/fit_numba.py --repeats 5 >> "$OUT"
uv run scripts/backend_bench/fit_numpy.py --repeats 3 >> "$OUT"
uv run scripts/backend_bench/fit_scipy.py --cells 200 >> "$OUT"
uv run --with torch scripts/backend_bench/fit_torch.py --device cpu --algorithms lm lm-bounded --jacobian analytic --repeats 3 >> "$OUT"
uv run --with torch scripts/backend_bench/fit_torch.py --device mps --algorithms lm lm-bounded --jacobian analytic --repeats 3 >> "$OUT"
uv run --with torch scripts/backend_bench/fit_torch.py --device mps --algorithms lm-bounded --jacobian autodiff --repeats 1 >> "$OUT"
uv run --with torch scripts/backend_bench/fit_torch.py --device mps --algorithms adam --repeats 1 >> "$OUT"
uv run --with tensorflow scripts/backend_bench/fit_tf.py --repeats 3 >> "$OUT" 2>/dev/null
uv run --with mlx scripts/backend_bench/fit_mlx.py --repeats 3 >> "$OUT"
echo "wrote $OUT"
