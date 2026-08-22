#!/usr/bin/env bash
# Runs all three benchmarks against what is already built: no nvcc, no
# Pkg.instantiate, no pip. Build first with `make` (or the per-project
# `pixi run build` / `pixi run instantiate`).
set -uo pipefail
cd "$(dirname "$0")"

status=0

echo "=== C++/CUDA ==="
if [ -x c++/cmy_right ]; then
	(cd c++ && pixi run --frozen ./cmy_right) || status=1
else
	echo "skipped: c++/cmy_right not built (cd c++ && pixi run build)"
	status=1
fi

echo
echo "=== Python ==="
.venv/bin/python python/cmy.py || status=1

echo
echo "=== Julia ==="
(cd julia && pixi run --frozen julia --project=. -t auto cmy.jl) || status=1

exit $status
