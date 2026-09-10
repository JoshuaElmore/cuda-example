#!/usr/bin/env bash

# Needs a visible GPU. On a cluster:
# srun --gres=gpu:a40:1 --cpus-per-task=16 --partition=gpupriority --account=priority-rci --pty ./run_all.sh

set -uo pipefail
cd "$(dirname "$0")"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${SCRATCH:-$HOME/scratch}/julia-depot}"
mkdir -p "${JULIA_DEPOT_PATH%%:*}"

status=0
DEPS=.venv/.deps-installed

echo "=== C++/CUDA ==="
if [ ! -x c++/cmy_right ]; then
	echo "compiling cmy_right ..."
	(cd c++ && pixi run --frozen build) || status=1
fi
if [ -x c++/cmy_right ]; then
	(cd c++ && pixi run --frozen ./cmy_right) || status=1
else
	echo "skipped: build failed"
	status=1
fi

echo
echo "=== Python ==="
if [ ! -x .venv/bin/python ] || [ python/requirments.txt -nt "$DEPS" ]; then
	echo "setting up .venv (first run downloads the CUDA wheels, ~2GB) ..."
	if [ ! -x .venv/bin/python ]; then
		"${PYTHON:-python3}" -m venv .venv
	fi
	if [ -x .venv/bin/python ]; then
		.venv/bin/python -m pip install --quiet --upgrade pip \
			&& .venv/bin/python -m pip install --quiet -r python/requirments.txt \
			&& touch "$DEPS"
	fi
fi
if [ -f "$DEPS" ]; then
	.venv/bin/python python/cmy.py || status=1
else
	echo "skipped: .venv setup failed"
	status=1
fi

echo
echo "=== Julia ==="
(cd julia && pixi run --frozen run) || status=1

exit $status
