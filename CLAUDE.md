# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A benchmarking exercise comparing CPU vs. GPU image processing across three languages: C++/CUDA, Python, and Julia. Each implementation loads `big.jpg` from the repo root and applies the same trivial per-pixel transform — CMY inversion (`255 - channel`) — using progressively different strategies, timing each one. The point of the exercise is the comparison itself: naive CPU loops vs. multiprocess/multithreaded CPU vs. a GPU approach that transfers data per-chunk ("wrong" — transfer overhead dominates) vs. a single bulk transfer + vectorized/kernel GPU pass ("right").

Each language directory is an independent, self-contained project (own dependency manifest, own lockfile) — there is no shared build system across `c++/`, `python/`, and `julia/`.

## Commands

Each subproject uses [pixi](https://pixi.sh) for environment + task management, except `python/`, which uses the root-level `.venv` (created with `uv`).

**C++/CUDA** (`c++/`):
```
pixi run build   # nvcc -O3 -std=c++17 -arch=native ... cmy_right.cu -o cmy_right
pixi run run     # builds, then runs ./cmy_right
```

**Julia** (`julia/`):
```
pixi run instantiate   # Pkg.instantiate() to resolve Project.toml/Manifest.toml
pixi run run           # julia --project=. -t auto cmy.jl
```

**Python** (`python/`):
```
# from repo root, using the existing .venv
.venv/bin/python python/cmy.py
```
Dependencies (`python/requirments.txt` — note the typo, matches the actual filename) are `Pillow`, `numpy`, `cupy-cuda13x[ctk]`; install with `.venv/bin/python -m pip install -r python/requirments.txt` if the venv needs rebuilding.

All three require an NVIDIA GPU with CUDA available to exercise the GPU code paths.

## Architecture notes

- **Mirrored structure across languages**: each implementation defines the same four functions/paths — `cpu_serial`, `cpu_parallel`, `gpu_parallel_wrong` (Python/Julia only), `gpu_parallel_right` — and prints a timing line for each. When changing behavior in one language's version (e.g. the transform, the chunking strategy, timing methodology), check whether the equivalent should change in the other two to keep the comparison meaningful. The C++ version currently only implements the "right" GPU path (`c++/cmy_right.cu`); it does not have CPU serial/parallel or "wrong" GPU variants.
- **The "wrong" vs. "right" GPU distinction is the core teaching point**: "wrong" (Python's `gpu_parallel_wrong`, Julia's `gpu_parallel_wrong`) ships data to the GPU and back per-chunk while still doing the actual per-pixel work on the CPU — transfer overhead dominates and the GPU does no real work. "right" (`gpu_parallel_right` in all three, and the only path in C++) does a single host→device transfer, a vectorized/kernel op on the whole buffer, and a single device→host transfer. Preserve this distinction; don't "fix" the wrong path to be efficient.
- **GPU warmup is excluded from timing** in both C++ (`warmup_gpu()`) and Julia (`warmup_gpu()`) to avoid attributing one-time CUDA context init / kernel JIT cost to the first real measurement. Python has no explicit warmup step. If adding timing-sensitive changes, preserve this separation.
- **CUDA kernel** (`c++/cmy_right.cu`): `rgb_to_cmy_kernel` is a flat 1-D kernel over the raw byte buffer (`width * height * channels`, not indexed by pixel/channel structure) — index math (`blockIdx.x * blockDim.x + threadIdx.x`) and grid sizing (`threads = 256`) follow from that flat layout.
- **Julia image layout**: `load_raw`/`save_raw` convert between `Images.jl`'s `H x W` array-of-`RGB` representation and a materialized `3 x H x W` `UInt8` array (channel-first) via `channelview`/`rawview` and `colorview`/`normedview`. GPU and CPU-parallel code operates on this channel-first raw array.
- **`c++/vendor/`** contains vendored single-header libraries (`stb_image.h`, `stb_image_write.h`) — used for JPEG/PNG I/O in the CUDA build. Don't replace with a package dependency without checking why they were vendored (likely to keep the pixi/conda dependency set minimal).
- **`WRITE_FILE` / output files**: all three implementations gate writing the transformed image to disk behind a `WRITE_FILE` constant (default `false`), since the exercise is about timing, not output. Output filenames differ per language (`big_out_cpp.png`, `big_out.jpg`, `big_out.jpg`) — check the constant near the top of each file before assuming output is written.
