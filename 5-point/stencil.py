"""Five-point stencil (Jacobi smoothing) on an image: parallel CPU vs. GPU.

Each iteration replaces every interior pixel with the average of itself and its
four neighbours:

    out[i,j] = (in[i,j] + in[i-1,j] + in[i+1,j] + in[i,j-1] + in[i,j+1]) / 5

Boundary pixels are held fixed (Dirichlet), so both buffers start as a copy of
the source and only interiors are ever written.

Two paths:

  cpu_parallel  row bands across threads, numpy slicing within each band
  gpu_parallel  one host->device transfer, ITERATIONS sweeps, one transfer back

Running ITERATIONS sweeps is what makes the GPU worth it: a single pass would be
dominated by the transfer, but the cost of shipping the image over PCIe is paid
once and then amortized over every sweep.

Both paths accumulate into a preallocated scratch buffer with in-place ops. The
obvious spelling, `b[...] = 0.2 * (a[...] + a[...] + a[...] + a[...] + a[...])`,
allocates five full-size temporaries per sweep -- 4.1 GiB per sweep at this
image size, and 5.75 GiB of peak VRAM. Accumulating in place holds it to one
0.82 GiB buffer for the whole run.

Timing note: the image is loaded once *outside* the timed region. Decoding a
73 MP JPEG costs more than the stencil itself and is identical for both paths.
The GPU path does time its host<->device transfers -- that is the point, and it
reports them separately from the sweeps so the two costs stay distinguishable.

Run from anywhere (paths resolve relative to this file):
    .venv/bin/python 5-point/stencil.py
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from PIL import Image
import numpy as np
import cupy as cp

THREADS = 16
ITERATIONS = 10

WRITE_FILE = False
ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "big.jpg"
OUT_FILE = ROOT / "big_out_stencil.png"


def load_image() -> np.ndarray:
    return np.asarray(Image.open(FILE).convert("RGB"), dtype=np.float32)


def save_image(a: np.ndarray) -> None:
    if WRITE_FILE:
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(OUT_FILE)


def _sweep(a, b, lo: int, hi: int, scratch) -> None:
    """One Jacobi sweep over output rows [lo, hi), reading neighbours from a.

    Works on numpy and cupy arrays alike. Rows lo-1 and hi are read as halos, so
    lo >= 1 and hi <= height-1. `scratch` is (hi-lo, width-2, channels) and is
    reused across sweeps; accumulating into it keeps this allocation-free.
    """
    xp = cp if isinstance(a, cp.ndarray) else np
    xp.add(a[lo:hi, 1:-1], a[lo - 1:hi - 1, 1:-1], out=scratch)
    scratch += a[lo + 1:hi + 1, 1:-1]
    scratch += a[lo:hi, :-2]
    scratch += a[lo:hi, 2:]
    xp.multiply(scratch, 0.2, out=b[lo:hi, 1:-1])


def _row_bands(height: int, workers: int) -> list[tuple[int, int]]:
    """Split the interior rows [1, height-1) into contiguous, non-empty bands."""
    bounds = np.linspace(1, height - 1, workers + 1).round().astype(int)
    return [(int(lo), int(hi)) for lo, hi in zip(bounds, bounds[1:]) if lo < hi]


def cpu_parallel(src: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    a, b = src.copy(), src.copy()
    bands = _row_bands(src.shape[0], THREADS)
    # One scratch buffer per band, so threads never write to the same memory.
    scratch = [np.empty((hi - lo, src.shape[1] - 2, src.shape[2]), dtype=np.float32)
               for lo, hi in bands]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for _ in range(ITERATIONS):
            list(executor.map(lambda n: _sweep(a, b, *bands[n], scratch[n]),
                              range(len(bands))))
            a, b = b, a
    sweeps = time.perf_counter() - start

    save_image(a)
    return a, {"sweeps": sweeps}


def gpu_parallel(src: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    a = cp.asarray(src) #asarray chunked transfer to VRAM
    b = a.copy()
    scratch = cp.empty((src.shape[0] - 2, src.shape[1] - 2, src.shape[2]),
                       dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    h2d = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        _sweep(a, b, 1, src.shape[0] - 1, scratch)
        a, b = b, a
    cp.cuda.runtime.deviceSynchronize()
    sweeps = time.perf_counter() - start

    start = time.perf_counter()
    out = cp.asnumpy(a) #asnumpy chunked transfer to system RAM
    cp.cuda.runtime.deviceSynchronize()
    d2h = time.perf_counter() - start

    save_image(out)
    return out, {"h2d": h2d, "sweeps": sweeps, "d2h": d2h}


def warmup_gpu() -> None:
    """Pay CUDA context init and kernel compilation before anything is timed.

    Exercises the same ops as the real sweeps, so nothing JITs mid-measurement.
    """
    a = cp.zeros((8, 8, 3), dtype=cp.float32)
    b = a.copy()
    scratch = cp.empty((6, 6, 3), dtype=cp.float32)
    _sweep(a, b, 1, 7, scratch)
    cp.asnumpy(b)
    cp.cuda.runtime.deviceSynchronize()


def describe_gpu() -> None:
    """Print what cupy is actually talking to -- a silent CPU fallback, a shared
    device, or too little VRAM all show up here rather than as a mystery time."""
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"GPU: {name}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free, "
          f"cupy {cp.__version__}, CUDA {cp.cuda.runtime.runtimeGetVersion()}")


def main() -> None:
    src = load_image()
    height, width, channels = src.shape
    buf = height * width * channels * 4 / 2**30
    print(f"{width}x{height} ({width * height / 1e6:.2f} MP), "
          f"{ITERATIONS} iterations, {THREADS} threads, {buf:.2f} GiB/buffer")
    describe_gpu()

    warmup_gpu()

    cpu_out, cpu_t = cpu_parallel(src)
    print(f"CPU parallel: {cpu_t['sweeps']:.4f}s "
          f"({cpu_t['sweeps'] / ITERATIONS * 1e3:.1f} ms/sweep)")

    gpu_out, gpu_t = gpu_parallel(src)
    total = gpu_t["h2d"] + gpu_t["sweeps"] + gpu_t["d2h"]
    print(f"GPU parallel: {total:.4f}s "
          f"(h2d {gpu_t['h2d']:.4f}s, sweeps {gpu_t['sweeps']:.4f}s "
          f"= {gpu_t['sweeps'] / ITERATIONS * 1e3:.1f} ms/sweep, "
          f"d2h {gpu_t['d2h']:.4f}s)")

    print(f"max |CPU - GPU|: {np.abs(cpu_out - gpu_out).max():.3e}")


if __name__ == "__main__":
    main()
