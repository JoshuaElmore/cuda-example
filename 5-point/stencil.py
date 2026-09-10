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

Timing note: the image is loaded once *outside* the timed region. Decoding a
73 MP JPEG costs more than the stencil itself and is identical for both paths.
The GPU path does time its host<->device transfers -- that is the point.

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

WRITE_FILE = True
ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "big.jpg"
OUT_FILE = ROOT / "big_out_stencil.png"


def load_image() -> np.ndarray:
    return np.asarray(Image.open(FILE).convert("RGB"), dtype=np.float32)


def save_image(a: np.ndarray) -> None:
    if WRITE_FILE:
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(OUT_FILE)


def _sweep(a, b, lo: int, hi: int) -> None:
    """One Jacobi sweep over output rows [lo, hi), reading neighbours from a.
    """
    b[lo:hi, 1:-1] = 0.2 * (a[lo:hi, 1:-1] + a[lo - 1:hi - 1, 1:-1]
                            + a[lo + 1:hi + 1, 1:-1]
                            + a[lo:hi, :-2] + a[lo:hi, 2:])


def _row_bands(height: int, workers: int) -> list[tuple[int, int]]:
    """Split the interior rows [1, height-1) into contiguous, non-empty bands."""
    bounds = np.linspace(1, height - 1, workers + 1).round().astype(int)
    return [(int(lo), int(hi)) for lo, hi in zip(bounds, bounds[1:]) if lo < hi]


def cpu_parallel(src: np.ndarray) -> np.ndarray:
    a, b = src.copy(), src.copy()
    bands = _row_bands(src.shape[0], THREADS)
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for _ in range(ITERATIONS):
            list(executor.map(lambda band: _sweep(a, b, *band), bands))
            a, b = b, a
    save_image(a)
    return a


def gpu_parallel(src: np.ndarray) -> np.ndarray:
    a = cp.asarray(src) #asarray chunked transfer to VRAM
    b = a.copy()
    for _ in range(ITERATIONS):
        _sweep(a, b, 1, src.shape[0] - 1)
        a, b = b, a
    out = cp.asnumpy(a) #asnumpy chunked transfer to system RAM
    cp.cuda.runtime.deviceSynchronize()
    save_image(out)
    return out


def warmup_gpu() -> None:
    """Pay CUDA context init and kernel compilation before anything is timed."""
    a = cp.zeros((8, 8, 3), dtype=cp.float32)
    b = a.copy()
    _sweep(a, b, 1, 7)
    cp.asnumpy(b)
    cp.cuda.runtime.deviceSynchronize()


def main() -> None:
    src = load_image()
    height, width, _ = src.shape
    print(f"{width}x{height} ({width * height / 1e6:.2f} MP), "
          f"{ITERATIONS} iterations, {THREADS} threads")

    warmup_gpu()

    start = time.perf_counter()
    cpu_out = cpu_parallel(src)
    print(f"CPU parallel: {time.perf_counter() - start:.4f}s")

    start = time.perf_counter()
    gpu_out = gpu_parallel(src)
    print(f"GPU parallel: {time.perf_counter() - start:.4f}s")

    print(f"max |CPU - GPU|: {np.abs(cpu_out - gpu_out).max():.3e}")


if __name__ == "__main__":
    main()
