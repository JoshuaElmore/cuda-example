from concurrent.futures import ProcessPoolExecutor
import time
from PIL import Image
import numpy as np
import cupy as cp

THREADS = 16

WRITE_FILE = False
FILE = 'big.jpg'
OUT_FILE = 'big_out.jpg'

def rgb_to_cmy(rgb:tuple[int,int,int]) -> tuple[int,int,int]:
    out = np.array([
        255 - rgb[0],
        255 - rgb[1],
        255 - rgb[2]
    ], dtype=np.uint8)
    return out


def cpu_serial():
    im: Image = Image.open(FILE)
    im_np = np.array(im)

    height, width , _ = im_np.shape
    
    im_np_out = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            im_np_out[i][j] = rgb_to_cmy(im_np[i][j])

    im_out = Image.fromarray(im_np_out)
    if (WRITE_FILE):
        im_out.save(OUT_FILE)
    
    

def _process_rows(rows: np.ndarray) -> np.ndarray:
    out = np.zeros(rows.shape[:2] + (3,), dtype=np.uint8)
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            out[i][j] = rgb_to_cmy(rows[i][j])
    return out


def cpu_parallel():
    im: Image = Image.open(FILE)
    im_np = np.array(im)

    num_workers = THREADS
    chunks = np.array_split(im_np, num_workers, axis=0)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_rows, chunks))

    im_np_out = np.concatenate(results, axis=0)

    im_out = Image.fromarray(im_np_out)
    if (WRITE_FILE):
        im_out.save(OUT_FILE)

def _process_rows_gpu_wrong(rows: np.ndarray) -> np.ndarray:
    rows_gpu = cp.asarray(rows)
    rows_cpu = cp.asnumpy(rows_gpu)

    out = np.zeros(rows_cpu.shape[:2] + (3,), dtype=np.uint8)
    for i in range(rows_cpu.shape[0]):
        for j in range(rows_cpu.shape[1]):
            out[i][j] = rgb_to_cmy(rows_cpu[i][j])
    return out


def gpu_parallel_wrong():
    im: Image = Image.open(FILE)
    im_np = np.array(im)

    num_workers = THREADS
    chunks = np.array_split(im_np, num_workers, axis=0)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_rows_gpu_wrong, chunks))

    im_np_out = np.concatenate(results, axis=0)

    im_out = Image.fromarray(im_np_out)
    if (WRITE_FILE):
        im_out.save(OUT_FILE)

def gpu_parallel_right():
    im: Image = Image.open(FILE)
    im_np = np.array(im)

    im_gpu = cp.asarray(im_np)
    im_gpu_out = 255 - im_gpu
    im_np_out = cp.asnumpy(im_gpu_out)

    im_out = Image.fromarray(im_np_out)
    if (WRITE_FILE):
        im_out.save(OUT_FILE)  


def main():

    start = time.perf_counter()
    cpu_serial()
    print(f"CPU serial: {time.perf_counter() - start:.4f}s")

    start = time.perf_counter()
    cpu_parallel()
    print(f"CPU parallel: {time.perf_counter() - start:.4f}s")

    start = time.perf_counter()
    gpu_parallel_wrong()
    print(f"GPU parallel (wrong): {time.perf_counter() - start:.4f}s")

    start = time.perf_counter()
    gpu_parallel_right()
    print(f"GPU parallel (right): {time.perf_counter() - start:.4f}s")


if __name__ == "__main__":
    main()