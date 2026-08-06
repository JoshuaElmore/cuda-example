from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import numpy as np
import cupy as cp



def rgb_to_cmy(rgb:tuple[int,int,int]) -> tuple[int,int,int]:
    out = np.array([
        255 - rgb[0],
        255 - rgb[1],
        255 - rgb[2]
    ], dtype=np.uint8)
    return out


def cpu_serial(write_file):
    im: Image = Image.open('dogo.jpg')
    im_np = np.array(im)

    height, width , _ = im_np.shape
    
    im_np_out = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            im_np_out[i][j] = rgb_to_cmy(im_np[i][j])

    im_out = Image.fromarray(im_np_out)
    if (write_file):
        im_out.save('dogo_out.jpg')
    
    

def _process_rows(rows: np.ndarray) -> np.ndarray:
    out = np.zeros(rows.shape[:2] + (3,), dtype=np.uint8)
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            out[i][j] = rgb_to_cmy(rows[i][j])
    return out


def cpu_parallel(write_file):
    im: Image = Image.open('dogo.jpg')
    im_np = np.array(im)

    height, width, _ = im_np.shape

    num_workers = multiprocessing.cpu_count()
    chunks = np.array_split(im_np, num_workers, axis=0)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_rows, chunks))

    im_np_out = np.concatenate(results, axis=0)

    im_out = Image.fromarray(im_np_out)
    if (write_file):
        im_out.save('dogo_out.jpg')

def gpu_parallel_wrong(write_file):


def main():

    write_file = False

    start = time.perf_counter()
    cpu_serial(write_file)
    print(f"CPU serial: {time.perf_counter() - start:.4f}s")

    start = time.perf_counter()
    cpu_parallel(write_file)
    print(f"CPU parallel: {time.perf_counter() - start:.4f}s")


if __name__ == "__main__":
    main()