using Images
using FileIO
using CUDA

const THREADS = 16

const WRITE_FILE = true
const FILE = joinpath(@__DIR__, "..", "big.jpg")
const OUT_FILE = joinpath(@__DIR__, "..", "big_out.jpg")

rgb_to_cmy(r::UInt8, g::UInt8, b::UInt8) = (0xff - r, 0xff - g, 0xff - b)

function load_raw(path)
    img = load(path)                          # H x W Array{RGB{N0f8}}
    return collect(rawview(channelview(img)))  # materialized 3 x H x W UInt8 array
end

function save_raw(raw::AbstractArray{UInt8,3}, path)
    img = colorview(RGB, normedview(raw))
    save(path, img)
end

function cpu_serial()
    raw = load_raw(FILE)
    _, height, width = size(raw)
    out = Array{UInt8}(undef, size(raw))

    for i in 1:height
        for j in 1:width
            out[1, i, j], out[2, i, j], out[3, i, j] =
                rgb_to_cmy(raw[1, i, j], raw[2, i, j], raw[3, i, j])
        end
    end

    WRITE_FILE && save_raw(out, OUT_FILE)
end

function _process_rows!(out, raw, rows)
    for i in rows
        for j in 1:size(raw, 3)
            out[1, i, j], out[2, i, j], out[3, i, j] =
                rgb_to_cmy(raw[1, i, j], raw[2, i, j], raw[3, i, j])
        end
    end
end

function cpu_parallel()
    raw = load_raw(FILE)
    _, height, width = size(raw)
    out = Array{UInt8}(undef, size(raw))

    num_workers = min(THREADS, Threads.nthreads())
    chunks = Iterators.partition(1:height, cld(height, num_workers))

    Threads.@threads for rows in collect(chunks)
        _process_rows!(out, raw, rows)
    end

    WRITE_FILE && save_raw(out, OUT_FILE)
end

# GPU "wrong": ships each row chunk to the GPU and immediately back, then
# still does the actual conversion element-by-element on the CPU. The
# transfer overhead dominates and the GPU does no real work.
function _process_rows_gpu_wrong!(out, raw, rows)
    chunk_cpu = Array(CuArray(raw[:, rows, :]))
    for (k, i) in enumerate(rows)
        for j in 1:size(raw, 3)
            out[1, i, j], out[2, i, j], out[3, i, j] =
                rgb_to_cmy(chunk_cpu[1, k, j], chunk_cpu[2, k, j], chunk_cpu[3, k, j])
        end
    end
end

function gpu_parallel_wrong()
    raw = load_raw(FILE)
    _, height, width = size(raw)
    out = Array{UInt8}(undef, size(raw))

    num_workers = min(THREADS, Threads.nthreads())
    chunks = Iterators.partition(1:height, cld(height, num_workers))

    Threads.@threads for rows in collect(chunks)
        _process_rows_gpu_wrong!(out, raw, rows)
    end

    WRITE_FILE && save_raw(out, OUT_FILE)
end

function gpu_parallel_right()
    raw = load_raw(FILE)

    raw_gpu = CuArray(raw)
    out_gpu = 0xff .- raw_gpu
    out = Array(out_gpu)

    WRITE_FILE && save_raw(out, OUT_FILE)
end

function warmup_gpu()
    dummy = CuArray(UInt8[0, 0])
    Array(0xff .- dummy)
    CUDA.synchronize()
end

function main()
    warmup_gpu() # forces one-time kernel JIT compilation before timing starts

    start = time()
    cpu_serial()
    println("CPU serial: $(round(time() - start, digits=4))s")

    start = time()
    cpu_parallel()
    println("CPU parallel ($(min(THREADS, Threads.nthreads())) threads): $(round(time() - start, digits=4))s")

    start = time()
    gpu_parallel_wrong()
    println("GPU parallel (wrong): $(round(time() - start, digits=4))s")

    start = time()
    gpu_parallel_right()
    println("GPU parallel (right): $(round(time() - start, digits=4))s")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
