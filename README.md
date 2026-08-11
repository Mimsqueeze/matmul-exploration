# matmul-exploration

analysis of different matrix multiplication algorithms and their efficiency

use `./run.ps1 [filename]` to test a CUDA script from src

## Results

### CPU (1024 x 1024 x 1024)

Timed region: the matrix multiply itself only (allocation, fill, and printing are excluded).

| Implementation | Time | Speedup vs. naive |
|---|---|---|
| [cpu_naive.cu](src/cpu_naive.cu) | 7182.41 ms | 1x |
| [cpu_divide_and_conquer.cu](src/cpu_divide_and_conquer.cu) | 3096.96 ms | 2.3x |
| [cpu_strassen.cu](src/cpu_strassen.cu) | 1826.67 ms | 3.9x |
| [cpu_winograd_strassen.cu](src/cpu_winograd_strassen.cu) | 1831.64 ms | 3.9x |

### GPU (4096 x 4096 x 4096)

Timed region: device allocation -> host-to-device copy -> compute -> device-to-host copy -> device deallocation (via CUDA events).

| Implementation | Time | Speedup vs. naive |
|---|---|---|
| [gpu_naive.cu](src/gpu_naive.cu) | 517.13 ms | 1x |
| [gpu_tiled.cu](src/gpu_tiled.cu) | 121.76 ms | 4.2x |
| [gpu_register_blocked.cu](src/gpu_register_blocked.cu) | 50.51 ms | 10.2x |
| [gpu_tensor_core.cu](src/gpu_tensor_core.cu) | 43.94 ms | 11.8x |
| [gpu_cublas.cu](src/gpu_cublas.cu) (warmed up) | 39.65 ms | 13.0x |

cuBLAS's first `cublasSgemm` call for a given problem shape pays a one-time cost for its internal kernel-selection heuristics. `gpu_cublas.cu` runs a throwaway warm-up GEMM on separate small buffers before the timed region so that cost isn't included in the measurement; the "cold" number (no warm-up) comes in around 105 ms.

### Notes

- CPU and GPU results use different matrix sizes (1024 vs. 4096) and are not directly comparable to each other -- each table should be read as a comparison within its own hardware.
- All GPU implementations were run on an RTX 3080 (Ampere, compute capability 8.6, `-arch=sm_86`).
- `gpu_tensor_core.cu` initially trailed `gpu_register_blocked.cu` despite tensor cores having far higher raw throughput than CUDA cores, because its shared-memory reuse factor was much lower (~4x, one WMMA tile per warp) than register-blocked's combined shared-memory + register reuse, and it pays an extra fp32->tf32 conversion pass (~256 MB of traffic) the CUDA-core kernels don't. Two fixes closed the gap: (1) each warp now computes a `WARP_TILE_M x WARP_TILE_N = 2x2` grid of WMMA tiles instead of one, reusing each loaded fragment across the other tile dimension; (2) the tf32 conversion now runs on a separate CUDA stream per operand, overlapping A's conversion with B's host-to-device copy (and vice versa) using pinned host memory. Together these took it from 76.2 ms down to 43.9 ms, ahead of `gpu_register_blocked.cu`.
