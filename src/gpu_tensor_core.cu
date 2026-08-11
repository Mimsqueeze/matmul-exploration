#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// general macro to check cuda errors
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

// tf32 WMMA tile shape. This is the only shape NVIDIA's tensor cores support
// for tf32 (fp16 uses 16x16x16 instead) -- requires compute capability >= 8.0
// (Ampere or newer, e.g. RTX 30-series).
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

// each warp is 32 threads laid out along x; blockDim.x/32 warps stack along x,
// blockDim.y warps stack along y.
#define WARPS_PER_BLOCK_X 4
#define WARPS_PER_BLOCK_Y 4

// each warp now computes a WARP_TILE_M x WARP_TILE_N grid of WMMA tiles instead
// of just one -- the tensor-core analog of THREAD_TILE in gpu_register_blocked.cu.
// Each aFrag/bFrag loaded from shared memory gets reused across the other tile
// dimension (WARP_TILE_N times for aFrag, WARP_TILE_M times for bFrag) instead of
// being used for a single mma_sync and discarded.
#define WARP_TILE_M 2
#define WARP_TILE_N 2

// a block covers BLOCK_M x BLOCK_N output elements: WARPS_PER_BLOCK_(X|Y) warps,
// each covering WARP_TILE_(M|N) WMMA tiles of WMMA_(M|N) elements.
#define BLOCK_M (WMMA_M * WARPS_PER_BLOCK_X * WARP_TILE_M) // 128
#define BLOCK_N (WMMA_N * WARPS_PER_BLOCK_Y * WARP_TILE_N) // 128

// how much of the shared/reduction dimension gets staged into shared memory per
// outer step. Must be a multiple of WMMA_K; bigger means fewer, larger shared
// memory loads at the cost of more shared memory per block.
#define BLOCK_K 32

// tensor cores don't operate on plain fp32 -- round each element down to tf32
// (same 4-byte storage, truncated mantissa) before the GEMM kernel can use it.
__global__ void floatToTf32(const float* in, float* out, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        out[idx] = wmma::__float_to_tf32(in[idx]);
    }
}

// tensor-core matmul: C[m x p] = A[m x n] * B[n x p], all row-major.
// Requires m and p to be multiples of BLOCK_M/BLOCK_N and n a multiple of
// BLOCK_K -- no boundary handling here, unlike the earlier kernels.
//
// Block-level shared memory tiling: all WARPS_PER_BLOCK_X*WARPS_PER_BLOCK_Y warps
// in a block cooperatively stage a BLOCK_M x BLOCK_K slice of A and a BLOCK_K x
// BLOCK_N slice of B into shared memory once per outer step, then every warp reads
// its fragments from that shared tile instead of hitting global memory directly.
// On top of that, each warp computes WARP_TILE_M x WARP_TILE_N output tiles instead
// of one: aFrag[wm] is loaded once and reused across all WARP_TILE_N bFrags, and
// bFrag[wn] is loaded once and reused across all WARP_TILE_M aFrags, the same
// load-once-reuse-many pattern gpu_register_blocked.cu uses for scalar registers.
__global__ void matMulTensorCore(const float* a, const float* b, float* c, int m, int n, int p) {
    __shared__ float tileA[BLOCK_M][BLOCK_K];
    __shared__ float tileB[BLOCK_K][BLOCK_N];

    int blockRow = blockIdx.x * BLOCK_M; // top-left row of this block's output tile
    int blockCol = blockIdx.y * BLOCK_N; // top-left col of this block's output tile

    // base offset, within the block, of this warp's WARP_TILE_M x WARP_TILE_N grid of tiles
    int localWarpRow = (threadIdx.x / warpSize) * (WARP_TILE_M * WMMA_M);
    int localWarpCol = threadIdx.y * (WARP_TILE_N * WMMA_N);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> aFrag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> bFrag[WARP_TILE_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag[WARP_TILE_M][WARP_TILE_N];
    for (int wm = 0; wm < WARP_TILE_M; wm++) {
        for (int wn = 0; wn < WARP_TILE_N; wn++) {
            wmma::fill_fragment(cFrag[wm][wn], 0.0f);
        }
    }

    for (int t = 0; t < n; t += BLOCK_K) {
        // cooperatively load the BLOCK_M x BLOCK_K slice of A
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += numThreads) {
            int r = idx / BLOCK_K;
            int c = idx % BLOCK_K;
            tileA[r][c] = a[(blockRow + r) * n + (t + c)];
        }
        // cooperatively load the BLOCK_K x BLOCK_N slice of B
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += numThreads) {
            int r = idx / BLOCK_N;
            int c = idx % BLOCK_N;
            tileB[r][c] = b[(t + r) * p + (blockCol + c)];
        }

        __syncthreads(); // wait for the whole tile to be loaded before any warp reads it

        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            for (int wm = 0; wm < WARP_TILE_M; wm++) {
                wmma::load_matrix_sync(aFrag[wm], &tileA[localWarpRow + wm * WMMA_M][kk], BLOCK_K);
            }
            for (int wn = 0; wn < WARP_TILE_N; wn++) {
                wmma::load_matrix_sync(bFrag[wn], &tileB[kk][localWarpCol + wn * WMMA_N], BLOCK_N);
            }
            for (int wm = 0; wm < WARP_TILE_M; wm++) {
                for (int wn = 0; wn < WARP_TILE_N; wn++) {
                    wmma::mma_sync(cFrag[wm][wn], aFrag[wm], bFrag[wn], cFrag[wm][wn]);
                }
            }
        }

        __syncthreads(); // wait for all warps to finish reading before the next iteration overwrites the tile
    }

    for (int wm = 0; wm < WARP_TILE_M; wm++) {
        for (int wn = 0; wn < WARP_TILE_N; wn++) {
            int row = blockRow + localWarpRow + wm * WMMA_M;
            int col = blockCol + localWarpCol + wn * WMMA_N;
            wmma::store_matrix_sync(c + row * p + col, cFrag[wm][wn], p, wmma::mem_row_major);
        }
    }
}


int main() {
    const int m = 4096;
    const int n = 4096;
    const int p = 4096;

    const size_t A_bytes = m * n * sizeof(float);
    const size_t B_bytes = n * p * sizeof(float);
    const size_t C_bytes = m * p * sizeof(float);

    // allocate matrices on host (row major). Pinned (page-locked) memory instead of
    // malloc -- cudaMemcpyAsync only actually overlaps with kernel execution when
    // the host buffer is pinned; with regular pageable memory the driver silently
    // stages through a pinned bounce buffer and the "async" copy behaves synchronously.
    float* h_a;
    float* h_b;
    float* h_c;
    CUDA_CHECK(cudaHostAlloc(&h_a, A_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_b, B_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_c, C_bytes, cudaHostAllocDefault));

    // fill the matrices with values
    for (int i = 0; i < m * n; i++) {
        h_a[i] = static_cast<float>(i);
    }
    for (int i = 0; i < n * p; i++) {
        h_b[i] = static_cast<float>(2 * i);
    }

    // events to time the GPU round trip: allocation -> H2D copy -> tf32 conversion -> kernel -> D2H copy -> dealloc
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    // device allocations, cuda check to ensure no errors
    float *d_a, *d_b, *d_c, *d_a_tf32, *d_b_tf32;
    CUDA_CHECK(cudaMalloc(&d_a, A_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, B_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, C_bytes));
    CUDA_CHECK(cudaMalloc(&d_a_tf32, A_bytes));
    CUDA_CHECK(cudaMalloc(&d_b_tf32, B_bytes));

    // two streams so B's H2D copy can run concurrently with A's tf32 conversion
    // (and vice versa) instead of the two operands being handled strictly serially.
    cudaStream_t streamA, streamB;
    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));

    const int convertThreads = 256;
    int aElements = m * n;
    int bElements = n * p;

    // each stream: copy this operand to the device, then convert it to tf32 --
    // ordered within the stream, but streamA and streamB can run concurrently
    // with each other since the GPU has separate copy and compute engines.
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, A_bytes, cudaMemcpyHostToDevice, streamA));
    floatToTf32<<<(aElements + convertThreads - 1) / convertThreads, convertThreads, 0, streamA>>>(d_a, d_a_tf32, aElements);

    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, B_bytes, cudaMemcpyHostToDevice, streamB));
    floatToTf32<<<(bElements + convertThreads - 1) / convertThreads, convertThreads, 0, streamB>>>(d_b, d_b_tf32, bElements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(streamA));
    CUDA_CHECK(cudaStreamSynchronize(streamB));
    CUDA_CHECK(cudaStreamDestroy(streamA));
    CUDA_CHECK(cudaStreamDestroy(streamB));

    // initialize the kernel: each warp computes a WARP_TILE_M x WARP_TILE_N grid of tiles
    dim3 threadsPerBlock(32 * WARPS_PER_BLOCK_X, WARPS_PER_BLOCK_Y);
    dim3 blocksPerGrid(m / BLOCK_M, p / BLOCK_N);
    matMulTensorCore<<<blocksPerGrid, threadsPerBlock>>>(d_a_tf32, d_b_tf32, d_c, m, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, C_bytes, cudaMemcpyDeviceToHost));

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_a_tf32);
    cudaFree(d_b_tf32);

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    printf("GPU time (alloc + H2D copy + tf32 convert + kernel + D2H copy + dealloc): %f ms\n", elapsedMs);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
