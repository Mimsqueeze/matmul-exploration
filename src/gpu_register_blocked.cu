#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

#define BLOCK_SIZE 64  // each block computes a BLOCK_SIZE x BLOCK_SIZE tile of C
#define TILE_K 16      // shared-dimension chunk size loaded into shared memory per step
#define THREAD_TILE 4  // each thread computes a THREAD_TILE x THREAD_TILE sub-tile of C

// register-blocked tiled matmul: same shared-memory tiling as gpu_tiled.cu, but each
// thread now computes a THREAD_TILE x THREAD_TILE block of output elements instead of
// just one. Values loaded from shared memory are staged into registers (regA/regB) and
// reused THREAD_TILE times each, cutting shared memory reads by a factor of THREAD_TILE
// on top of the global memory savings tiling already gave us.
__global__ void matMulRegisterBlocked(const float* a, const float* b, float* c, int m, int n, int p) {
    __shared__ float tileA[BLOCK_SIZE][TILE_K];
    __shared__ float tileB[TILE_K][BLOCK_SIZE];

    int blockRow = blockIdx.y * BLOCK_SIZE; // top-left row of this block's output tile
    int blockCol = blockIdx.x * BLOCK_SIZE; // top-left col of this block's output tile
    int threadRow = threadIdx.y * THREAD_TILE; // this thread's sub-tile offset within the block
    int threadCol = threadIdx.x * THREAD_TILE;

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // flattened thread index, for cooperative loads
    int numThreads = blockDim.x * blockDim.y;

    float acc[THREAD_TILE][THREAD_TILE] = {0.0f}; // per-thread output accumulators, live in registers

    int numTiles = (n + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        // cooperatively load the BLOCK_SIZE x TILE_K slice of A: 1024 elements / 256 threads = 4 each
        for (int idx = tid; idx < BLOCK_SIZE * TILE_K; idx += numThreads) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int globalRow = blockRow + r;
            int globalCol = t * TILE_K + c;
            tileA[r][c] = (globalRow < m && globalCol < n) ? a[globalRow * n + globalCol] : 0.0f;
        }
        // cooperatively load the TILE_K x BLOCK_SIZE slice of B
        for (int idx = tid; idx < TILE_K * BLOCK_SIZE; idx += numThreads) {
            int r = idx / BLOCK_SIZE;
            int c = idx % BLOCK_SIZE;
            int globalRow = t * TILE_K + r;
            int globalCol = blockCol + c;
            tileB[r][c] = (globalRow < n && globalCol < p) ? b[globalRow * p + globalCol] : 0.0f;
        }

        __syncthreads(); // wait for the whole tile to be loaded before reading it

        for (int k = 0; k < TILE_K; k++) {
            float regA[THREAD_TILE];
            float regB[THREAD_TILE];
            for (int i = 0; i < THREAD_TILE; i++) {
                regA[i] = tileA[threadRow + i][k];
            }
            for (int j = 0; j < THREAD_TILE; j++) {
                regB[j] = tileB[k][threadCol + j];
            }
            // each regA[i]/regB[j] read once from shared memory, reused THREAD_TILE times here
            for (int i = 0; i < THREAD_TILE; i++) {
                for (int j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads(); // wait for all reads before the next iteration overwrites the tile
    }

    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            int globalRow = blockRow + threadRow + i;
            int globalCol = blockCol + threadCol + j;
            if (globalRow < m && globalCol < p) {
                c[globalRow * p + globalCol] = acc[i][j];
            }
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

    // allocate matrices on host (row major)
    float* h_a = (float*)malloc(A_bytes);
    float* h_b = (float*)malloc(B_bytes);
    float* h_c = (float*)malloc(C_bytes);

    // fill the matrices with values
    for (int i = 0; i < m * n; i++) {
        h_a[i] = static_cast<float>(i);
    }
    for (int i = 0; i < n * p; i++) {
        h_b[i] = static_cast<float>(2 * i);
    }

    // events to time the GPU round trip: allocation -> H2D copy -> kernel -> D2H copy -> dealloc
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    // device allocations, cuda check to ensure no errors
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, A_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, B_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, C_bytes));

    // copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, A_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, B_bytes, cudaMemcpyHostToDevice));

    // initialize the 2D kernel: one thread per THREAD_TILE x THREAD_TILE sub-tile of output
    dim3 threadsPerBlock(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE); // 16x16 = 256 threads
    dim3 blocksPerGrid((p + BLOCK_SIZE - 1) / BLOCK_SIZE,  // x -> columns of C/B
                        (m + BLOCK_SIZE - 1) / BLOCK_SIZE); // y -> rows of C/A
    matMulRegisterBlocked<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, C_bytes, cudaMemcpyDeviceToHost));

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    printf("GPU time (alloc + H2D copy + kernel + D2H copy + dealloc): %f ms\n", elapsedMs);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
