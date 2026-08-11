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

#define TILE_SIZE 16

// tiled matmul: each block cooperatively loads TILE_SIZE x TILE_SIZE tiles of
// a and b into shared memory and reuses them across all threads in the block,
// instead of every thread re-reading the same rows/columns from global memory.
__global__ void matMulTiled(const float* a, const float* b, float* c, int m, int n, int p) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x; // column within a's tile
        int bRow = t * TILE_SIZE + threadIdx.y; // row within b's tile

        // each thread loads one element per matrix into shared memory (zero-padded past the edge)
        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < n) ? a[row * n + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < n && col < p) ? b[bRow * p + col] : 0.0f;

        __syncthreads(); // wait for the whole tile to be loaded before reading it

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads(); // wait for all reads before the next iteration overwrites the tile
    }

    if (row < m && col < p) {
        c[row * p + col] = sum;
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

    // events to time the GPU round trip: allocation -> H2D copy -> kernel -> D2H copy
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

    // initialize the 2D kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE); // 256 threads per block
    dim3 blocksPerGrid((p + TILE_SIZE - 1) / TILE_SIZE,  // x -> columns of C/B
                        (m + TILE_SIZE - 1) / TILE_SIZE); // y -> rows of C/A
    matMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);
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

