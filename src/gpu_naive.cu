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

// kernel to multiply two matrices stored as flat arrays
__global__ void matMul(const float* a, const float* b, float* c, int m, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col
    if (i < m && j < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[i*n + k] * b[k*p + j]; // reading from global memory, slow!
        }
        c[i*p + j] = sum;
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
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (p + threadsPerBlock.y - 1) / threadsPerBlock.y); // number of blocks
    matMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);
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

