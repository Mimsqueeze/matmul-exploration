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

// kernel to add two matrices, stored row-major as flat arrays
__global__ void matAdd(const float* a, const float* b, float* c, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int rows = 1024;
    const int cols = 1024;
    const int n = rows * cols;

    const size_t bytes = n * sizeof(float); // size to allocate in bytes

    // allocate matrices on host (row-major, flattened)
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // fill the matrices with values
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // device allocations, cuda check to ensure no errors
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // events to time the GPU round trip: H2D copy -> kernel -> D2H copy
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    // copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // initialize the 2D kernel
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // Verify a few results
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            ok = false;
            printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            break;
        }
    }
    printf(ok ? "Matrix addition succeeded.\n" : "Matrix addition FAILED.\n");
    printf("GPU time (H2D copy + kernel + D2H copy): %f ms\n", elapsedMs);

    // print the top-left corner of A, B, and C
    const int printRows = 5;
    const int printCols = 5;

    printf("A (top-left %dx%d corner):\n", printRows, printCols);
    for (int row = 0; row < printRows; ++row) {
        for (int col = 0; col < printCols; ++col) {
            printf("%8.1f ", h_a[row * cols + col]);
        }
        printf("\n");
    }

    printf("B (top-left %dx%d corner):\n", printRows, printCols);
    for (int row = 0; row < printRows; ++row) {
        for (int col = 0; col < printCols; ++col) {
            printf("%8.1f ", h_b[row * cols + col]);
        }
        printf("\n");
    }

    printf("C (top-left %dx%d corner):\n", printRows, printCols);
    for (int row = 0; row < printRows; ++row) {
        for (int col = 0; col < printCols; ++col) {
            printf("%8.1f ", h_c[row * cols + col]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return ok ? 0 : 1;
}
