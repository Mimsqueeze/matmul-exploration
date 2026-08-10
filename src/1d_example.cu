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

// kernal to add two entries
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20; // vector with ~1M elements

    const size_t bytes = n * sizeof(float); // size to allocate in bytes

    // allocate vectors on host
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // fill the vectors with values
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i); // static cast int i into a float
        h_b[i] = static_cast<float>(3 * i);
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

    // initialize the 1D kernel
    const int threadsPerBlock = 256; // 256 threads per block
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // compute number of blocks needed
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n); // run kernel
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back to the host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // verify a few results
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            ok = false;
            printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            break;
        }
    }
    printf(ok ? "Vector addition succeeded.\n" : "Vector addition FAILED.\n");
    printf("GPU time (H2D copy + kernel + D2H copy): %f ms\n", elapsedMs);

    // print the first few elements of A, B, and C
    const int printCount = 10;

    printf("A (first %d elements):\n", printCount);
    for (int i = 0; i < printCount; ++i) {
        printf("%8.1f ", h_a[i]);
    }
    printf("\n");

    printf("B (first %d elements):\n", printCount);
    for (int i = 0; i < printCount; ++i) {
        printf("%8.1f ", h_b[i]);
    }
    printf("\n");

    printf("C (first %d elements):\n", printCount);
    for (int i = 0; i < printCount; ++i) {
        printf("%8.1f ", h_c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return ok ? 0 : 1;
}
