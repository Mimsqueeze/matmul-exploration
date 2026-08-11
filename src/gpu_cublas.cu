#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// same idea, for cublas status codes
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = (call);                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__,    \
                    __LINE__, (int)status);                                    \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

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

    // handle creation/destruction is a one-time library setup cost, not part of
    // a single GEMM call, so it's kept outside the timed region like context
    // creation is for the other files.
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS's first call for a given problem pays a one-time cost to run its
    // internal heuristics and pick the best available kernel implementation.
    // Do a throwaway warm-up GEMM on separate small buffers here so that cost
    // lands here instead of inside the timed measurement below.
    float *d_warmupA, *d_warmupB, *d_warmupC;
    const int warmupSize = 256;
    CUDA_CHECK(cudaMalloc(&d_warmupA, warmupSize * warmupSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_warmupB, warmupSize * warmupSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_warmupC, warmupSize * warmupSize * sizeof(float)));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              warmupSize, warmupSize, warmupSize,
                              &alpha,
                              d_warmupB, warmupSize,
                              d_warmupA, warmupSize,
                              &beta,
                              d_warmupC, warmupSize));
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_warmupA);
    cudaFree(d_warmupB);
    cudaFree(d_warmupC);

    // events to time the GPU round trip: allocation -> H2D copy -> gemm -> D2H copy -> dealloc
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

    // cuBLAS assumes column-major storage, but our matrices are row-major. The
    // standard trick: row-major C[m x p] = A[m x n] * B[n x p] is the same memory
    // as column-major C^T[p x m] = B^T[p x n] * A^T[n x m], and reinterpreting a
    // row-major buffer as column-major IS its transpose -- so we can pass our
    // row-major A/B/C buffers directly (untransposed, CUBLAS_OP_N) as long as we
    // swap the operand order (B before A) and swap the M/N dimensions accordingly.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              p, m, n,
                              &alpha,
                              d_b, p,
                              d_a, n,
                              &beta,
                              d_c, p));
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

    printf("GPU time (alloc + H2D copy + cublasSgemm + D2H copy + dealloc): %f ms\n", elapsedMs);

    CUBLAS_CHECK(cublasDestroy(handle));

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
