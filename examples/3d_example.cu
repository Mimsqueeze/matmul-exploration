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

// kernel to add two tensors, stored as flat arrays with layout z*(dimY*dimX) + y*dimX + x
__global__ void tensorAdd(const float* a, const float* b, float* c, int dimX, int dimY, int dimZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dimX && y < dimY && z < dimZ) {
        int idx = z * (dimY * dimX) + y * dimX + x;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int dimX = 64;
    const int dimY = 64;
    const int dimZ = 64;
    const int n = dimX * dimY * dimZ;

    const size_t bytes = n * sizeof(float); // size to allocate in bytes

    // allocate tensors on host (flattened)
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // fill the tensors with values
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

    // initialize the 3D kernel
    dim3 threadsPerBlock(8, 8, 8); // 512 threads per block (max 1024 on most GPUs)
    dim3 blocksPerGrid((dimX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (dimY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                        (dimZ + threadsPerBlock.z - 1) / threadsPerBlock.z);
    tensorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, dimX, dimY, dimZ);
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
    printf(ok ? "Tensor addition succeeded.\n" : "Tensor addition FAILED.\n");
    printf("GPU time (H2D copy + kernel + D2H copy): %f ms\n", elapsedMs);

    // print a small corner block of A, B, and C, one z-slice at a time
    const int printX = 4;
    const int printY = 4;
    const int printZ = 2;

    printf("A (corner %dx%dx%d block):\n", printX, printY, printZ);
    for (int z = 0; z < printZ; ++z) {
        printf("z = %d:\n", z);
        for (int y = 0; y < printY; ++y) {
            for (int x = 0; x < printX; ++x) {
                printf("%8.1f ", h_a[z * (dimY * dimX) + y * dimX + x]);
            }
            printf("\n");
        }
    }

    printf("B (corner %dx%dx%d block):\n", printX, printY, printZ);
    for (int z = 0; z < printZ; ++z) {
        printf("z = %d:\n", z);
        for (int y = 0; y < printY; ++y) {
            for (int x = 0; x < printX; ++x) {
                printf("%8.1f ", h_b[z * (dimY * dimX) + y * dimX + x]);
            }
            printf("\n");
        }
    }

    printf("C (corner %dx%dx%d block):\n", printX, printY, printZ);
    for (int z = 0; z < printZ; ++z) {
        printf("z = %d:\n", z);
        for (int y = 0; y < printY; ++y) {
            for (int x = 0; x < printX; ++x) {
                printf("%8.1f ", h_c[z * (dimY * dimX) + y * dimX + x]);
            }
            printf("\n");
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return ok ? 0 : 1;
}
