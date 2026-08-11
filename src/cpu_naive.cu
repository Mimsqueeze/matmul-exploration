#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

int main() {
    const int m = 1024;
    const int n = 1024;
    const int p = 1024;

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

    // compute results
    bool ok = true;
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < m; i++) { // C row index, A row index
        for (int j = 0; j < p; j++) { // C column index, B column index
            // Note, C[i][j] = SUM(A[i,:] * B[:,j])
            float sum = 0.0f;
            for (int k = 0; k < n; k++) { // A column index and B row index
                sum += h_a[i*n + k] * h_b[k*p + j];
            }
            h_c[i*p + j] = sum;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    printf("CPU time (compute only): %f ms\n", elapsedMs);

    free(h_a);
    free(h_b);
    free(h_c);

    return ok ? 0 : 1;
}
