#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

// base case size below which we stop recursing and just do the direct triple loop
const int BASE_CASE_SIZE = 64;

// recursive divide-and-conquer matmul: C[M x P] += A[M x N] * B[N x P]
// strideA/strideB/strideC are the row lengths of the *original* full matrices,
// since A/B/C here may point into a submatrix of a larger allocation.
// C must be zero-initialized before the first call, since N-splits accumulate into it.
void matMulRecursive(const float* A, const float* B, float* C,
                      int M, int N, int P,
                      int strideA, int strideB, int strideC) {
    if (M <= BASE_CASE_SIZE && N <= BASE_CASE_SIZE && P <= BASE_CASE_SIZE) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * strideA + k] * B[k * strideB + j];
                }
                C[i * strideC + j] += sum;
            }
        }
        return;
    }

    // split whichever dimension is currently largest
    if (M >= N && M >= P) {
        // split A's rows / C's rows: two independent halves, no accumulation needed
        int M1 = M / 2;
        matMulRecursive(A, B, C, M1, N, P, strideA, strideB, strideC);
        matMulRecursive(A + M1 * strideA, B, C + M1 * strideC, M - M1, N, P, strideA, strideB, strideC);
    } else if (P >= M && P >= N) {
        // split B's columns / C's columns: two independent halves, no accumulation needed
        int P1 = P / 2;
        matMulRecursive(A, B, C, M, N, P1, strideA, strideB, strideC);
        matMulRecursive(A, B + P1, C + P1, M, N, P - P1, strideA, strideB, strideC);
    } else {
        // split the shared dimension N: C = A1*B1 + A2*B2, both halves write the same C region
        int N1 = N / 2;
        matMulRecursive(A, B, C, M, N1, P, strideA, strideB, strideC);
        matMulRecursive(A + N1, B + N1 * strideB, C, M, N - N1, P, strideA, strideB, strideC);
    }
}

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
    memset(h_c, 0, C_bytes); // must start at zero since the recursion accumulates into it

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
    matMulRecursive(h_a, h_b, h_c, m, n, p, n, p, p);
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    printf("CPU time (compute only): %f ms\n", elapsedMs);

    free(h_a);
    free(h_b);
    free(h_c);

    return ok ? 0 : 1;
}
