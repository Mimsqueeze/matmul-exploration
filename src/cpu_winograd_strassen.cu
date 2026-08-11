#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// base case size below which we stop recursing and just do the direct triple loop
const int BASE_CASE_SIZE = 64;

// Z = X (+/-) Y, elementwise, over a size x size block. Each matrix can have its own stride.
static void matAddSub(const float* X, int strideX, const float* Y, int strideY,
                       float* Z, int strideZ, int size, bool subtract) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            Z[i * strideZ + j] = X[i * strideX + j] + (subtract ? -Y[i * strideY + j] : Y[i * strideY + j]);
        }
    }
}

// Winograd's variant of Strassen's algorithm: C[N x N] = A[N x N] * B[N x N].
// Same 7 recursive multiplications as Strassen, but the quadrant sums/differences
// are chained to reuse shared subexpressions, cutting 18 additions down to 15.
// Requires N to be square and evenly divisible by 2 down to BASE_CASE_SIZE (e.g. a power of two).
void winogradStrassen(const float* A, const float* B, float* C, int N,
                       int strideA, int strideB, int strideC) {
    if (N <= BASE_CASE_SIZE) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * strideA + k] * B[k * strideB + j];
                }
                C[i * strideC + j] = sum;
            }
        }
        return;
    }

    int half = N / 2;
    const float* A11 = A;
    const float* A12 = A + half;
    const float* A21 = A + half * strideA;
    const float* A22 = A + half * strideA + half;

    const float* B11 = B;
    const float* B12 = B + half;
    const float* B21 = B + half * strideB;
    const float* B22 = B + half * strideB + half;

    float* C11 = C;
    float* C12 = C + half;
    float* C21 = C + half * strideC;
    float* C22 = C + half * strideC + half;

    const size_t halfBytes = (size_t)half * half * sizeof(float);

    // S1..S8 are a dependency chain: each reuses the previous S where possible,
    // which is what gets the addition count down from 18 to 15.
    float* S1 = (float*)malloc(halfBytes); // A21 + A22
    float* S2 = (float*)malloc(halfBytes); // S1 - A11
    float* S3 = (float*)malloc(halfBytes); // A11 - A21
    float* S4 = (float*)malloc(halfBytes); // A12 - S2
    float* S5 = (float*)malloc(halfBytes); // B12 - B11
    float* S6 = (float*)malloc(halfBytes); // B22 - S5
    float* S7 = (float*)malloc(halfBytes); // B22 - B12
    float* S8 = (float*)malloc(halfBytes); // S6 - B21

    matAddSub(A21, strideA, A22, strideA, S1, half, half, false);
    matAddSub(S1, half, A11, strideA, S2, half, half, true);
    matAddSub(A11, strideA, A21, strideA, S3, half, half, true);
    matAddSub(A12, strideA, S2, half, S4, half, half, true);
    matAddSub(B12, strideB, B11, strideB, S5, half, half, true);
    matAddSub(B22, strideB, S5, half, S6, half, half, true);
    matAddSub(B22, strideB, B12, strideB, S7, half, half, true);
    matAddSub(S6, half, B21, strideB, S8, half, half, true);

    float* P1 = (float*)malloc(halfBytes);
    float* P2 = (float*)malloc(halfBytes);
    float* P3 = (float*)malloc(halfBytes);
    float* P4 = (float*)malloc(halfBytes);
    float* P5 = (float*)malloc(halfBytes);
    float* P6 = (float*)malloc(halfBytes);
    float* P7 = (float*)malloc(halfBytes);

    winogradStrassen(S2, S6, P1, half, half, half, half);     // P1 = S2 * S6
    winogradStrassen(A11, B11, P2, half, strideA, strideB, half); // P2 = A11 * B11
    winogradStrassen(A12, B21, P3, half, strideA, strideB, half); // P3 = A12 * B21
    winogradStrassen(S3, S7, P4, half, half, half, half);     // P4 = S3 * S7
    winogradStrassen(S1, S5, P5, half, half, half, half);     // P5 = S1 * S5
    winogradStrassen(S4, B22, P6, half, half, strideB, half); // P6 = S4 * B22
    winogradStrassen(A22, S8, P7, half, strideA, half, half); // P7 = A22 * S8

    free(S1); free(S2); free(S3); free(S4);
    free(S5); free(S6); free(S7); free(S8);

    float* T1 = (float*)malloc(halfBytes); // P1 + P2
    float* T2 = (float*)malloc(halfBytes); // T1 + P4
    matAddSub(P1, half, P2, half, T1, half, half, false);
    matAddSub(T1, half, P4, half, T2, half, half, false);

    matAddSub(P2, half, P3, half, C11, strideC, half, false); // C11 = P2 + P3
    // C12 = T1 + P5 + P6
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            C12[i * strideC + j] = T1[i * half + j] + P5[i * half + j] + P6[i * half + j];
        }
    }
    matAddSub(T2, half, P7, half, C21, strideC, half, true);  // C21 = T2 - P7
    matAddSub(T2, half, P5, half, C22, strideC, half, false); // C22 = T2 + P5

    free(T1); free(T2);
    free(P1); free(P2); free(P3); free(P4); free(P5); free(P6); free(P7);
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
    winogradStrassen(h_a, h_b, h_c, m, n, p, p);
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    printf("CPU time (compute only): %f ms\n", elapsedMs);

    free(h_a);
    free(h_b);
    free(h_c);

    return ok ? 0 : 1;
}
