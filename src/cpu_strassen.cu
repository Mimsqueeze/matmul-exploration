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

// Strassen's algorithm: C[N x N] = A[N x N] * B[N x N], 7 recursive multiplications instead of 8.
// Requires N to be square and evenly divisible by 2 down to BASE_CASE_SIZE (e.g. a power of two).
void strassen(const float* A, const float* B, float* C, int N,
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

    // sums/differences of quadrants, needed by more than one M term below
    float* S1 = (float*)malloc(halfBytes);  // A11 + A22
    float* S2 = (float*)malloc(halfBytes);  // B11 + B22
    float* S3 = (float*)malloc(halfBytes);  // A21 + A22
    float* S4 = (float*)malloc(halfBytes);  // B12 - B22
    float* S5 = (float*)malloc(halfBytes);  // B21 - B11
    float* S6 = (float*)malloc(halfBytes);  // A11 + A12
    float* S7 = (float*)malloc(halfBytes);  // A21 - A11
    float* S8 = (float*)malloc(halfBytes);  // B11 + B12
    float* S9 = (float*)malloc(halfBytes);  // A12 - A22
    float* S10 = (float*)malloc(halfBytes); // B21 + B22

    matAddSub(A11, strideA, A22, strideA, S1, half, half, false);
    matAddSub(B11, strideB, B22, strideB, S2, half, half, false);
    matAddSub(A21, strideA, A22, strideA, S3, half, half, false);
    matAddSub(B12, strideB, B22, strideB, S4, half, half, true);
    matAddSub(B21, strideB, B11, strideB, S5, half, half, true);
    matAddSub(A11, strideA, A12, strideA, S6, half, half, false);
    matAddSub(A21, strideA, A11, strideA, S7, half, half, true);
    matAddSub(B11, strideB, B12, strideB, S8, half, half, false);
    matAddSub(A12, strideA, A22, strideA, S9, half, half, true);
    matAddSub(B21, strideB, B22, strideB, S10, half, half, false);

    float* M1 = (float*)malloc(halfBytes);
    float* M2 = (float*)malloc(halfBytes);
    float* M3 = (float*)malloc(halfBytes);
    float* M4 = (float*)malloc(halfBytes);
    float* M5 = (float*)malloc(halfBytes);
    float* M6 = (float*)malloc(halfBytes);
    float* M7 = (float*)malloc(halfBytes);

    strassen(S1, S2, M1, half, half, half, half);    // M1 = (A11+A22)(B11+B22)
    strassen(S3, B11, M2, half, half, strideB, half); // M2 = (A21+A22)*B11
    strassen(A11, S4, M3, half, strideA, half, half); // M3 = A11*(B12-B22)
    strassen(A22, S5, M4, half, strideA, half, half); // M4 = A22*(B21-B11)
    strassen(S6, B22, M5, half, half, strideB, half); // M5 = (A11+A12)*B22
    strassen(S7, S8, M6, half, half, half, half);     // M6 = (A21-A11)(B11+B12)
    strassen(S9, S10, M7, half, half, half, half);    // M7 = (A12-A22)(B21+B22)

    free(S1); free(S2); free(S3); free(S4); free(S5);
    free(S6); free(S7); free(S8); free(S9); free(S10);

    // C11 = M1 + M4 - M5 + M7
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            C11[i * strideC + j] = M1[i * half + j] + M4[i * half + j] - M5[i * half + j] + M7[i * half + j];
        }
    }
    matAddSub(M3, half, M5, half, C12, strideC, half, false); // C12 = M3 + M5
    matAddSub(M2, half, M4, half, C21, strideC, half, false); // C21 = M2 + M4
    // C22 = M1 - M2 + M3 + M6
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            C22[i * strideC + j] = M1[i * half + j] - M2[i * half + j] + M3[i * half + j] + M6[i * half + j];
        }
    }

    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
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
    strassen(h_a, h_b, h_c, m, n, p, p);
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    printf("CPU time (compute only): %f ms\n", elapsedMs);

    free(h_a);
    free(h_b);
    free(h_c);

    return ok ? 0 : 1;
}
