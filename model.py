"""
Matrix Multiplication Algorithms in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_cpu
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }

            C[m * N + n] = sum;
        }
    }
}

# Step 2 - max_abs_diff
#include <cmath>

float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;

    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}

# Step 3 - matmul_naive_kernel
#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);

    matmul_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 4 - matmul_coalesced_kernel
#include <cuda_runtime.h>

__global__ void matmul_coalesced_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void launch_matmul_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matmul_coalesced_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 5 - time_launch_ms
#include <cuda_runtime.h>

typedef void (*matmul_launch_fn)(const float*, const float*, float*, int, int, int);

float time_launch_ms(matmul_launch_fn launch, const float* dA, const float* dB, float* dC, int M, int N, int K, int iters) {
    launch(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iters; ++i) {
        launch(dA, dB, dC, M, N, K);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_ms / static_cast<float>(iters);
}

double matmul_gflops(int M, int N, int K, float ms) {
    return (2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K))
           / (static_cast<double>(ms) * 1e-3)
           / 1e9;
}

# Step 6 - matmul_tiled_kernel
#include <cuda_runtime.h>

constexpr int TILE_SMEM = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SMEM][TILE_SMEM];
    __shared__ float Bs[TILE_SMEM][TILE_SMEM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_SMEM + tx;
    int row = blockIdx.y * TILE_SMEM + ty;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SMEM - 1) / TILE_SMEM;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int a_col = tile * TILE_SMEM + tx;
        int b_row = tile * TILE_SMEM + ty;

        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SMEM; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE_SMEM, TILE_SMEM);
    dim3 grid((N + TILE_SMEM - 1) / TILE_SMEM,
              (M + TILE_SMEM - 1) / TILE_SMEM);

    matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 7 - matmul_tiled_1d_kernel
#include <cuda_runtime.h>

constexpr int R1_BM = 64, R1_BN = 64, R1_BK = 8, R1_TM = 8;

__global__ void matmul_tiled_1d_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[R1_BM * R1_BK];
    __shared__ float Bs[R1_BK * R1_BN];

    int tid = threadIdx.x;

    // Each thread owns one column and 8 consecutive rows.
    int col = tid % R1_BN;
    int row_start = (tid / R1_BN) * R1_TM;

    int global_col = blockIdx.x * R1_BN + col;
    int global_row_start = blockIdx.y * R1_BM + row_start;

    float acc[R1_TM] = {0.0f};

    int num_tiles = (K + R1_BK - 1) / R1_BK;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int k_base = tile * R1_BK;

        // Each thread loads one element of the 64 x 8 A tile.
        int a_row = tid / R1_BK;
        int a_col = tid % R1_BK;

        int global_a_row = blockIdx.y * R1_BM + a_row;
        int global_a_col = k_base + a_col;

        if (global_a_row < M && global_a_col < K) {
            As[a_row * R1_BK + a_col] =
                A[global_a_row * K + global_a_col];
        } else {
            As[a_row * R1_BK + a_col] = 0.0f;
        }

        // Each thread loads one element of the 8 x 64 B tile.
        int b_row = tid / R1_BN;
        int b_col = tid % R1_BN;

        int global_b_row = k_base + b_row;
        int global_b_col = blockIdx.x * R1_BN + b_col;

        if (global_b_row < K && global_b_col < N) {
            Bs[b_row * R1_BN + b_col] =
                B[global_b_row * N + global_b_col];
        } else {
            Bs[b_row * R1_BN + b_col] = 0.0f;
        }

        __syncthreads();

        // Each thread reuses one B value for all 8 output rows.
        for (int k = 0; k < R1_BK; ++k) {
            float b = Bs[k * R1_BN + col];

            #pragma unroll
            for (int i = 0; i < R1_TM; ++i) {
                int a_row_local = row_start + i;
                acc[i] += As[a_row_local * R1_BK + k] * b;
            }
        }

        __syncthreads();
    }

    // Store the 8 results owned by this thread.
    #pragma unroll
    for (int i = 0; i < R1_TM; ++i) {
        int row = global_row_start + i;

        if (row < M && global_col < N) {
            C[row * N + global_col] = acc[i];
        }
    }
}

void launch_matmul_tiled_1d(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int THREADS = 512;

    dim3 block(THREADS);
    dim3 grid((N + R1_BN - 1) / R1_BN,
              (M + R1_BM - 1) / R1_BM);

    matmul_tiled_1d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 8 - matmul_tiled_2d_kernel
#include <cuda_runtime.h>

constexpr int R2_BM = 64, R2_BN = 64, R2_BK = 8, R2_TM = 4, R2_TN = 4;

__global__ void matmul_tiled_2d_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[R2_BM * R2_BK];
    __shared__ float Bs[R2_BK * R2_BN];

    const int tid = threadIdx.x;

    // Each thread owns a 4x4 output sub-tile.
    const int row_start = (tid / 16) * R2_TM;
    const int col_start = (tid % 16) * R2_TN;

    const int block_row = blockIdx.y * R2_BM;
    const int block_col = blockIdx.x * R2_BN;

    float acc[R2_TM][R2_TN] = {};

    const int num_tiles = (K + R2_BK - 1) / R2_BK;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int k_base = tile * R2_BK;

        // Load the 64x8 A tile. Each thread loads two elements.
        for (int i = tid; i < R2_BM * R2_BK; i += 256) {
            const int local_row = i / R2_BK;
            const int local_col = i % R2_BK;

            const int global_row = block_row + local_row;
            const int global_col = k_base + local_col;

            if (global_row < M && global_col < K) {
                As[i] = A[global_row * K + global_col];
            } else {
                As[i] = 0.0f;
            }
        }

        // Load the 8x64 B tile. Each thread loads two elements.
        for (int i = tid; i < R2_BK * R2_BN; i += 256) {
            const int local_row = i / R2_BN;
            const int local_col = i % R2_BN;

            const int global_row = k_base + local_row;
            const int global_col = block_col + local_col;

            if (global_row < K && global_col < N) {
                Bs[i] = B[global_row * N + global_col];
            } else {
                Bs[i] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < R2_BK; ++k) {
            float regA[R2_TM];
            float regB[R2_TN];

            #pragma unroll
            for (int i = 0; i < R2_TM; ++i) {
                regA[i] = As[(row_start + i) * R2_BK + k];
            }

            #pragma unroll
            for (int j = 0; j < R2_TN; ++j) {
                regB[j] = Bs[k * R2_BN + (col_start + j)];
            }

            #pragma unroll
            for (int i = 0; i < R2_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < R2_TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Store the 4x4 result with bounds checks.
    #pragma unroll
    for (int i = 0; i < R2_TM; ++i) {
        const int global_row = block_row + row_start + i;

        #pragma unroll
        for (int j = 0; j < R2_TN; ++j) {
            const int global_col = block_col + col_start + j;

            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = acc[i][j];
            }
        }
    }
}

void launch_matmul_tiled_2d(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int THREADS = 256;

    dim3 block(THREADS);
    dim3 grid((N + R2_BN - 1) / R2_BN,
              (M + R2_BM - 1) / R2_BM);

    matmul_tiled_2d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 9 - matmul_vectorized_kernel (not yet solved)
# TODO: implement

# Step 10 - matmul_double_buffered_kernel (not yet solved)
# TODO: implement

# Step 11 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 12 - matmul_batched_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_splitk_kernel (not yet solved)
# TODO: implement

# Step 14 - gemv_kernel (not yet solved)
# TODO: implement

# Step 15 - matmul_bias_relu_kernel (not yet solved)
# TODO: implement

# Step 16 - matrix_addsub_kernel (not yet solved)
# TODO: implement

# Step 17 - strassen_one_level (not yet solved)
# TODO: implement

# Step 18 - csr_spmm_kernel (not yet solved)
# TODO: implement

# Step 19 - matmul_lower_triangular_kernel (not yet solved)
# TODO: implement

# Step 20 - matmul_dispatch (not yet solved)
# TODO: implement

