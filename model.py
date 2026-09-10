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

# Step 9 - matmul_vectorized_kernel
#include <cuda_runtime.h>

constexpr int V_BM = 64, V_BN = 64, V_BK = 8, V_TM = 4, V_TN = 4;

__global__ void matmul_vectorized_kernel(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    // A tile is stored transposed:
    // As[k][m] -> As[k * V_BM + m]
    __shared__ float As[V_BK * V_BM];

    // B tile remains row-major.
    __shared__ __align__(16) float Bs[V_BK * V_BN];

    const int tid = threadIdx.x;

    // Each thread owns a 4x4 output tile.
    const int trow = (tid / 16) * V_TM;
    const int tcol = (tid % 16) * V_TN;

    const int block_row = blockIdx.y * V_BM;
    const int block_col = blockIdx.x * V_BN;

    const int global_row_start = block_row + trow;
    const int global_col_start = block_col + tcol;

    float acc[V_TM][V_TN] = {};

    const int num_tiles = (K + V_BK - 1) / V_BK;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int k_base = tile * V_BK;

        // ------------------------------------------------------------
        // Load A tile.
        // Threads 0..127 each load one float4.
        // r = tid / 2
        // c = (tid % 2) * 4
        // ------------------------------------------------------------
        if (tid < 128) {
            const int r = tid / 2;
            const int c = (tid % 2) * 4;

            const int global_row = block_row + r;
            const int global_col = k_base + c;

            float a0 = 0.0f;
            float a1 = 0.0f;
            float a2 = 0.0f;
            float a3 = 0.0f;

            if (global_row < M) {
                // Normal aligned float4 load when all four elements
                // are inside the K dimension.
                if (global_col + 3 < K) {
                    float4 value =
                        *reinterpret_cast<const float4*>(
                            A + global_row * K + global_col);

                    a0 = value.x;
                    a1 = value.y;
                    a2 = value.z;
                    a3 = value.w;
                } else {
                    // Partial final K tile.
                    if (global_col < K) {
                        a0 = A[global_row * K + global_col];
                    }
                    if (global_col + 1 < K) {
                        a1 = A[global_row * K + global_col + 1];
                    }
                    if (global_col + 2 < K) {
                        a2 = A[global_row * K + global_col + 2];
                    }
                    if (global_col + 3 < K) {
                        a3 = A[global_row * K + global_col + 3];
                    }
                }
            }

            // Store A transposed:
            // As[c + q][r]
            As[(c + 0) * V_BM + r] = a0;
            As[(c + 1) * V_BM + r] = a1;
            As[(c + 2) * V_BM + r] = a2;
            As[(c + 3) * V_BM + r] = a3;
        }

        // ------------------------------------------------------------
        // Load B tile.
        // Threads 0..127 each load one float4.
        // r = tid / 16
        // c = (tid % 16) * 4
        //
        // IMPORTANT: guard both row and column. This is required for
        // partial and narrow N tiles.
        // ------------------------------------------------------------
        if (tid < 128) {
            const int r = tid / 16;
            const int c = (tid % 16) * 4;

            const int global_row = k_base + r;
            const int global_col = block_col + c;

            float b0 = 0.0f;
            float b1 = 0.0f;
            float b2 = 0.0f;
            float b3 = 0.0f;

            if (global_row < K) {
                if (global_col + 3 < N) {
                    float4 value =
                        *reinterpret_cast<const float4*>(
                            B + global_row * N + global_col);

                    b0 = value.x;
                    b1 = value.y;
                    b2 = value.z;
                    b3 = value.w;
                } else {
                    // Partial final N tile.
                    if (global_col < N) {
                        b0 = B[global_row * N + global_col];
                    }
                    if (global_col + 1 < N) {
                        b1 = B[global_row * N + global_col + 1];
                    }
                    if (global_col + 2 < N) {
                        b2 = B[global_row * N + global_col + 2];
                    }
                    if (global_col + 3 < N) {
                        b3 = B[global_row * N + global_col + 3];
                    }
                }
            }

            reinterpret_cast<float4*>(
                &Bs[r * V_BN + c])[0] =
                make_float4(b0, b1, b2, b3);
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Compute 4x4 register tile.
        // For each k:
        //   - load 4 contiguous A values
        //   - load one float4 B value
        //   - perform 16 FMAs
        // ------------------------------------------------------------
        for (int k = 0; k < V_BK; ++k) {
            float regA[V_TM];
            float4 regB;

            #pragma unroll
            for (int i = 0; i < V_TM; ++i) {
                regA[i] = As[k * V_BM + trow + i];
            }

            regB = reinterpret_cast<const float4*>(
                &Bs[k * V_BN + tcol])[0];

            #pragma unroll
            for (int i = 0; i < V_TM; ++i) {
                acc[i][0] += regA[i] * regB.x;
                acc[i][1] += regA[i] * regB.y;
                acc[i][2] += regA[i] * regB.z;
                acc[i][3] += regA[i] * regB.w;
            }
        }

        __syncthreads();
    }

    // ------------------------------------------------------------
    // Store results.
    // Use float4 when the complete 4-column vector is in range.
    // For a partial N tile, store only valid scalar elements.
    // ------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < V_TM; ++i) {
        const int row = global_row_start + i;

        if (row < M) {
            if (global_col_start + 3 < N) {
                float4 out = make_float4(
                    acc[i][0],
                    acc[i][1],
                    acc[i][2],
                    acc[i][3]
                );

                reinterpret_cast<float4*>(
                    C + row * N + global_col_start)[0] = out;
            } else {
                if (global_col_start < N) {
                    C[row * N + global_col_start] = acc[i][0];
                }
                if (global_col_start + 1 < N) {
                    C[row * N + global_col_start + 1] = acc[i][1];
                }
                if (global_col_start + 2 < N) {
                    C[row * N + global_col_start + 2] = acc[i][2];
                }
                if (global_col_start + 3 < N) {
                    C[row * N + global_col_start + 3] = acc[i][3];
                }
            }
        }
    }
}

void launch_matmul_vectorized(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + V_BN - 1) / V_BN,
              (M + V_BM - 1) / V_BM);

    matmul_vectorized_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 10 - matmul_double_buffered_kernel
constexpr int D_BM = 64, D_BN = 64, D_BK = 8, D_TM = 4, D_TN = 4;

__global__ void matmul_double_buffered_kernel(const float* A, const float* B, float* C,
                                              int M, int N, int K) {
    __shared__ float As[2][D_BM * D_BK];
    __shared__ float Bs[2][D_BK * D_BN];

    const int tid = threadIdx.x;

    // Each thread owns a 4x4 output tile.
    const int trow = (tid / 16) * D_TM;
    const int tcol = (tid % 16) * D_TN;

    const int block_row = blockIdx.y * D_BM;
    const int block_col = blockIdx.x * D_BN;

    const int global_row_start = block_row + trow;
    const int global_col_start = block_col + tcol;

    float acc[D_TM][D_TN] = {};

    const int num_tiles = (K + D_BK - 1) / D_BK;

    // ------------------------------------------------------------
    // Load K-tile 0 into shared-memory stage 0.
    // There are 512 A elements and 512 B elements, so each of
    // 256 threads loads two A elements and two B elements.
    // ------------------------------------------------------------
    for (int i = tid; i < D_BM * D_BK; i += 256) {
        const int local_row = i / D_BK;
        const int local_col = i % D_BK;

        const int global_row = block_row + local_row;
        const int global_col = local_col;

        if (global_row < M && global_col < K) {
            As[0][i] = A[global_row * K + global_col];
        } else {
            As[0][i] = 0.0f;
        }
    }

    for (int i = tid; i < D_BK * D_BN; i += 256) {
        const int local_row = i / D_BN;
        const int local_col = i % D_BN;

        const int global_row = local_row;
        const int global_col = block_col + local_col;

        if (global_row < K && global_col < N) {
            Bs[0][i] = B[global_row * N + global_col];
        } else {
            Bs[0][i] = 0.0f;
        }
    }

    __syncthreads();

    int current_stage = 0;

    // ------------------------------------------------------------
    // Process each K tile.
    // ------------------------------------------------------------
    for (int t = 0; t < num_tiles; ++t) {
        const int next_stage = 1 - current_stage;

        // --------------------------------------------------------
        // Prefetch tile t+1 into registers.
        // Each thread loads two A values and two B values.
        // --------------------------------------------------------
        float prefetch_A[2] = {0.0f, 0.0f};
        float prefetch_B[2] = {0.0f, 0.0f};

        if (t + 1 < num_tiles) {
            const int next_k_base = (t + 1) * D_BK;

            // The two A elements owned by this thread.
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + q * 256;

                const int local_row = idx / D_BK;
                const int local_col = idx % D_BK;

                const int global_row = block_row + local_row;
                const int global_col = next_k_base + local_col;

                if (global_row < M && global_col < K) {
                    prefetch_A[q] =
                        A[global_row * K + global_col];
                }
            }

            // The two B elements owned by this thread.
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + q * 256;

                const int local_row = idx / D_BN;
                const int local_col = idx % D_BN;

                const int global_row = next_k_base + local_row;
                const int global_col = block_col + local_col;

                if (global_row < K && global_col < N) {
                    prefetch_B[q] =
                        B[global_row * N + global_col];
                }
            }
        }

        // --------------------------------------------------------
        // Compute from the current shared-memory stage.
        // --------------------------------------------------------
        for (int k = 0; k < D_BK; ++k) {
            float regA[D_TM];
            float regB[D_TN];

            #pragma unroll
            for (int i = 0; i < D_TM; ++i) {
                regA[i] =
                    As[current_stage][(trow + i) * D_BK + k];
            }

            #pragma unroll
            for (int j = 0; j < D_TN; ++j) {
                regB[j] =
                    Bs[current_stage][k * D_BN + (tcol + j)];
            }

            #pragma unroll
            for (int i = 0; i < D_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < D_TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        // --------------------------------------------------------
        // Write prefetched tile t+1 from registers into the
        // alternate shared-memory stage.
        // Every thread writes exactly the values it prefetched.
        // --------------------------------------------------------
        if (t + 1 < num_tiles) {
            const int next_k_base = (t + 1) * D_BK;

            for (int q = 0; q < 2; ++q) {
                const int idx = tid + q * 256;

                const int local_row = idx / D_BK;
                const int local_col = idx % D_BK;

                As[next_stage][idx] = prefetch_A[q];
            }

            for (int q = 0; q < 2; ++q) {
                const int idx = tid + q * 256;

                Bs[next_stage][idx] = prefetch_B[q];
            }

            // Exactly one synchronization before the next stage
            // becomes visible to all threads.
            __syncthreads();

            current_stage = next_stage;
        }
    }

    // ------------------------------------------------------------
    // Store the 4x4 result tile with bounds checks.
    // ------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < D_TM; ++i) {
        const int row = global_row_start + i;

        if (row < M) {
            #pragma unroll
            for (int j = 0; j < D_TN; ++j) {
                const int col = global_col_start + j;

                if (col < N) {
                    C[row * N + col] = acc[i][j];
                }
            }
        }
    }
}

void launch_matmul_double_buffered(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + D_BN - 1) / D_BN,
              (M + D_BM - 1) / D_BM);

    matmul_double_buffered_kernel<<<grid, block>>>(
        A, B, C, M, N, K
    );
}

# Step 11 - matmul_nt_kernel
constexpr int TILE_NT = 16;

__global__ void matmul_nt_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    __shared__ float As[TILE_NT][TILE_NT];
    __shared__ float Bs[TILE_NT][TILE_NT];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE_NT + ty;
    const int col = blockIdx.x * TILE_NT + tx;

    float acc = 0.0f;

    const int num_tiles = (K + TILE_NT - 1) / TILE_NT;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int k0 = tile * TILE_NT;
        const int k = k0 + tx;

        // Load A[row][k0 + tx].
        if (row < M && k < K) {
            As[ty][tx] = A[row * K + k];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B[block_col + ty][k0 + tx].
        // B has N rows and K columns.
        const int b_row = blockIdx.x * TILE_NT + ty;

        if (b_row < N && k < K) {
            Bs[ty][tx] = B[b_row * K + k];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // C[row][col] = A[row] dot B[col].
        #pragma unroll
        for (int i = 0; i < TILE_NT; ++i) {
            acc += As[ty][i] * Bs[tx][i];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_matmul_nt(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 block(TILE_NT, TILE_NT);
    dim3 grid((N + TILE_NT - 1) / TILE_NT,
              (M + TILE_NT - 1) / TILE_NT);

    matmul_nt_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 12 - matmul_batched_kernel
constexpr int TILE_BATCH = 16;

__global__ void matmul_batched_kernel(const float* A, const float* B, float* C,
                                      int M, int N, int K) {
    __shared__ float As[TILE_BATCH][TILE_BATCH];
    __shared__ float Bs[TILE_BATCH][TILE_BATCH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE_BATCH + ty;
    const int col = blockIdx.x * TILE_BATCH + tx;

    const int batch_idx = blockIdx.z;

    // Each batch element is stored contiguously:
    // A: M x K
    // B: K x N
    // C: M x N
    const float* Ab = A + batch_idx * M * K;
    const float* Bb = B + batch_idx * K * N;
    float* Cb = C + batch_idx * M * N;

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_BATCH) {
        // Load A tile.
        const int a_col = k0 + tx;

        if (row < M && a_col < K) {
            As[ty][tx] = Ab[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile.
        const int b_row = k0 + ty;

        if (b_row < K && col < N) {
            Bs[ty][tx] = Bb[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_BATCH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Cb[row * N + col] = sum;
    }
}

void launch_matmul_batched(const float* A, const float* B, float* C,
                           int M, int N, int K, int batch) {
    dim3 block(TILE_BATCH, TILE_BATCH);
    dim3 grid((N + TILE_BATCH - 1) / TILE_BATCH,
              (M + TILE_BATCH - 1) / TILE_BATCH,
              batch);

    matmul_batched_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

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

