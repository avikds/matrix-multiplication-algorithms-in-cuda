# Matrix Multiplication Algorithms in CUDA

Write general matrix multiplication in CUDA a dozen different ways and measure every one of them on a real GPU. Start from a CPU reference and a naive kernel, fix memory coalescing, then climb the classic optimization ladder: shared-memory tiling, one- and two-dimensional register blocking, vectorized float4 loads with a transposed tile, and double buffering. Cover the shapes that need a different algorithm - transposed operands, batched products, split-K for skinny outputs, a warp-per-row matrix-vector kernel, a fused bias-ReLU epilogue - then leave the cubic world with one level of Strassen, a CSR sparse-times-dense kernel and a triangular product that skips empty tiles. Finish with a shape-aware dispatcher and a benchmark table in GFLOP/s.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** matmul_cpu
- [x] **2.** max_abs_diff
- [x] **3.** matmul_naive_kernel
- [x] **4.** matmul_coalesced_kernel
- [x] **5.** time_launch_ms
- [x] **6.** matmul_tiled_kernel
- [x] **7.** matmul_tiled_1d_kernel
- [x] **8.** matmul_tiled_2d_kernel
- [x] **9.** matmul_vectorized_kernel
- [x] **10.** matmul_double_buffered_kernel
- [x] **11.** matmul_nt_kernel
- [x] **12.** matmul_batched_kernel
- [x] **13.** matmul_splitk_kernel
- [x] **14.** gemv_kernel
- [x] **15.** matmul_bias_relu_kernel
- [x] **16.** matrix_addsub_kernel
- [x] **17.** strassen_one_level
- [x] **18.** csr_spmm_kernel
- [x] **19.** matmul_lower_triangular_kernel
- [x] **20.** matmul_dispatch

## Results

```
GPU: Tesla T4 (40 SMs, 320 GB/s peak bandwidth)

512x512x512, fp32, 10 timed launches each:
  naive                 6.053 ms      44.3 GFLOP/s  max|err| 1.1e-05
  coalesced             1.110 ms     241.7 GFLOP/s  max|err| 1.1e-05
  tiled 16x16           0.725 ms     370.4 GFLOP/s  max|err| 1.1e-05
  tiled 1D regs         0.351 ms     764.9 GFLOP/s  max|err| 1.1e-05
  tiled 2D regs         0.282 ms     951.2 GFLOP/s  max|err| 1.1e-05
  vectorized            0.256 ms    1049.6 GFLOP/s  max|err| 1.1e-05
  double buffered       0.237 ms    1130.6 GFLOP/s  max|err| 1.1e-05
  strassen 1 level      1.468 ms     182.9 GFLOP/s* max|err| 4.4e-05   (*counted as 2MNK; 7 tiled products of 256^3 plus 18 additions)
  lower triangular      0.297 ms  vs tiled on the same triangular A    0.551 ms  max|err| 9.5e-06

shape-specific kernels:
  64x64x4096: double buffered 0.702 ms (16 blocks) vs split-K x8 0.095 ms (128 blocks), max|err| 2.0e-04
  4096x4096 times a vector: coalesced GEMM 1.844 ms vs warp-per-row GEMV 0.256 ms (262 GB/s), max|err| 1.8e-04

matmul_dispatch decisions:
  4096 x    1 x 4096 -> gemv            max|err| 1.6e-04
    32 x   32 x 2048 -> split-K         max|err| 6.5e-05
   512 x  512 x  512 -> vectorized      max|err| 1.0e-05
    33 x   65 x   17 -> double buffered max|err| 7.2e-07
```
