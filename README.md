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
- [ ] **14.** gemv_kernel
- [ ] **15.** matmul_bias_relu_kernel
- [ ] **16.** matrix_addsub_kernel
- [ ] **17.** strassen_one_level
- [ ] **18.** csr_spmm_kernel
- [ ] **19.** matmul_lower_triangular_kernel
- [ ] **20.** matmul_dispatch

---

Built on Deep-ML.
