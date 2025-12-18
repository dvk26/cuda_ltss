## Detailed Implementation Plan: Memory Aliasing (Ping-Pong Gradient Pools) (`src/gpu_v1.4/`)

This plan expands `documents/12-18-2025/plans/structure-memory-aliasing.md` into a step-by-step checklist that implements **Option A: Static Double Pooling (Ping-Pong)** for intermediate backprop gradients, with minimal disruption to the existing v1.4 kernels and training loop.

Goal: replace per-layer `float* d_d*` feature-map gradient allocations with **two reusable `float` pools** sized to the largest gradient tensor, while keeping weight gradients (`d_gw*`, `d_gb*`) unchanged.

---

## Phase 1 — Scope & lifetime audit (what can be pooled safely)

### 1. Identify “eligible” vs “must-persist” buffers

Eligible for ping-pong pooling (ephemeral, strictly sequential in `backward_compute_gradients()`):
- `d_dc5_`, `d_du2_`, `d_dr4_`, `d_dc4_`, `d_du1_`, `d_dr3_`, `d_dc3_`, `d_dp2_`, `d_dr2_`, `d_dc2_`, `d_dp1_`, `d_dr1_`, `d_dc1_`, `d_dx_`

Must remain separately allocated (persist across steps / used outside the chain):
- Weights + optimizer state: `d_w*`, `d_b*`
- Weight gradients: `d_gw*`, `d_gb*` (written with atomics in `conv2d_backward_filter_kernel*`)
- Forward activations / masks / checkpoint scratch: `d_x_`, `d_xh_`, `d_c*`, `d_u*`, `d_p*`, `d_mask*`, `d_scratch*`
- Loss pipeline buffers: `d_dy_`, `d_loss_accum_`, `d_found_inf_nan_`

### 2. Confirm kernel lifetime constraints
Key constraint to validate: every backward op consumes one “upstream” gradient and produces one “downstream” gradient, so only **two live feature-map gradients** are needed at a time.

Special case:
- `maxpool2x2_backward_kernel` uses `atomicAdd(&dX[x_idx], dY[idx])` and therefore requires `dX` to be **zeroed immediately before** the call (because pooled memory may contain arbitrary values from earlier steps).
  - See `src/gpu_v1.4/gpu_autoencoder.cu:558` for the atomicAdd write.

Acceptance criteria:
- A written mapping exists for each backward op: `(reads grad) -> (writes grad)` and confirms they can alternate between two pools.
- Atomic-output kernels (currently maxpool backward, weight-gradient kernels) are explicitly identified as requiring destination initialization.

---

## Phase 2 — Pool sizing & allocation (replace many `cudaMalloc`s with 2)

### 3. Add 2 pool members (+ optional size tracking)
Edits:
- `src/gpu_v1.4/gpu_autoencoder.hpp:97`
  - Keep the existing named pointers (`d_dc5_`, `d_du2_`, …) to minimize call-site churn, but stop allocating them individually.
  - Add pool owners and (optional) capacity:
    - Add:
      - `float* d_grad_pool_a_;`
      - `float* d_grad_pool_b_;`
      - `size_t grad_pool_elems_;` (stores “max elements” used to allocate both pools)

Acceptance criteria:
- The header clearly distinguishes “pool owners” vs “alias pointers” (so free logic cannot double-free).

### 4. Compute max gradient size and allocate the pools in `alloc_all()`
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu:716` (`GPUAutoencoder::alloc_all`)
  - Remove these individual allocations:
    - `cudaMalloc(&d_dc5_, ...)` through `cudaMalloc(&d_dx_, ...)` (currently `:753-766`).
  - Replace with:
    1. Compute `grad_pool_elems_ = max(...)` over all eligible gradient tensor element counts, using existing shape helpers:
       - Candidate sizes (elements):
         - `nchw_size(N_, 256, H_,  W_)`   (covers `d_du2_`, `d_dr1_`, `d_dc1_`)
         - `nchw_size(N_, 256, H1_, W1_)`  (covers `d_dr4_`, `d_dc4_`, `d_dp1_`)
         - `nchw_size(N_, 128, H1_, W1_)`  (covers `d_du1_`, `d_dc2_`)
         - `nchw_size(N_, 128, H2_, W2_)`  (covers `d_dr3_`, `d_dc3_`, `d_dp2_`)
         - `nchw_size(N_, 3,   H_,  W_)`   (covers `d_dc5_`, `d_dx_`)
    2. Allocate:
       - `cudaMalloc(&d_grad_pool_a_, grad_pool_elems_ * sizeof(float))`
       - `cudaMalloc(&d_grad_pool_b_, grad_pool_elems_ * sizeof(float))`
    3. Assign alias pointers to either pool A or B (static mapping; no offsets):
       - Pool A: `d_dc5_`, `d_dr4_`, `d_du1_`, `d_dc3_`, `d_dr2_`, `d_dp1_`, `d_dc1_`
       - Pool B: `d_du2_`, `d_dc4_`, `d_dr3_`, `d_dp2_`, `d_dc2_`, `d_dr1_`, `d_dx_`

Rationale for static mapping:
- In `backward_compute_gradients()` (see `src/gpu_v1.4/gpu_autoencoder.cu:868-922`), each operation reads an upstream grad and writes a downstream grad; alternating pools ensures “read” and “write” never alias during a single kernel launch.

Acceptance criteria:
- The binary allocates exactly 2 large intermediate-gradient buffers (plus weight gradients) instead of 14+.
- The program still compiles after removing the individual `cudaMalloc` calls.

### 5. Update `free_all()` to free only the pools (avoid double-free)
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu:774` (`GPUAutoencoder::free_all`)
  - Remove `safe_free(...)` calls for the alias pointers (currently `:794-798`).
  - Add:
    - `safe_free(d_grad_pool_a_);`
    - `safe_free(d_grad_pool_b_);`
  - Set alias pointers to `nullptr` after freeing pools (defensive clarity), e.g.:
    - `d_dc5_ = d_du2_ = ... = nullptr;`

Acceptance criteria:
- Running under CUDA runtime does not trigger invalid frees (no “double free” / “invalid device pointer” errors).

---

## Phase 3 — Pointer aliasing contract (make the reuse explicit and hard to misuse)

### 6. Codify the backward “gradient flow” as a table comment near the assignments
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu:716` (after pool allocation + alias assignment)
  - Add a short comment table documenting:
    - Backward step order
    - Which named pointer is “current” (read) and “next” (write)
    - Which pool each named pointer aliases

Example table content (conceptual, not code):
- `dc5(A) -> du2(B) -> dr4(A) -> dc4(B) -> du1(A) -> dr3(B) -> dc3(A) -> dp2(B) -> dr2(A) -> dc2(B) -> dp1(A) -> dr1(B) -> dc1(A) -> dx(B)`

Acceptance criteria:
- A future edit to `backward_compute_gradients()` can be reviewed against the contract without re-deriving the lifetime graph.

---

## Phase 4 — Backward pass integration (remove redundant clears, add required clears)

### 7. Stop zeroing all intermediate gradient buffers at the top
Problem:
- Current code zeroes every intermediate gradient pointer (see `src/gpu_v1.4/gpu_autoencoder.cu:884-888`).
- After pooling, these pointers alias the same two buffers; blanket “zero everything” is both redundant and, for atomicAdd outputs, can be **incorrectly timed** (because pooled memory is overwritten before maxpool backward runs).

Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu:868` (`GPUAutoencoder::backward_compute_gradients`)
  - Keep zeroing weight gradients (`d_gw*`, `d_gb*`) as-is.
  - Remove the `zero_buf(...)` calls for:
    - `d_du2_ d_dr4_ d_dc4_ d_du1_ d_dr3_ d_dc3_ d_dp2_ d_dr2_ d_dc2_ d_dp1_ d_dr1_ d_dc1_ d_dx_`
  - Keep the initial `d_dc5_` fill (copy from host or `d_dy_`) as-is.

Acceptance criteria:
- Backward still produces finite gradients (no NaNs introduced by stale values).
- Kernel execution remains correct despite removing clears (all non-atomic outputs are fully overwritten by their producing kernels).

### 8. Zero pooled destinations immediately before atomicAdd kernels
Required because maxpool backward accumulates into `dX`:
- `maxpool2x2_backward_kernel(..., d_dr2_, ...)` at `src/gpu_v1.4/gpu_autoencoder.cu:908`
- `maxpool2x2_backward_kernel(..., d_dr1_, ...)` at `src/gpu_v1.4/gpu_autoencoder.cu:917`

Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu:868` (in `backward_compute_gradients`)
  - Immediately before the first maxpool backward call (`:908`), add a clear of the destination region:
    - Clear `d_dr2_` for `nchw_size(N_,128,H1_,W1_)` floats.
  - Immediately before the second maxpool backward call (`:917`), add a clear of the destination region:
    - Clear `d_dr1_` for `nchw_size(N_,256,H_,W_)` floats.
  - Use either:
    - `zero_kernel<<<...>>>(d_dr2_, elems)` / `zero_kernel<<<...>>>(d_dr1_, elems)`, or
    - `cudaMemset(d_dr2_, 0, elems*sizeof(float))` / `cudaMemset(d_dr1_, 0, ...)`

Acceptance criteria:
- Pooling does not change maxpool-backprop correctness (no accumulation of old values).
- Loss curve remains comparable to pre-aliasing baseline for the first epoch(s).

---

## Phase 5 — Validation & benchmarking (prove VRAM wins and keep correctness)

### 9. Build + smoke run
Build (example; adjust `-arch` to the GPU):
- `nvcc -std=c++17 -O2 src/gpu_v1.4/main.cu src/gpu_v1.4/gpu_autoencoder.cu -I./src/gpu_v1.4 -arch=sm_75 -o autoencoder_gpu_v1_4`

Run:
- `./autoencoder_gpu_v1_4 cifar-10-batches-bin --keep-partial`

Acceptance criteria:
- Compiles without warnings/errors.
- Runs at least 1 epoch without CUDA errors (`cudaGetLastError` / `CUDA_CHECK` stays clean).
- Outputs weights into `out-gpu-v1.4/weights_epoch_*.bin`.

### 10. Memory reduction check (peak VRAM)
Measurement:
- Use existing “Peak VRAM” logging in `src/gpu_v1.4/main.cu` (already tracks `peak_used_bytes` via `cudaMemGetInfo`).

Acceptance criteria:
- “Peak VRAM” decreases versus the current baseline (same `batch_size`, same `checkpoint_mode_`, same GPU).
- Expected order-of-magnitude: removal of ~14 separate feature-map gradient allocations replaced by 2 pools sized to `N*256*H*W` floats.

### 11. Correctness regression check (quick)
Acceptance criteria:
- `compute_loss()` remains finite and within a reasonable range compared to baseline for the same run setup.
- No NaNs/Infs are detected by the existing non-finite checks over `d_gw*`/`d_gb*` in `train_step()`.

