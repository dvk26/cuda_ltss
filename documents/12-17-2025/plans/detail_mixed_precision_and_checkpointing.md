## Detailed Implementation Plan: Mixed Precision + Checkpointing (`src/gpu_v1.4/`)

This plan expands `documents/12-17-2025/plans/structure_mixed_precision_and_checkpointing.md` into a step-by-step execution checklist. It references current `src/gpu_v1.3/` line numbers as anchors; after copying into `src/gpu_v1.4/`, the corresponding code will be edited there.

---

## Phase 1 — `gpu_v1.4` baseline replication and measurement

### 1. Create the new variant directory
1. Create `src/gpu_v1.4/` by copying `src/gpu_v1.3/`.
2. Keep public APIs stable: `GPUAutoencoder` interface remains compatible with `main.cu`.

### 2. Fix variant naming consistency (avoid “v1.4” inside `gpu_v1.3`)
Edits:
- `src/gpu_v1.4/main.cu` (from `src/gpu_v1.3/main.cu`)
  - Change the binary usage string to match the new binary name (optional).
  - Keep output directory naming consistent with variant name.

Reference lines (current `src/gpu_v1.3/main.cu`):
- L50: `const std::string out_dir = "out-gpu-v1.4";`
  - Keep as-is in `src/gpu_v1.4/main.cu`, or rename to `out-gpu-v1.4-mp-ckpt` once features land.

### 3. Remove redundant activation copies (low-risk baseline cleanup)
Rationale: forward uses fused `conv+ReLU` kernels, but still copies `d_c* -> d_r*` purely to satisfy `relu_backward_kernel`. Removing the duplicates reduces VRAM before checkpointing.

Edits:
- `src/gpu_v1.4/gpu_autoencoder.hpp` (reference: `src/gpu_v1.3/gpu_autoencoder.hpp:72-78`)
  - Remove `d_r1_ d_r2_ d_r3_ d_r4_` members.
  - Update comments to reflect “ReLU output stored directly in `d_c*`”.

  Change (conceptual):
  - From:
    - `float* d_c1_; float* d_r1_; float* d_p1_;`
  - To:
    - `float* d_c1_; float* d_p1_;`

- `src/gpu_v1.4/gpu_autoencoder.cu`
  - `alloc_all()` (reference: `src/gpu_v1.3/gpu_autoencoder.cu:481-493`)
    - Remove `cudaMalloc` for `d_r1_ d_r2_ d_r3_ d_r4_`.
  - `free_all()` (search for `safe_free(d_r1_)` etc.)
    - Remove frees for those pointers.
  - `forward_pass()` (reference: `src/gpu_v1.3/gpu_autoencoder.cu:571-606`)
    - Remove `cudaMemcpy(d_r*_, d_c*_, ...)`.
    - Route pooling/upsampling inputs directly from `d_c*`:
      - L581: `maxpool2x2_forward_kernel(... d_r1_, d_p1_, ...)` → use `d_c1_`
      - L588: `... d_r2_ ...` → use `d_c2_`
      - L595: `upsample(... d_r3_, d_u1_, ...)` → use `d_c3_`
      - L602: `upsample(... d_r4_, d_u2_, ...)` → use `d_c4_`
  - `backward_pass()` (reference: `src/gpu_v1.3/gpu_autoencoder.cu` around the `relu_backward_kernel` calls)
    - Replace `relu_backward_kernel(... d_r4_ ...)` with `relu_backward_kernel(... d_c4_ ...)` (and similarly for `d_r3_, d_r2_, d_r1_`).
    - Replace maxpool backward destinations accordingly:
      - `maxpool2x2_backward_kernel(..., d_mask2_, d_dr2_, ...)` currently writes into `d_dr2_`; keep grads unchanged, but ensure the activation used by ReLU backward is `d_c2_`.

Acceptance criteria:
- Compiles and trains identically to `src/gpu_v1.3/` (loss curve similar; no NaNs).
- Lower VRAM than `src/gpu_v1.3/` due to removing `d_r*` buffers.

### 4. Add baseline measurement hooks (time + peak VRAM)
Edits:
- `src/gpu_v1.4/main.cu`
  - Add optional measurement printouts per epoch:
    - training throughput (already printed as “batches/s”)
    - peak VRAM usage (add a helper using `cudaMemGetInfo` and track minimum free memory observed).

Example change (conceptual):
- Add:
  - `size_t free_bytes, total_bytes; cudaMemGetInfo(&free_bytes, &total_bytes);`
  - Track `peak_used_bytes = max(peak_used_bytes, total_bytes - free_bytes)`.

---

## Phase 2 — Mixed precision (single strategy, two phrasings)

Target strategy (single strategy; two accepted phrasings):
- “mixed-precision training (FP16 activations, FP32 master weights)”
- “FP16 activations with FP32 accumulation”

Scope choices (to keep this iteration controlled):
- Keep weights, weight gradients, and optimizer state in FP32 (`float`).
- Store most forward activations in FP16 (`__half`) to reduce VRAM.
- Keep the final network output (`d_c5_`) in FP32 initially to avoid rewriting loss/metrics code.

### 5. Introduce FP16 activation buffer types
Edits:
- `src/gpu_v1.4/gpu_autoencoder.hpp` (reference: `src/gpu_v1.3/gpu_autoencoder.hpp:1-8, 68-79`)
  - Add include:
    - Change:
      - `#include <cuda_runtime.h>`
    - To:
      - `#include <cuda_runtime.h>`
      - `#include <cuda_fp16.h>`
  - Change activation pointer types from `float*` to `__half*` where desired.

Proposed activation dtype mapping:
- Keep FP32:
  - `d_x_` (input), `d_c5_` (final output), `d_dc5_` and other gradient buffers (initially)
- Change to FP16:
  - `d_c1_ d_p1_ d_c2_ d_p2_ d_c3_ d_u1_ d_c4_ d_u2_`

Example edits (conceptual):
- From (`src/gpu_v1.3/gpu_autoencoder.hpp:69-78`):
  - `float* d_c1_; float* d_p1_;`
- To (`src/gpu_v1.4/gpu_autoencoder.hpp`):
  - `__half* d_c1_; __half* d_p1_;`

### 6. Add explicit conversion kernels at FP32/FP16 boundaries
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu`
  - Add:
    - `fp32_to_fp16_kernel(const float* in, __half* out, int n)`
    - `fp16_to_fp32_kernel(const __half* in, float* out, int n)` (if needed)

Planned usage:
- Convert input once per batch:
  - In `set_input()` (reference: `src/gpu_v1.3/gpu_autoencoder.cu:562-565`), keep copying into `d_x_` (FP32), then convert to `d_xh_` (new FP16 input buffer) used by the FP16 forward path.

### 7. Implement FP16-IO forward kernels (FP32 accumulation)
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu`
  - Create FP16 variants of forward conv kernels that:
    - read `__half* x`
    - accumulate in `float`
    - write `__half* y` (for ReLU layers)
  - Keep the final layer output as FP32 for now:
    - read `__half* x`
    - accumulate in `float`
    - write `float* y`

Concrete edits (function signatures):
- From (reference: `src/gpu_v1.3/gpu_autoencoder.cu` wrappers):
  - `void launch_conv2d_relu_tiled(const float* x, const float* w, const float* b, float* y, ...)`
- To (`src/gpu_v1.4/gpu_autoencoder.cu`):
  - `void launch_conv2d_relu_tiled_fp16io(const __half* x, const float* w, const float* b, __half* y, ...)`
  - `void launch_conv2d_tiled_fp16in_fp32out(const __half* x, const float* w, const float* b, float* y, ...)`

### 8. Implement FP16 pooling and upsampling forward kernels
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu`
  - Add FP16 forward versions:
    - `maxpool2x2_forward_kernel_fp16(const __half* x, __half* y, int* mask, ...)`
    - `upsample2x_forward_kernel_fp16(const __half* x, __half* y, ...)`
  - Keep backward kernels (gradients) in FP32 initially:
    - maxpool backward writes FP32 gradients
    - upsample backward writes FP32 gradients

### 9. Adjust backward pass to consume FP16 activations
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu`
  - Replace ReLU backward to accept FP16 activations:
    - From (reference: `src/gpu_v1.3/gpu_autoencoder.cu`, kernel):
      - `relu_backward_kernel(const float* y, const float* dY, float* dX, ...)`
    - To:
      - `relu_backward_kernel_fp16y(const __half* y, const float* dY, float* dX, ...)`
  - Modify conv backward filter to read FP16 `x` activations:
    - From:
      - `conv2d_backward_filter_kernel(const float* x, const float* dY, float* gW, ...)`
    - To:
      - `conv2d_backward_filter_kernel_fp16x(const __half* x, const float* dY, float* gW, ...)`
  - Keep conv backward data as FP32 output gradients.

Acceptance criteria:
- Compiles and runs end-to-end.
- Loss remains finite and within an expected tolerance vs FP32 baseline (not necessarily identical).
- VRAM usage decreases due to FP16 activations.

---

## Phase 3 — Numerical stability (dynamic loss scaling)

This phase implements **dynamic loss scaling** as described in `documents/12-17-2025/plans/structure_mixed_precision_and_checkpointing.md`: Scale Loss → Backward → Unscale Gradients, plus a safety check for `Inf`/`NaN` gradients that can skip the optimizer step and adjust the scale factor.

### 10. Add dynamic loss-scaling state
Edits:
- `src/gpu_v1.4/gpu_autoencoder.hpp` (reference: `src/gpu_v1.3/gpu_autoencoder.hpp:55-60, 92-105`)
  - Add members (names can vary; this is the intended state):
    - `float loss_scale_;`
    - `float loss_scale_min_;`
    - `float loss_scale_max_;`
    - `float loss_scale_growth_factor_;` (e.g., `2.0f`)
    - `float loss_scale_backoff_factor_;` (e.g., `0.5f`)
    - `int loss_scale_growth_interval_steps_;` (e.g., `2000`)
    - `int loss_scale_good_steps_;` (counts consecutive “finite” steps)
  - Initialize in the constructor:
    - `loss_scale_ = 128.0f; loss_scale_min_ = 1.0f; loss_scale_max_ = 65536.0f;`
    - `loss_scale_growth_factor_ = 2.0f; loss_scale_backoff_factor_ = 0.5f;`
    - `loss_scale_growth_interval_steps_ = <N>; loss_scale_good_steps_ = 0;`

### 11. Scale the loss (implemented as scaling `dY`) and unscale updates
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu`
  - Modify the MSE gradient production to include `loss_scale_`:
    - In `mse_loss_kernel` (reference: `src/gpu_v1.3/gpu_autoencoder.cu` near the kernel definition), change signature:
      - From:
        - `__global__ void mse_loss_kernel(..., int total_elements)`
      - To:
        - `__global__ void mse_loss_kernel(..., int total_elements, float loss_scale)`
    - Change the gradient write:
      - From:
        - `dY[idx] = (2.0f * diff) / (float)total_elements;`
      - To:
        - `dY[idx] = loss_scale * (2.0f * diff) / (float)total_elements;`
  - Ensure the optimizer step is effectively unscaled:
    - Use an “effective learning rate” for updates:
      - Change update calls in `backward_pass()` (reference: `src/gpu_v1.3/gpu_autoencoder.cu:660-666`) from `lr` to `lr / loss_scale_` **only when gradients are finite**.

### 12. Add a safety check for `Inf`/`NaN` gradients and conditional update
Rationale: Dynamic loss scaling requires detecting overflow/invalid gradients and skipping the update when it happens.

Required refactor:
- `src/gpu_v1.4/gpu_autoencoder.hpp` (reference: `src/gpu_v1.3/gpu_autoencoder.hpp:101-105`)
  - Split gradient computation from weight update so an update can be conditionally skipped:
    - Change:
      - `void backward_pass(const float* d_dy_host_ptr, float lr);`
    - To one of the following (pick one; both satisfy dynamic scaling):
      - Option A (explicit split):
        - `void backward_compute_gradients(const float* d_dy_host_ptr);`
        - `void apply_sgd_update(float effective_lr);`
      - Option B (single function with gating):
        - `void backward_pass(const float* d_dy_host_ptr, float effective_lr, bool do_update);`

Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu`
  - Add device-side overflow flag:
    - Allocate in `alloc_all()`:
      - `int* d_found_inf_nan_;`
    - Free in `free_all()`.
  - Add kernels:
    - `check_nonfinite_kernel(const float* x, int n, int* found_inf_nan)`:
      - sets `*found_inf_nan = 1` if any `!isfinite(x[i])`
    - (optional) `clear_int_kernel(int* x)` or use `cudaMemset`.
  - In `train_step()` (reference: `src/gpu_v1.3/gpu_autoencoder.cu:676-697`):
    1. Call `mse_loss_kernel(..., loss_scale_)` to produce scaled `d_dy_`.
    2. Call `backward_compute_gradients(nullptr)` (or `backward_pass(..., do_update=false)`).
    3. Set `d_found_inf_nan_ = 0`.
    4. Run `check_nonfinite_kernel` over a minimal but sufficient set of gradient buffers, for example:
       - `d_gw1_ ... d_gw5_` and `d_gb1_ ... d_gb5_`
       - (optionally also check `d_dc5_` if it is the first scaled gradient)
    5. Copy `d_found_inf_nan_` back to host.
    6. If `found_inf_nan`:
       - Skip update.
       - Update loss scale:
         - `loss_scale_ = max(loss_scale_ * loss_scale_backoff_factor_, loss_scale_min_)`
       - Reset `loss_scale_good_steps_ = 0`.
    7. Else (finite gradients):
       - Apply update using `effective_lr = lr / loss_scale_`.
       - Increment `loss_scale_good_steps_ += 1`.
       - If `loss_scale_good_steps_ % loss_scale_growth_interval_steps_ == 0`:
         - `loss_scale_ = min(loss_scale_ * loss_scale_growth_factor_, loss_scale_max_)`

Acceptance criteria:
- With FP16 activations enabled, training does not diverge (no NaNs) for the initial epochs.
- Overflows cause skipped steps and a reduction in `loss_scale_` (observable via logging).
- Convergence remains comparable to the FP32 baseline given enough steps.

---

## Phase 4 — Manual gradient checkpointing (recompute forward segments)

Goal: reduce VRAM by not storing selected intermediate activations; instead recompute them during backward.

### 12. Define checkpoint boundaries and recomputation functions
Recommended checkpoints for this model:
- Checkpoint A: input `x`
- Checkpoint B: pooled activation `p1`
- Checkpoint C: pooled activation (latent) `p2`

Edits:
- `src/gpu_v1.4/gpu_autoencoder.hpp`
  - Add a small configuration:
    - `enum class CheckpointMode { none, stage_boundaries };`
    - `CheckpointMode checkpoint_mode_;`
  - Add private helpers:
    - `void forward_encoder_to_p1(/* writes p1 + mask1 */);`
    - `void forward_encoder_p1_to_p2(/* writes p2 + mask2 */);`
    - `void forward_decoder_from_p2(/* writes c3/u1/c4/u2/c5 */);`

### 13. Refactor `forward_pass()` into segments
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu` (reference: `src/gpu_v1.3/gpu_autoencoder.cu:571-606`)
  - Replace the monolithic forward with calls to the segment helpers.
  - When `checkpoint_mode_ == stage_boundaries`:
    - Do not preserve non-checkpoint activations beyond what is required for immediate computation.
    - Keep only checkpoint buffers (`p1`, `p2`) and masks (either stored or recomputed).

Concrete “what changes” example:
- From:
  - `launch_conv2d_relu_tiled(...); maxpool2x2_forward_kernel(...); launch_conv2d_relu_tiled(...); ...`
- To:
  - `forward_encoder_to_p1(); forward_encoder_p1_to_p2(); forward_decoder_from_p2();`

### 14. Modify `backward_pass()` to recompute segments on demand
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu` (reference: `src/gpu_v1.3/gpu_autoencoder.cu` backward section)
  - Before backpropagating through a segment, recompute the needed forward activations for that segment:
    - Decoder backward:
      - recompute `c3/u1/c4/u2/c5` from checkpoint `p2`
    - Encoder backward:
      - recompute `c1` and `mask1` from `x`
      - recompute `c2` and `mask2` from `p1`
  - Keep gradients and weight updates in FP32 as already planned.

### 15. Remove allocations for dropped intermediates (actual VRAM savings)
Edits:
- `src/gpu_v1.4/gpu_autoencoder.cu` `alloc_all()` (reference: `src/gpu_v1.3/gpu_autoencoder.cu:465+`)
  - Only allocate:
    - checkpoint buffers (`p1`, `p2`) and masks (if not recomputed)
    - “scratch” buffers for recomputation that can be reused per segment (e.g., `c1`, `c2`, `c3`, `c4`, `u1`, `u2`) rather than persistent duplicates.
  - Ensure `free_all()` matches the new allocation set.

Acceptance criteria:
- VRAM decreases versus Phase 2 (mixed precision only).
- Training throughput remains acceptable (some slowdown from recomputation is expected).

---

## Validation checklist (smoke test)

1. Build:
   - `nvcc -std=c++17 -O2 src/gpu_v1.4/main.cu src/gpu_v1.4/gpu_autoencoder.cu -I./src/gpu_v1.4 -arch=sm_75 -o autoencoder_gpu_v1.4`
2. Run (small data / partial):
   - `./autoencoder_gpu_v1.4 cifar-10-batches-bin --keep-partial`
3. Confirm:
   - Weights written to `out-gpu-v1.4/`
   - Loss is finite; no CUDA errors; throughput/VRAM measurements print as expected.
