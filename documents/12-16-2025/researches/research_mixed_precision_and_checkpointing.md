---
date: 2025-12-16T14:29:52+07:00
researcher: Codex CLI by OpenAI
git_commit: ef02d6f622d4645b70e3b3e778da1bf8a3254521
branch: main
repository: cuda_ltss
topic: "Mixed precision training and gradient checkpointing status (src/gpu_v1.3)"
tags: [research, cuda, autoencoder, mixed-precision, fp16, checkpointing]
status: complete
last_updated: 2025-12-16
last_updated_by: Codex CLI by OpenAI
---

# Research: Mixed precision training and gradient checkpointing status (src/gpu_v1.3)

## Research Question
What is the current status of (1) mixed precision / FP16 forward and (2) gradient checkpointing in `src/gpu_v1.3`, and where are the precision and activation-lifetime decisions in the current implementation?

## Summary (as-is)
- **Mixed precision / FP16 forward**: not present; device buffers and kernels are FP32 (`float`) throughout.
- **Gradient checkpointing**: not present; intermediate activations and pooling masks are stored in persistent device buffers and read directly during backprop.

## Implementation Plan (proposed)
Goal: add (A) mixed precision training and (B) gradient checkpointing to the `src/gpu_v1.3` CUDA autoencoder while keeping host-side tensors and dataset IO in FP32.

### Phase 0 — Baseline + guardrails
- Add lightweight NaN/inf checks (e.g., track `max_abs` of a few key buffers) and optional `cudaMemGetInfo()` logging around `alloc_all()` to quantify memory savings.

### Phase 1 — Remove redundant activation copies (quick win)
Current `forward_pass()` copies `d_c{1,2,3,4}_ -> d_r{1,2,3,4}_` only because ReLU backward reads `d_r*`. Since the fused Conv+ReLU kernels already produce post-ReLU values, backward can gate on `d_c*` instead.
- Update ReLU backward calls to read from `d_c*` directly.
- Remove those device-to-device copies and the `d_r*` allocations (and any now-unused gradient buffers).
- Smoke-check: loss should match FP32 baseline.

### Phase 2 — Mixed precision (AMP)
Two viable scopes; both can live behind a flag (compile-time macro or runtime config passed into `GPUAutoencoder`).

#### MP-A: FP16 activations, FP32 compute/weights (memory-first, low risk)
Store most activations as `__half` to halve activation memory, but keep math/accumulation and all gradients/weight updates in FP32.
- Add `<cuda_fp16.h>` and a config flag (`use_fp16_activations`).
- Convert activation buffers (likely `d_c1_..d_u2_`) to `__half*`; keep `d_x_`, `d_c5_`, loss buffers, weights, and all gradient buffers as `float*`.
- Update forward kernels to:
  - Load activations as `__half` (cast to `float`), accumulate in `float`, and write activation outputs as `__half`.
  - Keep final output (`d_c5_`) as `float` for MSE and logging.
- Update backward kernels to:
  - Keep gradients in `float`.
  - Accept FP16 activations as inputs and cast to `float` where needed (conv backward/filter backward; ReLU gating).
- Validate: compare loss curves FP32 vs MP-A for a short run (expect small drift, no divergence).

#### MP-B: FP16 activations + FP16 compute weights (+ FP32 master weights) (performance-first)
Higher effort: maintain FP32 master weights, FP16 “compute copies”, and dynamic loss scaling to prevent underflow/overflow.
- Add FP16 weight copies (`d_w*_h`) and loss scaling logic.
- Optional: add `half2` vectorization for elementwise kernels; Tensor Core use likely requires a larger rework (conv→GEMM/im2col or WMMA).

### Phase 3 — Gradient checkpointing (activation rematerialization)
Trade compute for memory by recomputing forward activations during backward instead of storing all of them.

#### GC-A: Manual checkpoints at stage boundaries (recommended first)
- Add `CheckpointPolicy` (runtime): `none` (current), `encoder_only`, `minimal`.
- Refactor forward into explicit stages:
  - `forward_encoder()` (up to `p2`)
  - `forward_decoder_from_p2()` (from `p2` to `c5`)
- During backward, if an activation isn’t stored, recompute the minimum forward prefix into scratch buffers before that layer’s backward.
- Decide on pooling masks:
  - Recompute masks on-demand (deterministic unless tie-breaking differs), or store masks as checkpoints if ties are a concern.
- Memory strategy: replace many persistent activation buffers with 1–2 scratch buffers sized for the largest tensor in the recomputed segment.

#### GC-B: Buffer reuse (no recomputation)
Implement a simple lifetime-based scratch allocator to reuse large buffers across forward ops; simpler than GC-A but offers less memory reduction because backward still needs distinct values.

### Phase 4 — Integration (AMP + checkpointing)
- Ensure recomputation uses the same precision mode as the original forward for that step.
- Keep weight updates after the full backward (current behavior), otherwise recomputation would be incorrect.

## Detailed Findings

### Host-side control flow (training + eval)
- Training uses `GPUAutoencoder::train_step(x, lr)` per batch (`src/gpu_v1.3/main.cu:87`), which returns a single scalar loss to the host.
- Evaluation uses `GPUAutoencoder::compute_loss(x_test)` (`src/gpu_v1.3/main.cu:127`), which also returns a scalar loss.
- The loop skips incomplete batches because the GPU pipeline assumes fixed `batch_size` (`src/gpu_v1.3/main.cu:78`).

### GPUAutoencoder: buffers and precision
- The model stores weights, activations, and gradients as `float*` device pointers (`src/gpu_v1.3/gpu_autoencoder.hpp:61`).
- Pooling masks are stored as `int*` (`src/gpu_v1.3/gpu_autoencoder.hpp:80`).
- v1.3 includes full-GPU loss/gradient buffers: `d_dy_` (output gradient computed on GPU) and `d_loss_accum_` (single-float accumulator) (`src/gpu_v1.3/gpu_autoencoder.hpp:92`).

### Full-GPU training step (Forward -> MSE -> Backward -> Update)
- `GPUAutoencoder::train_step` runs `set_input` (H2D copy), `forward_pass`, then launches `mse_loss_kernel` to compute both `dY` and loss on GPU, then calls `backward_pass(nullptr, lr)` to use the already-computed device gradient (`src/gpu_v1.3/gpu_autoencoder.cu:676`, `src/gpu_v1.3/gpu_autoencoder.cu:686`, `src/gpu_v1.3/gpu_autoencoder.cu:690`).
- `mse_loss_kernel` writes `dY[idx] = (2 * (pred - target)) / total_elements` and reduces loss within a block into shared memory before `atomicAdd` into `loss_out` (`src/gpu_v1.3/gpu_autoencoder.cu:24`, `src/gpu_v1.3/gpu_autoencoder.cu:33`, `src/gpu_v1.3/gpu_autoencoder.cu:61`).
- `GPUAutoencoder::compute_loss` follows the same forward + MSE path but does not run backprop (`src/gpu_v1.3/gpu_autoencoder.cu:700`).

### Mixed precision / FP16 forward: current status
- All data paths are FP32: parameters, activations, and gradients are `float*` (`src/gpu_v1.3/gpu_autoencoder.hpp:61`).
- Forward uses float kernels and writes float activations (`src/gpu_v1.3/gpu_autoencoder.cu:571`).
- Backward consumes float activations (including `d_r*` ReLU outputs) and updates float weights via `sgd_update_kernel` (`src/gpu_v1.3/gpu_autoencoder.cu:614`, `src/gpu_v1.3/gpu_autoencoder.cu:660`).
- There are no `__half` buffers, `cuda_fp16.h` includes, or FP16-specific kernels in `src/gpu_v1.3` (no `__half`/`fp16` tokens in the directory).

### Gradient checkpointing: current status (activation lifetime)
- `alloc_all` allocates dedicated buffers for each activation and gradient buffer, plus pooling masks, and retains them for the lifetime of the `GPUAutoencoder` instance (`src/gpu_v1.3/gpu_autoencoder.cu:465`).
- `forward_pass` writes each stage output into these persistent buffers, including copies from fused conv outputs into `d_r*` buffers for use by ReLU backward (`src/gpu_v1.3/gpu_autoencoder.cu:571`, `src/gpu_v1.3/gpu_autoencoder.cu:575`).
- `backward_pass` reads stored activations and masks directly (e.g., Conv5 backward reads `d_u2_`, ReLU backward reads `d_r4_`, maxpool backward reads `d_mask*`) (`src/gpu_v1.3/gpu_autoencoder.cu:642`, `src/gpu_v1.3/gpu_autoencoder.cu:644`, `src/gpu_v1.3/gpu_autoencoder.cu:651`).

## Notes on `src/gpu_v1.3/gpu_autocoder_run.cu`
`src/gpu_v1.3/gpu_autocoder_run.cu` contains a second set of similar kernel definitions; whether it is used depends on how the project is compiled (there is no build system in-repo). The primary entrypoint in this variant is `src/gpu_v1.3/main.cu` which includes `src/gpu_v1.3/gpu_autoencoder.hpp`.

## Practical next decision points
- If the goal is primarily **VRAM reduction / bigger batches**, start with **Phase 1 + MP-A**, then add **GC-A**.
- If the goal is primarily **throughput**, stabilize **MP-A** first, then evaluate whether MP-B complexity is worth it.
