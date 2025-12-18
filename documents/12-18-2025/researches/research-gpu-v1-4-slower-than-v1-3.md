---
date: 2025-12-18T11:27:42+07:00
researcher: AI Agent (Codex CLI)
git_commit: c85cb3594828e723a0dcf99250027af82d44cdc1
branch: anhkhoi-branch
repository: cuda_ltss
topic: "Differences between gpu_v1.4 and gpu_v1.3 that can make gpu_v1.4 slower at the same batch size"
tags: [research, cuda, gpu, gpu_v1.3, gpu_v1.4, performance]
status: complete
last_updated: 2025-12-18
last_updated_by: TheMetaSetter
---

# Research: Differences between gpu_v1.4 and gpu_v1.3 that can make gpu_v1.4 slower at the same batch size

**Date**: 2025-12-18T11:27:42+07:00  
**Researcher**: AI Agent (Codex CLI)  
**Git Commit**: c85cb3594828e723a0dcf99250027af82d44cdc1  
**Branch**: anhkhoi-branch  

## Research Question
Compare `src/gpu_v1.4/` vs `src/gpu_v1.3/` and identify the concrete host/device behavior differences that can explain why v1.4 runs slower than v1.3 at the same `batch_size`.

## Summary
The v1.4 variant adds several per-batch/per-step behaviors that increase host-side work, kernel count, and synchronization frequency compared to v1.3:

1. **Host-side peak VRAM sampling per batch**: v1.4 calls `cudaMemGetInfo()` inside the batch loop and updates `peak_used_bytes` (`src/gpu_v1.4/main.cu:95`). v1.3 does not sample VRAM inside the batch loop (`src/gpu_v1.3/main.cu:74`).
2. **Mixed-precision activation path (FP16 activations)**: v1.4 copies input to `d_x_` (FP32) and then launches `fp32_to_fp16_kernel` to produce `d_xh_` (FP16) every batch (`src/gpu_v1.4/gpu_autoencoder.cu:825`). v1.3 only does the FP32 H2D copy (`src/gpu_v1.3/gpu_autoencoder.cu:562`).
3. **Manual checkpointing via recomputation during backward**: v1.4 sets `checkpoint_mode_ = stage_boundaries` (`src/gpu_v1.4/gpu_autoencoder.cu:700`) and explicitly re-runs forward sub-stages inside `backward_compute_gradients` (`src/gpu_v1.4/gpu_autoencoder.cu:890`, `src/gpu_v1.4/gpu_autoencoder.cu:904`, `src/gpu_v1.4/gpu_autoencoder.cu:913`). v1.3 stores intermediate activations and does not do this style of recomputation.
4. **Dynamic loss scaling + nonfinite gradient checks**: v1.4 scales `dY` inside `mse_loss_kernel` (`src/gpu_v1.4/gpu_autoencoder.cu:952`), then launches `check_nonfinite_kernel` over every weight/bias gradient buffer and copies back an `int` flag to host to decide whether to apply the update (`src/gpu_v1.4/gpu_autoencoder.cu:960`, `src/gpu_v1.4/gpu_autoencoder.cu:973`). v1.3 always applies the SGD update path in-device, without these extra checks (`src/gpu_v1.3/gpu_autoencoder.cu:660`).
5. **More synchronization points per step**: v1.4 synchronizes at the end of `forward_pass`, at the end of `backward_compute_gradients`, and again at the end of `apply_sgd_update` (`src/gpu_v1.4/gpu_autoencoder.cu:841`, `src/gpu_v1.4/gpu_autoencoder.cu:922`, `src/gpu_v1.4/gpu_autoencoder.cu:933`). v1.3 synchronizes at the end of `forward_pass` and at the end of `backward_pass` (`src/gpu_v1.3/gpu_autoencoder.cu:607`, `src/gpu_v1.3/gpu_autoencoder.cu:668`).

Taken together, v1.4’s additional host API calls (VRAM sampling, device-to-host flag reads) and additional device work (FP16 conversion, recomputation, per-buffer nonfinite scans) are direct structural differences that can make v1.4 slower even if `batch_size` is held constant.

## Detailed Findings

### Host-Side Implementation (training loop differences)

#### v1.3: per-batch host flow
In v1.3, each host batch runs:
- `gpu_ae.train_step(x, lr)` (`src/gpu_v1.3/main.cu:87`)
- progress printing every 10 batches (`src/gpu_v1.3/main.cu:93`)

There is no additional CUDA runtime query inside the batch loop besides what is inside `train_step`.

#### v1.4: per-batch host flow adds VRAM sampling
In v1.4, each host batch runs the same `gpu_ae.train_step(x, lr)` call, but the host additionally samples device memory every batch:
- `cudaMemGetInfo(&free_bytes, &total_bytes)` (`src/gpu_v1.4/main.cu:97`)
- `peak_used_bytes = std::max(peak_used_bytes, total_bytes - free_bytes)` (`src/gpu_v1.4/main.cu:98`)

This means v1.4 has an extra CUDA runtime API call in the hot inner training loop.

### Device-Side Implementation (GPUAutoencoder pipeline differences)

#### v1.3: FP32 activation storage + no recomputation
Key properties visible in v1.3 forward/backward:
- `set_input` does only H2D copy into `d_x_` (`src/gpu_v1.3/gpu_autoencoder.cu:562`).
- `forward_pass` uses FP32 activations and also performs device-to-device copies `d_c* -> d_r*` for ReLU backward compatibility (`src/gpu_v1.3/gpu_autoencoder.cu:573`, `src/gpu_v1.3/gpu_autoencoder.cu:577`).
- `backward_pass` uses stored activations/masks and always runs the SGD update kernels (`src/gpu_v1.3/gpu_autoencoder.cu:614`, `src/gpu_v1.3/gpu_autoencoder.cu:661`).

#### v1.4: FP16 activation path, scratch reuse, and checkpoint recomputation
In v1.4, `GPUAutoencoder` changes activation storage and adds extra control paths:

**FP16 activations and input conversion**
- `gpu_autoencoder.hpp` introduces `__half*` activations (e.g., `d_xh_`, `d_c1_`, `d_p1_`, etc.) (`src/gpu_v1.4/gpu_autoencoder.hpp:70`).
- `set_input` performs H2D to `d_x_` and then launches `fp32_to_fp16_kernel` each batch (`src/gpu_v1.4/gpu_autoencoder.cu:827`, `src/gpu_v1.4/gpu_autoencoder.cu:829`).

**Scratch buffers and aliasing (activation lifetime change)**
v1.4 allocates scratch buffers and aliases multiple activation pointers onto them:
- `d_scratch256_hw_`, `d_scratch256_h1w1_`, `d_scratch128_h1w1_`, `d_scratch128_h2w2_` (`src/gpu_v1.4/gpu_autoencoder.hpp:88`)
- aliasing examples: `d_c1_ = d_scratch256_hw_` and `d_u2_ = d_scratch256_hw_` (`src/gpu_v1.4/gpu_autoencoder.cu:743`, `src/gpu_v1.4/gpu_autoencoder.cu:744`)

This reduces persistent activation storage but means earlier activations may not still exist at backward time.

**Checkpoint mode triggers recomputation**
v1.4 sets `checkpoint_mode_` to `stage_boundaries` in the constructor (`src/gpu_v1.4/gpu_autoencoder.cu:700`).
During backward, it conditionally recomputes forward segments:
- recompute decoder segment: `forward_decoder_from_p2()` (`src/gpu_v1.4/gpu_autoencoder.cu:891`)
- recompute late encoder segment: `forward_encoder_p1_to_p2()` (`src/gpu_v1.4/gpu_autoencoder.cu:905`)
- recompute early encoder segment: `forward_encoder_to_p1()` (`src/gpu_v1.4/gpu_autoencoder.cu:914`)

This introduces additional kernel launches and device work per training step compared to v1.3’s “store activations once” approach.

#### v1.4: dynamic loss scaling and per-step safety checks
v1.4’s `train_step` adds a “scale/check/conditional-update” loop:
- `mse_loss_kernel` receives `loss_scale_` and writes scaled `dY` (`src/gpu_v1.4/gpu_autoencoder.cu:952`).
- Gradient nonfinite scanning kernels run across each gradient buffer (`src/gpu_v1.4/gpu_autoencoder.cu:960` through `src/gpu_v1.4/gpu_autoencoder.cu:969`).
- A device-to-host copy reads the `found_inf_nan` flag (`src/gpu_v1.4/gpu_autoencoder.cu:973`), which is used for conditional update and scale adjustment (`src/gpu_v1.4/gpu_autoencoder.cu:975`).

v1.3’s `train_step` path is simpler: compute loss/gradient and run `backward_pass` + SGD update without per-buffer scanning or conditional update branching (`src/gpu_v1.3/gpu_autoencoder.cu:676`).

### Build & Configuration
Both notebooks build with `nvcc -std=c++17 -O2 -arch=sm_75`, but the include path differs:
- v1.3 notebook uses `-I./include` (`ltss_gpu_v1_3.ipynb:99`)
- v1.4 notebook uses `-I./src/gpu_v1.4` (`ltss_gpu_v1_4.ipynb:101`)

## Code References
- `src/gpu_v1.4/main.cu:97` - `cudaMemGetInfo()` sampled per batch
- `src/gpu_v1.4/gpu_autoencoder.cu:829` - per-batch `fp32_to_fp16_kernel` launch
- `src/gpu_v1.4/gpu_autoencoder.cu:700` - `checkpoint_mode_ = CheckpointMode::stage_boundaries`
- `src/gpu_v1.4/gpu_autoencoder.cu:891` - recompute decoder stage in backward
- `src/gpu_v1.4/gpu_autoencoder.cu:905` - recompute encoder stage in backward
- `src/gpu_v1.4/gpu_autoencoder.cu:914` - recompute early encoder stage in backward
- `src/gpu_v1.4/gpu_autoencoder.cu:960` - `check_nonfinite_kernel` gradient scans
- `src/gpu_v1.4/gpu_autoencoder.cu:973` - device-to-host `found_inf_nan` read
- `src/gpu_v1.3/gpu_autoencoder.cu:562` - v1.3 `set_input` is H2D only
- `src/gpu_v1.3/gpu_autoencoder.cu:571` - v1.3 stores activations and does not checkpoint-recompute

## Historical Context (from documents/)
- `documents/12-17-2025/researches/research-epoch-time-peak-vram-logging.md` documents that v1.4 added per-batch `cudaMemGetInfo()` sampling for peak VRAM logging, which is not present in earlier variants.
- `documents/12-16-2025/researches/research_mixed_precision_and_checkpointing.md` describes v1.3 as FP32-only and without checkpointing; v1.4 implements the FP16-activation + checkpointing direction described in later plans.

## Open Questions
- When comparing “same batch size” between v1.3 and v1.4, are both runs using the same binary flags and the same GPU architecture target (e.g., both built with `-arch=sm_75`)?
- Is the slowdown observed in end-to-end epoch wall-clock time (includes dataloader + logging) or in the per-batch GPU compute portion only?
