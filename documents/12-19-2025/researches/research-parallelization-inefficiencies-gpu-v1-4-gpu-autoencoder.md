---
date: 2025-12-19T15:44:10+07:00
researcher: AI Agent (Codex CLI)
git_commit: 20f04d44c04ffedfc77ab70ba5fc45d497220bb1
branch: anhkhoi-branch
repository: cuda_ltss
topic: "Parallelization inefficiencies inside src/gpu_v1.4/gpu_autoencoder.cu (atomicAdd, serialization, synchronization)"
tags: [research, cuda, gpu, gpu_v1.4, performance, atomicAdd, synchronization]
status: complete
last_updated: 2025-12-19
last_updated_by: TheMetaSetter
---

# Research: Parallelization inefficiencies inside src/gpu_v1.4/gpu_autoencoder.cu (atomicAdd, serialization, synchronization)

**Date**: 2025-12-19T15:44:10+07:00  
**Researcher**: AI Agent (Codex CLI)  
**Git Commit**: 20f04d44c04ffedfc77ab70ba5fc45d497220bb1  
**Branch**: anhkhoi-branch

## Research Question
parallelization inefficiencies inside src/gpu_v1.4/gpu_autoencoder.cu (such as serialization caused by atomicAdd, etc.)

## Summary
The v1.4 CUDA path uses several kernels that serialize work via global atomics and block-level synchronization. The most visible sources are: global `atomicAdd` in loss reduction, in convolution weight/bias gradient accumulation, and in maxpool backward accumulation. The tiled convolution kernels also enforce per-channel phases with `__syncthreads()`, which serializes each block through load/compute steps for every input channel. On the host side, explicit `cudaDeviceSynchronize()` calls between major stages serialize the kernel stream per training step.

## Detailed Findings

### Host-Side Implementation
- **Kernel sequencing and sync points**: `GPUAutoencoder::forward_pass` launches multiple kernels and then calls `cudaDeviceSynchronize()` (`src/gpu_v1.4/gpu_autoencoder.cu:857`). `GPUAutoencoder::backward_compute_gradients` synchronizes after the backward chain (`src/gpu_v1.4/gpu_autoencoder.cu:941`). `GPUAutoencoder::apply_sgd_update` synchronizes after the weight updates (`src/gpu_v1.4/gpu_autoencoder.cu:952`). These sync points force completion of prior work before the host continues.
- **Per-step kernel fan-out**: `train_step` calls `mse_loss_kernel`, then runs the full backward chain and multiple per-buffer scans (`src/gpu_v1.4/gpu_autoencoder.cu:960`). This keeps the entire training step serialized in a single stream (no explicit streams are used in this file).

### Device-Side Implementation (Kernels)
- **`mse_loss_kernel` (loss + dY)**: Uses per-block shared-memory reduction with `__syncthreads()` in a loop, then a single `atomicAdd` to the global loss accumulator (`src/gpu_v1.4/gpu_autoencoder.cu:52`, `src/gpu_v1.4/gpu_autoencoder.cu:55`, `src/gpu_v1.4/gpu_autoencoder.cu:64`). The global accumulation step serializes updates across blocks.
- **`conv2d_backward_filter_kernel`**: Each thread handles one output gradient element and then loops over `Cin` and 3x3 kernel positions, performing `atomicAdd` into `gb` and `gW` (`src/gpu_v1.4/gpu_autoencoder.cu:441`, `src/gpu_v1.4/gpu_autoencoder.cu:454`). These atomics serialize updates to shared weight/bias gradients across many threads.
- **`conv2d_backward_filter_kernel_fp16x`**: Same atomic accumulation pattern for `gb` and `gW` (`src/gpu_v1.4/gpu_autoencoder.cu:477`, `src/gpu_v1.4/gpu_autoencoder.cu:488`).
- **`maxpool2x2_backward_kernel`**: Writes input gradients using `atomicAdd` (`src/gpu_v1.4/gpu_autoencoder.cu:572`). This introduces atomic serialization for gradient accumulation into `dX`.
- **Tiled forward conv kernels** (`conv2d_forward_tiled_kernel`, `conv2d_relu_forward_tiled_kernel`, FP16 variants): Each input channel is processed in a loop with two `__syncthreads()` per channel (load -> compute -> barrier) (`src/gpu_v1.4/gpu_autoencoder.cu:126`, `src/gpu_v1.4/gpu_autoencoder.cu:152`, `src/gpu_v1.4/gpu_autoencoder.cu:165`, `src/gpu_v1.4/gpu_autoencoder.cu:201`, `src/gpu_v1.4/gpu_autoencoder.cu:213`, `src/gpu_v1.4/gpu_autoencoder.cu:222`). This makes each block advance channel-by-channel with explicit barriers.
- **`conv2d_backward_data_tiled_kernel`**: The backward data kernel similarly loops over output channels with `__syncthreads()` between shared-memory load and compute phases (`src/gpu_v1.4/gpu_autoencoder.cu:347`, `src/gpu_v1.4/gpu_autoencoder.cu:407`). The output is written once per thread without atomics.

### Build & Configuration
- No build system logic exists in this file. The repository uses direct `nvcc` invocations; see `documents/12-18-2025/plans/detail-memory-aliasing.md` for an example build command (`nvcc -std=c++17 -O2 src/gpu_v1.4/main.cu src/gpu_v1.4/gpu_autoencoder.cu -I./src/gpu_v1.4 -arch=sm_75 -o autoencoder_gpu_v1_4`).

## Code References
- `src/gpu_v1.4/gpu_autoencoder.cu:25` - `mse_loss_kernel` definition
- `src/gpu_v1.4/gpu_autoencoder.cu:55` - Block-level loss reduction with `__syncthreads()`
- `src/gpu_v1.4/gpu_autoencoder.cu:64` - Global loss `atomicAdd`
- `src/gpu_v1.4/gpu_autoencoder.cu:422` - `conv2d_backward_filter_kernel`
- `src/gpu_v1.4/gpu_autoencoder.cu:441` - Bias `atomicAdd`
- `src/gpu_v1.4/gpu_autoencoder.cu:454` - Weight `atomicAdd`
- `src/gpu_v1.4/gpu_autoencoder.cu:461` - `conv2d_backward_filter_kernel_fp16x`
- `src/gpu_v1.4/gpu_autoencoder.cu:488` - Weight `atomicAdd` (FP16 input)
- `src/gpu_v1.4/gpu_autoencoder.cu:558` - `maxpool2x2_backward_kernel`
- `src/gpu_v1.4/gpu_autoencoder.cu:572` - Maxpool `atomicAdd`
- `src/gpu_v1.4/gpu_autoencoder.cu:857` - `forward_pass` sync
- `src/gpu_v1.4/gpu_autoencoder.cu:941` - `backward_compute_gradients` sync
- `src/gpu_v1.4/gpu_autoencoder.cu:952` - `apply_sgd_update` sync

## Architecture Documentation
The v1.4 pipeline runs entirely on GPU per step: input is copied to device, converted to FP16, then fused conv+ReLU + pooling/upsampling kernels produce `d_c5_`. The `mse_loss_kernel` computes both loss and `dY` on device, followed by a backward chain that alternates conv-backward, upsample-backward, ReLU-backward, and maxpool-backward kernels. Weight and bias gradients are accumulated in-place with global atomics, then a separate SGD update kernel applies parameter updates.

## Historical Context (from documents/)
- `documents/12-18-2025/plans/detail-memory-aliasing.md` notes that `maxpool2x2_backward_kernel` uses `atomicAdd` into `dX` and therefore requires zeroing of `dX` before launch.
- `documents/12-18-2025/researches/research-gpu-v1-4-slower-than-v1-3.md` documents additional v1.4 per-step synchronization points (forward, backward, update), which align with the syncs observed here.

## Open Questions
- Is `maxpool2x2_backward_kernel` expected to have overlapping writes to `dX` in this architecture (stride-2 maxpool), or is the `atomicAdd` retained mainly for correctness safety?
- Are any kernels launched on non-default streams elsewhere in the codebase, or is the entire training step serialized on the default stream only?
