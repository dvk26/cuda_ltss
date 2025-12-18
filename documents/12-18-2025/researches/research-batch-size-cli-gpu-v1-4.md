---
date: 2025-12-18T12:07:29+07:00
researcher: AI Agent (Codex CLI)
git_commit: c85cb3594828e723a0dcf99250027af82d44cdc1
branch: anhkhoi-branch
repository: cuda_ltss
topic: "How to add a batch-size command line argument to gpu_v1.4"
tags: [research, cuda, gpu, gpu_v1.4, cli, batch_size]
status: complete
last_updated: 2025-12-18
last_updated_by: TheMetaSetter
---

# Research: How to add a batch-size command line argument to gpu_v1.4

**Date**: 2025-12-18T12:07:29+07:00  
**Researcher**: AI Agent (Codex CLI)  
**Git Commit**: c85cb3594828e723a0dcf99250027af82d44cdc1  
**Branch**: anhkhoi-branch  

## Research Question
Research on how to add a batch-size command line argument into `gpu_v1.4` to help experiment with different batch sizes.

## Summary
`src/gpu_v1.4/main.cu` currently hard-codes `batch_size = 64` and uses that single value for:
1) host-side batching via `DataLoader`, and 2) device-side buffer sizing and kernel indexing via `GPUAutoencoder(N_)`.

To make batch size configurable from the command line, the batch size must be parsed on the host **before** constructing `GPUAutoencoder` and `DataLoader`, because `GPUAutoencoder` allocates all device buffers using its constructor’s `batch_size` and `set_input()` performs a fixed-size H2D copy based on `N_`.

## Detailed Findings

### Host-Side Implementation
**Where batch size is defined and used today**
- `src/gpu_v1.4/main.cu:47` hard-codes `int batch_size = 64;`.
- That value is used to construct:
  - the GPU model: `GPUAutoencoder gpu_ae(batch_size, 32, 32);` (`src/gpu_v1.4/main.cu:57`)
  - the loaders: `DataLoader(..., batch_size, ...)` (`src/gpu_v1.4/main.cu:60`, `src/gpu_v1.4/main.cu:61`)
- The training loop explicitly skips any batch whose `Tensor.N()` does not equal `batch_size` (`src/gpu_v1.4/main.cu:81` to `src/gpu_v1.4/main.cu:84`). The test loop does the same (`src/gpu_v1.4/main.cu:134`).

**Why incomplete batches happen**
- `DataLoader::next()` emits `cur_batch_size = min(batch_size_, remaining)` (`src/include/dataset.hpp:170` to `src/include/dataset.hpp:176`), so the final batch is often smaller than `batch_size_`.

**Existing CLI parsing pattern in this repo**
- v1.4 currently parses only `--keep-partial` and rejects unknown flags (`src/gpu_v1.4/main.cu:22` to `src/gpu_v1.4/main.cu:31`).
- Other GPU entrypoints use the same “scan argv in a `while (i < argc)` loop” pattern (example: `src/gpu_v2/main.cu:50` to `src/gpu_v2/main.cu:61`).

**Where a `--batch-size` argument would fit**
- The natural integration point is the existing `while (i < argc)` loop in `src/gpu_v1.4/main.cu` (currently only recognizes `--keep-partial`).
- The usage string would need to mention the new option (`src/gpu_v1.4/main.cu:15`).

### Device-Side Implementation (Kernels + Model)
**Batch size is a construction-time constant for `GPUAutoencoder`**
- `GPUAutoencoder::GPUAutoencoder(int batch_size, ...)` stores `batch_size` in the member `N_` (`src/gpu_v1.4/gpu_autoencoder.cu:697`).
- `alloc_all()` sizes device buffers using `nchw_size(N_, ...)` (`src/gpu_v1.4/gpu_autoencoder.cu:731` to `src/gpu_v1.4/gpu_autoencoder.cu:741` and many subsequent allocations).

**Fixed-size H2D copy assumes `x_host` matches `N_`**
- `set_input()` copies exactly `nchw_size(N_, 3, H_, W_)` floats from the host tensor into `d_x_` (`src/gpu_v1.4/gpu_autoencoder.cu:825` to `src/gpu_v1.4/gpu_autoencoder.cu:830`).
- The training loop’s “skip incomplete batch” check on the host (`src/gpu_v1.4/main.cu:81` to `src/gpu_v1.4/main.cu:84`) prevents calling `train_step()` with a tensor whose `N()` does not match the model’s `N_`.

**Kernels index batch using `N`/`n`**
Several kernels use the batch dimension (`n`) as part of their indexing, for example the conv forward kernels map `blockIdx.z` to `(n, c_out)` (`src/gpu_v1.4/gpu_autoencoder.cu:112` to `src/gpu_v1.4/gpu_autoencoder.cu:116`). This mapping is configured via wrapper launches that set `grid.z = N * Cout`.

**Loss uses `N_` to determine total elements**
- `train_step()` computes `total = N_ * 3 * H_ * W_` and launches `mse_loss_kernel` over `total` elements (`src/gpu_v1.4/gpu_autoencoder.cu:948` to `src/gpu_v1.4/gpu_autoencoder.cu:953`).

### Build & Configuration
- The v1.4 notebook compiles with:
  - `nvcc -std=c++17 -O2 src/gpu_v1.4/main.cu src/gpu_v1.4/gpu_autoencoder.cu -I./src/gpu_v1.4 -arch=sm_75 -o autoencoder_gpu_v1.4` (`ltss_gpu_v1_4.ipynb:101`)

## Code References
- `src/gpu_v1.4/main.cu:15` - usage string (currently only `--keep-partial`)
- `src/gpu_v1.4/main.cu:22` - argv scanning loop begins
- `src/gpu_v1.4/main.cu:47` - hard-coded `batch_size`
- `src/gpu_v1.4/main.cu:57` - `GPUAutoencoder` constructed with `batch_size`
- `src/gpu_v1.4/main.cu:81` - skip incomplete batch
- `src/include/dataset.hpp:170` - `cur_batch_size = min(batch_size_, remaining)`
- `src/gpu_v1.4/gpu_autoencoder.cu:697` - `N_(batch_size)` in constructor
- `src/gpu_v1.4/gpu_autoencoder.cu:731` - allocations depend on `N_`
- `src/gpu_v1.4/gpu_autoencoder.cu:825` - fixed-size host-to-device copy uses `N_`
- `src/gpu_v1.4/gpu_autoencoder.cu:112` - batch/channel mapping via `blockIdx.z`

## Architecture Documentation
In `gpu_v1.4`, “batch size” is a single scalar on the host, used consistently to:
- define how many samples `DataLoader` returns per `Batch` (except the final partial batch),
- define the model’s fixed batch dimension `N_` used by all device buffers (`d_x_`, FP16 activations, masks, gradients),
- define kernel grid shapes and indexing (notably `grid.z = N * Cout` in conv kernels and `total = N * 3 * H * W` in MSE).

## Historical Context (from documents/)
- `documents/12-16-2025/researches/research_mixed_precision_and_checkpointing.md` notes that the training loop skips incomplete batches because the GPU pipeline assumes a fixed `batch_size`.
- `documents/12-18-2025/researches/research-gpu-v1-4-slower-than-v1-3.md` uses “same batch size” comparisons as a baseline for v1.3 vs v1.4 behavior.

## Open Questions
- Should the CLI accept only “exact” batches (keep the current “skip incomplete last batch” behavior), or should v1.4 run the final partial batch by padding/reallocating?
- Do you want the batch size argument to affect only training, or both training and final test evaluation (today it affects both via shared `batch_size`)?

