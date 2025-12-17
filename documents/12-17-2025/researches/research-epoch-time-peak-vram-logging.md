---
date: 2025-12-17T22:16:46+07:00
researcher: AI Agent
git_commit: 1ab0e2177a3fc5c18fc42b09bc10e85fbb7b3e5a
branch: main
repository: cuda_ltss
topic: "Ways to add logging of epoch processing time and peak VRAM"
tags: [research, cuda, gpu, logging, epoch, vram]
status: complete
last_updated: 2025-12-17
last_updated_by: AI Agent
---

# Research: Ways to add logging of epoch processing time and peak VRAM

**Date**: 2025-12-17T22:16:46+07:00  
**Researcher**: AI Agent  
**Git Commit**: 1ab0e2177a3fc5c18fc42b09bc10e85fbb7b3e5a  
**Branch**: main

## Research Question
Research the codebase to find ways to add logging of:
- processing time for each epoch
- peak VRAM usage

## Summary
- Epoch-level wall-clock timing is already implemented in all training entrypoints via `std::chrono` around the per-epoch loop (CPU and GPU variants).
- Peak VRAM logging is already implemented in `src/gpu_v1.4/main.cu` by sampling `cudaMemGetInfo()` during the batch loop and tracking the maximum of `(total - free)`.
- For other GPU variants (`src/gpu/main.cu`, `src/gpu_v1.1/main.cu`, `src/gpu_v1.2/main.cu`, `src/gpu_v1.3/main.cu`), the most direct way to add peak VRAM is to replicate the `src/gpu_v1.4/main.cu` pattern at the same host-side location (inside the `while (train_loader.has_next())` loop).

## Detailed Findings

### Host-Side Implementation

#### Epoch timing already exists
All entrypoints wrap epoch work with `std::chrono::high_resolution_clock` and print elapsed seconds.

- CPU training loop: `src/cpu/main.cpp` prints `Epoch ... | time=...s` each epoch.
- GPU baseline training loop: `src/gpu/main.cu:99` to `src/gpu/main.cu:135` measures and prints `time=...s`.
- GPU v1.1 training loop: `src/gpu_v1.1/main.cu:61` to `src/gpu_v1.1/main.cu:96` measures and prints `time=...s`.
- GPU v1.2 training loop: `src/gpu_v1.2/main.cu:61` to `src/gpu_v1.2/main.cu:96` measures and prints `time=...s`.
- GPU v1.3 training loop: `src/gpu_v1.3/main.cu:69` to `src/gpu_v1.3/main.cu:106` measures and prints `-> Time: ...s` plus `batches/s`.
- GPU v1.4 training loop: `src/gpu_v1.4/main.cu:71` to `src/gpu_v1.4/main.cu:116` measures and prints `-> Time: ...s` plus `batches/s`.

The timing is wall-clock at the host level (includes dataloader + CPU-side work + GPU work that is synchronized by the called routines).

#### Peak VRAM logging exists in v1.4 only
`src/gpu_v1.4/main.cu` tracks “peak used bytes” by sampling `cudaMemGetInfo()` during the batch loop:
- Initialize per-epoch accumulator: `src/gpu_v1.4/main.cu:75`
- Sample and update peak: `src/gpu_v1.4/main.cu:95` to `src/gpu_v1.4/main.cu:98`
- Print peak VRAM in MiB: `src/gpu_v1.4/main.cu:115`

This approach is directly portable to other GPU main programs, since they already have a per-batch loop on the host.

### Device-Side Implementation (Synchronization context)
The epoch timing in the main programs is only meaningful if the batch operations complete before the host advances. In this codebase, the GPU implementations commonly call `cudaDeviceSynchronize()` at the end of major GPU phases:

- Baseline GPU forward pass sync: `src/gpu/gpu_autoencoder.cu:769`
- Baseline GPU backward/update sync: `src/gpu/gpu_autoencoder.cu:973`
- v1.1 forward pass sync: `src/gpu_v1.1/gpu_autoencoder.cu:518`
- v1.2 forward pass sync: `src/gpu_v1.2/gpu_autoencoder.cu:684`
- v1.4 train step / backward/update uses multiple synchronizations (example: `src/gpu_v1.4/gpu_autoencoder.cu:922`, `src/gpu_v1.4/gpu_autoencoder.cu:933`)

This makes the host-side epoch timer reflect completed GPU work (rather than just enqueued kernels), as implemented today.

### Build & Configuration
There is no centralized build system; GPU builds use `nvcc` directly (documented in `AGENTS.md`). Adding peak VRAM logging via `cudaMemGetInfo()` only requires `<cuda_runtime.h>` and no extra link dependencies beyond the CUDA runtime already used.

## How to Add the Missing Peak VRAM Logs (Patterns in this repo)

### Pattern A: sample `cudaMemGetInfo()` per batch (matches v1.4)
**Where**: inside the batch loop in each `main.cu`, after a call that synchronizes GPU work (e.g., after `gpu_ae.backward_and_update(...)` or `gpu_ae.train_step(...)`).  
**What to store**: `peak_used_bytes = max(peak_used_bytes, total_bytes - free_bytes)`.

Places to apply the same pattern:
- Baseline GPU: after `gpu_ae.backward_and_update(dY, lr)` at `src/gpu/main.cu:127`
- v1.1: after `gpu_ae.backward_and_update(dY, lr)` at `src/gpu_v1.1/main.cu:88`
- v1.2: after `gpu_ae.backward_and_update(dY, lr)` at `src/gpu_v1.2/main.cu:88`
- v1.3: after `gpu_ae.train_step(x, lr)` at `src/gpu_v1.3/main.cu:87`

### Pattern B: sample once per epoch (lower overhead, less “peak-accurate”)
**Where**: at the end of each epoch just before printing stats.  
**Limitation**: this captures the VRAM usage at that instant, not necessarily the per-batch peak within the epoch.

### Pattern C: use NVML for process-level VRAM (external dependency)
If the intent is “VRAM used by this process” rather than device-wide usage, the codebase would need to introduce NVML (`libnvidia-ml`) calls. This is not present anywhere in the repo currently (no `nvml*` symbols found).

## Code References
- `src/gpu/main.cu:96` - baseline GPU epoch loop (time logging)
- `src/gpu_v1.1/main.cu:58` - v1.1 epoch loop (time logging)
- `src/gpu_v1.2/main.cu:58` - v1.2 epoch loop (time logging)
- `src/gpu_v1.3/main.cu:66` - v1.3 epoch loop (time logging)
- `src/gpu_v1.4/main.cu:68` - v1.4 epoch loop (time + peak VRAM logging)
- `src/gpu_v1.4/main.cu:97` - `cudaMemGetInfo()` sampling for peak VRAM

## Historical Context (from thoughts/)


## Open Questions
- Should “peak VRAM” mean device-wide used memory (what `cudaMemGetInfo()` can infer), or process-specific usage (requires NVML or other tooling)?
- Is per-batch sampling acceptable overhead for the intended benchmarks, or should sampling be periodic (every N batches) or per-epoch only?

