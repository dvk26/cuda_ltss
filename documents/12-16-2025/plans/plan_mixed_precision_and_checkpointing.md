---
date: 2025-12-16
topic: "Implementation plan: mixed precision + checkpointing (src/gpu_v1.3)"
repository: cuda_ltss
---

Based on my research, here's what I found:

## Current State
- `src/gpu_v1.3` is FP32 end-to-end: weights, activations, gradients are all `float*`, and there are no `__half` buffers/kernels/includes.
- No gradient checkpointing: `alloc_all()` permanently allocates every activation + mask buffer; backward reads them directly. Also, forward does redundant `d_c* -> d_r*` copies purely to satisfy ReLU backward.

## Design Options
- Option A: **Memory-first (recommended to start)** — FP16 activations + FP32 compute/weights (AMP-lite) + manual checkpointing at stage boundaries (recompute forward segments during backward).
- Option B: **Performance-first** — FP16 activations + FP16 weight “compute copies” + FP32 master weights + dynamic loss scaling (and optional `half2`/Tensor Core work), plus checkpointing if VRAM still limits batch size.

## Open Questions
- Is the primary goal **VRAM reduction (bigger batch)**, **speed**, or both—and what GPU `-arch` are you targeting?
- Do you want this added as **runtime flags** (e.g., `--amp`, `--checkpoint=minimal`) or as **separate variant directories** (e.g., `src/gpu_v1.4/`)?

Which approach aligns best with your vision?

