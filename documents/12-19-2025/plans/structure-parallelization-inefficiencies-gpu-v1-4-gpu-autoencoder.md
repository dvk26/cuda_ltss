## Overview
Outline a phased plan to overlap H2D transfers with compute, reduce sync overhead in the training loop, and improve parallel efficiency in the three target conv kernels without touching maxpool backward.

## Implementation Phases
3. Streamed batch pipeline – introduce dual-stream H2D + compute flow with events to overlap batch N+1 transfers and batch N execution.
4. Training loop synchronization cleanup – audit and remove unnecessary synchronizations, keeping syncs only for loss logging and epoch boundaries.
5. Kernel parallelism upgrades – rework the three target kernels to reduce global atomics and per-channel syncs while preserving FP16 IO paths.
6. Workspace + determinism decisions – define scratch buffer sizing, reuse strategy, and reduction ordering trade-offs.
7. Validation + performance smoke checks – compile/run with `--keep-partial`, verify outputs, and capture basic timing deltas.

Does this phasing make sense? Should I adjust the order or granularity?
