# Detailed Implementation Plan: Parallelization Inefficiencies (gpu_v1.4)

## Overview
This plan adds dual-stream batch overlap, trims unnecessary synchronization in the training loop, and improves parallel efficiency in three conv kernels (`conv2d_backward_filter_kernel_fp16x`, `conv2d_relu_forward_tiled_kernel_fp16io`, `conv2d_backward_data_tiled_kernel`) while leaving maxpool backward unchanged.

## Phase 3: Streamed batch pipeline
### Edits
- Identify the gpu_v1.4 training entrypoint and data-loading path.
- Introduce two CUDA streams: `stream_h2d` for async host-to-device copies and `stream_compute` for kernel launches.
- Add CUDA events to signal when the next batch is ready on device.
- Convert host buffers to pinned memory (or add a pinned staging buffer) to enable `cudaMemcpyAsync`.
- Ensure the compute stream waits on the H2D event for batch N before launching kernels.

### Edit Content
- Add stream creation and destruction in the gpu_v1.4 setup/teardown path.
- Wire `cudaMemcpyAsync` for inputs/labels (batch N+1) on `stream_h2d`.
- Insert `cudaEventRecord` after H2D copy and `cudaStreamWaitEvent` before compute begins.
- Preserve existing data pipeline behavior and keep batch ordering unchanged.

### Acceptance Criteria
- H2D copies run on `stream_h2d` while kernels run on `stream_compute`.
- No functional changes to outputs for a short run.
- Streams and events are cleaned up without leaks.

## Phase 4: Training loop synchronization cleanup
### Edits
- Audit all uses of `cudaDeviceSynchronize`, `cudaStreamSynchronize`, and implicit sync points.
- Restrict synchronization to logging points where loss is read back to host.
- Keep epoch boundary synchronization only if needed for correctness or output flushing.

### Edit Content
- Move loss readback to explicit sync points only.
- Replace global syncs with stream-specific syncs when possible.
- Remove redundant syncs between consecutive kernels on the same compute stream.

### Acceptance Criteria
- Synchronization occurs only when loss values are required or at epoch boundaries.
- No race conditions between H2D and compute streams (validated by events).
- Training loop runs without extra device-wide barriers.

## Phase 5: Kernel parallelism upgrades
### Edits
- `conv2d_backward_filter_kernel_fp16x`: eliminate global atomics in the main kernel by producing per-block partial gradients.
- `conv2d_relu_forward_tiled_kernel_fp16io`: reduce per-channel `__syncthreads()` by re-tiling or unrolling channels.
- `conv2d_backward_data_tiled_kernel`: reduce per-channel sync overhead while preserving FP16 IO.

### Edit Content
- Add shared-memory tiles for filter-grad accumulation in `conv2d_backward_filter_kernel_fp16x`.
- Write per-block partials to a workspace buffer and add a reduction kernel to combine them.
- Re-tile channel loops for forward and backward data kernels to reduce the number of sync phases.
- Keep kernel interfaces and FP16 IO paths intact.

### Acceptance Criteria
- `conv2d_backward_filter_kernel_fp16x` no longer uses global atomics for the full accumulation path.
- Forward/backward data kernels reduce sync phases without correctness regressions.
- Kernel launches still succeed for current layer shapes.

## Phase 6: Workspace + determinism decisions
### Edits
- Determine per-layer workspace size for partial gradients.
- Define allocation strategy (per-layer persistent buffer vs shared scratch pool).
- Decide reduction ordering and document determinism trade-offs.

### Edit Content
- Add workspace sizing logic based on output channels, filter size, and block layout.
- Implement a buffer reuse policy to avoid frequent allocations.
- Choose deterministic or non-deterministic reduction order and document expected numeric drift.

### Acceptance Criteria
- Workspace allocation fits within expected GPU memory budget.
- Reduction strategy is consistent across runs or explicitly documented otherwise.

## Phase 7: Validation + performance smoke checks
### Edits
- Compile the gpu_v1.4 target.
- Run a short training session with `--keep-partial`.
- Capture simple timing deltas (per-epoch or per-batch) before/after changes.

### Edit Content
- Use existing run commands and record wall-clock timings.
- Confirm outputs generated in `out-gpu/`.

### Acceptance Criteria
- Build succeeds without warnings that indicate correctness risks.
- Outputs are produced and loss logging still functions.
- Basic timing indicates no regressions from added stream overhead.
