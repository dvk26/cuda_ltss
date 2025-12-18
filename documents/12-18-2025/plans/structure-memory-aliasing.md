## Overview
Reduce GPU VRAM usage during backprop by replacing per-layer activation-gradient buffers with a static 2-buffer “ping-pong” pool, while preserving correctness and keeping the v1.4 training flow unchanged.

## Implementation Phases
1. **Scope & Lifetime Audit** – Identify which `d_*` buffers are true activation/feature-map gradients eligible for reuse vs. which must persist across layers/steps (e.g., weight gradients, optimizer state), and confirm no kernel requires “old” activation gradients after the next layer consumes them.

2. **Pool Sizing & Allocation** – In `alloc_all()`, compute the maximum required element count across eligible gradient tensors (prefer dynamic sizing from known layer shapes), allocate `Pool_A`/`Pool_B` once, and remove/skip individual `cudaMalloc` calls for those eligible per-layer gradient buffers.

3. **Pointer Aliasing Contract** – Redirect existing member pointers (e.g., `d_dc5_`, `d_du2_`, …) to views of `Pool_A` or `Pool_B` based on the backward traversal order; define a consistent “current grad” / “next grad” mapping so kernels can remain largely unchanged.

4. **Backward Pass Integration** – Update `backward_compute_gradients()` to swap pools at the exact boundaries where a layer no longer needs the incoming gradient, ensuring each kernel reads the correct “upstream” gradient while writing the “downstream” gradient into the alternate pool.

5. **Validation & Benchmarking** – Run a correctness check (loss curve / reconstruction samples) and measure peak VRAM before/after (same batch size), documenting the achieved memory reduction and any performance impact.

Does this phasing make sense, or should I adjust the order/granularity (e.g., split kernel-compatibility checks into a dedicated phase)?
