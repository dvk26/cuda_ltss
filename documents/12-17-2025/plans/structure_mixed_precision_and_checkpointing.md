## Overview
Create a new `src/gpu_v1.4/` variant that implements Mixed-Precision Training specifically optimized for Tesla T4 (Tensor Cores utilization).

## Implementation Phases:

1.  **Baseline Replication & Metrics:**
    * Copy `src/gpu_v1.3/` to `src/gpu_v1.4/`.
    * Verify bit-wise (or acceptable tolerance) output equivalence with v1.3 in FP32.
    * Measure baseline: Training time per epoch, Kernel execution time, and Peak VRAM usage.

2.  **Mixed Precision Integration (Storage & Compute):**
    * **Data Layout:** Implement FP16 buffers for Activations and Gradients. Keep Master Weights and Optimizer State in FP32.
    * **Compute:** Modify CUDA kernels to perform matrix multiplications using **Tensor Cores** (FP16 input, FP32 accumulator).
    * **Cast Ops:** Insert kernels for casting: FP32 Weights -> FP16 (before Forward) and FP16 Gradients -> FP32 (before Optimizer step).
    * *Benchmark 1:* Check Throughput improvement (expecting speedup).

3.  **Numerical Stability (Dynamic Loss Scaling):**
    * Implement **Loss Scaling** workflow: Scale Loss -> Backward -> Unscale Gradients.
    * Implement **Safety Check:** Before updating Master Weights, create a reduction kernel to check for `Inf`/`NaN` in the gradients.
        * If `Inf`/`NaN` found: Skip weight update, reduce scale factor.
        * If Clean: Update weights, potentially increase scale factor (if stable for N steps).
    * *Benchmark 2:* Verify convergence matches the FP32 baseline.

4.  **Manual Gradient Checkpointing (Memory Optimization):**
    * Define checkpoint boundaries to drop intermediate activations.
    * Implement **Recomputation Logic**: Re-run specific Forward segments during Backward pass using the **same FP16 kernels** as Phase 2.
    * *Benchmark 3:* Measure VRAM reduction and the trade-off in training time.