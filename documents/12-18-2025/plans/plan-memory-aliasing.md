Based on my research, here's what I found:

## Current State

* **Buffer Redundancy:** The `alloc_all()` function in `gpu_autoencoder.cu` currently allocates ~14 distinct `float*` buffers for gradients (e.g., `d_dc5_`, `d_du2_`, `d_dr4_`, etc.). Each buffer is sized for the full batch N, leading to O(\text{Layers} \times N) VRAM consumption.
* **Sequential Dependency:** The `backward_compute_gradients()` method follows a strict linear sequence. Once a gradient for layer i-1 is computed using the gradient from layer i, the buffer for layer i is never accessed again during that step.
* **Existing Checkpointing:** The code already implements `stage_boundaries` checkpointing for activations, but it hasn't applied similar logic to gradient memory management yet.

## Design Options

* **Option A: Static Double Pooling (Ping-Pong):** Allocate exactly two large buffers (`Pool_A` and `Pool_B`) equal to the size of the largest layer. Redirect all existing gradient pointers to alternate between these two pools. This is the most efficient for sequential networks.
* **Option B: Unified Gradient Scratchpad:** Use a single massive scratchpad and use offsets to assign pointers. This is more flexible if layers have wildly different sizes but requires more complex offset math.

## Open Questions

* **Kernel Compatibility:** Are there any kernels (specifically the custom `conv2d_backward_filter_kernel`) that require an input activation buffer and its corresponding gradient buffer to remain valid simultaneously in a way that conflicts with a 2-pool swap?
* **Max Layer Calculation:** Should we hardcode the pool size based on the 256 \times 32 \times 32 layer, or calculate it dynamically in `alloc_all()` to support different input resolutions?

Which approach (Option A or B) aligns best with your vision?

I will prepare the file updates once you confirm the approach. Note: I will focus on updating `gpu_autoencoder.hpp` and `gpu_autoencoder.cu` as they contain the primary v1.4 logic.