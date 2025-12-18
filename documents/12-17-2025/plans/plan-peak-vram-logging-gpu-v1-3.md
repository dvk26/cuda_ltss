Based on my research, here's what I found:

## Current State
- `src/gpu_v1.3/main.cu` already logs epoch wall-clock time using `std::chrono` and prints `-> Time: ...s (...) batches/s` per epoch.
- Peak VRAM logging is implemented only in `src/gpu_v1.4/main.cu` by sampling `cudaMemGetInfo()` during the per-batch loop and tracking the maximum of `(total_bytes - free_bytes)` per epoch.
- `gpu_v1.3` has the same host-side per-batch training loop structure as `gpu_v1.4`, so the v1.4 “sample-per-batch” pattern can be ported with minimal surface area (main program only).

## Design Options
- Option A (match v1.4, most accurate): Sample `cudaMemGetInfo()` once per batch and track `peak_used_bytes` per epoch.
  - Implementation steps:
    - In `src/gpu_v1.3/main.cu`, add per-epoch variables: `size_t peak_used_bytes = 0;`.
    - Inside the `while (train_loader.has_next())` loop (after `gpu_ae.train_step(...)` so GPU work is completed/synchronized by the model), call `cudaMemGetInfo(&free_bytes, &total_bytes)` and update `peak_used_bytes = max(peak_used_bytes, total_bytes - free_bytes)`.
    - After the epoch finishes, print `peak_used_bytes` in MiB alongside the existing epoch timing line (e.g., `Peak VRAM: XXX MiB`).
    - Optionally repeat the same measurement in the final test loop (or keep it training-only) depending on what you want benchmarked.
    - Ensure compilation includes `cudaMemGetInfo` declarations (e.g., include `<cuda_runtime.h>` in `src/gpu_v1.3/main.cu` if not already transitively included).
- Option B (lower overhead, less “peak-accurate”): Sample `cudaMemGetInfo()` once per epoch (end of epoch) and report “VRAM used at epoch end”.
  - Implementation steps:
    - Add a single `cudaMemGetInfo()` call right before printing epoch stats; compute `used_bytes = total - free`.
    - Print `used_bytes` (not “peak”) to avoid misleading semantics.
- Option C (compromise): Sample every N batches (e.g., every 10/25/50) to reduce overhead while approximating peak.
  - Implementation steps:
    - Add `--vram-sample-every <N>` (default `1`) to `src/gpu_v1.3/main.cu` argv parsing.
    - Only call `cudaMemGetInfo()` when `(train_nb % N == 0)`; still track `peak_used_bytes`.

## Open Questions
- Should “peak VRAM” mean device-wide used memory inferred from `cudaMemGetInfo()` (current v1.4 behavior), or process-specific usage (would require NVML / external dependency)?
- Do you want peak VRAM reported per-epoch (resets each epoch like v1.4) or a single peak across the whole training run?
- Is per-batch sampling acceptable overhead for your benchmarks, or should we sample every N batches / once per epoch?

Which approach aligns best with your vision?

