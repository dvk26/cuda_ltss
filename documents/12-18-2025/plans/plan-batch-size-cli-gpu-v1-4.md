Based on my research, here's what I found:

## Current State
- `src/gpu_v1.4/main.cu` hard-codes `batch_size = 64`, then uses it to construct both `GPUAutoencoder(batch_size, 32, 32)` and the `DataLoader`s.
- `GPUAutoencoder` treats batch size as a construction-time constant (`N_`), and allocates/copies using `N_` (e.g., fixed-size `set_input()` H2D copy), so it must be known before model construction.
- `DataLoader` emits a smaller final batch (`min(batch_size, remaining)`), and the v1.4 training + test loops currently **skip** any batch where `batch.N() != batch_size`.
- v1.4 already has minimal argv parsing for `--keep-partial` and rejects unknown args; other GPU variants use the same scan-argv loop style.

## Design Options
- Option A (minimal / safest): Add `--batch-size <int>` and keep “skip incomplete final batch” behavior.
  - Implementation steps:
    - Update `src/gpu_v1.4/main.cu` usage text to include `--batch-size <int>` (and document defaults).
    - Extend the existing argv scan loop to parse `--batch-size` (support both `--batch-size 128` and optionally `--batch-size=128`).
    - Validate inputs early (positive, reasonable upper bound optional) and fail fast with a clear error message.
    - Use parsed `batch_size` to construct `GPUAutoencoder` and both `DataLoader`s.
    - Keep current `if (batch.N() != batch_size) continue;` behavior (training + test) so GPU buffer shapes remain fixed and safe.
    - Update any related notebook command lines (e.g., `ltss_gpu_v1_4.ipynb`) to demonstrate the new flag.
- Option B (more flexible): Add `--batch-size <int>` and also support running the final partial batch.
  - Two sub-approaches:
    - B1 (pad on host): Pad the final batch to `batch_size` before calling `train_step()` / `test_step()`, and optionally mask loss/metrics.
    - B2 (dynamic batch): Teach `GPUAutoencoder` to accept variable `N` per step (reallocate/resize buffers and adjust kernel launches), which is higher risk and larger surface area.
  - This option changes semantics and likely touches `src/gpu_v1.4/gpu_autoencoder.cu` (loss total size, kernel grids using `N`, and buffer allocation/copies).

## Open Questions
- Should v1.4 continue skipping incomplete final batches (current behavior), or do you want the last partial batch supported?
- Should `--batch-size` affect both training and test evaluation (current shared usage), or should we allow separate flags (e.g., `--train-batch-size` / `--test-batch-size`)?
- Do you want strict argument handling (unknown args error) preserved, or should we allow pass-through / ignore-unknown for notebook convenience?

Which approach aligns best with your vision?

