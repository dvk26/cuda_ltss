# Repository Guidelines

## Project Structure & Module Organization

- `src/include/`: shared C++ headers (tensor, layers, dataset, autoencoder).
- `src/cpu/`: CPU training entrypoint (`main.cpp`).
- `src/gpu/`, `src/gpu_v1.1/`, `src/gpu_v1.2/`, `src/gpu_v1.3/`: CUDA implementations and entrypoints (`main.cu`) for different iterations.
- `src/extract_feature/`: extracts latent features to `train_features.bin` / `test_features.bin`.
- `src/train_svm/`: C++ SVM utilities; `src/train_svm_cuml/`: Python + cuML SVM script.
- Root `*.ipynb`: experiment notebooks (training, feature extraction, reporting).

## Build, Test, and Development Commands

This repo does not use CMake/Make; compile directly.

```bash
# CPU autoencoder
g++ -std=c++17 -O3 -march=native -ffast-math -DNDEBUG src/cpu/*.cpp -o autoencoder_cpu
./autoencoder_cpu cifar-10-batches-bin [--keep-partial]

# GPU autoencoder (adjust -arch to your GPU)
nvcc -std=c++17 -O2 src/gpu/main.cu src/gpu/gpu_autoencoder.cu -I./src/gpu -arch=sm_75 -o autoencoder_gpu
./autoencoder_gpu cifar-10-batches-bin [--keep-partial]
```

Feature extraction + cuML SVM workflow:
- Build/run `src/extract_feature/extract_features_to_file.cpp` (needs CUDA) to produce `*_features.bin`.
- Run `python src/train_svm_cuml/train_svm.py` (requires `cuml`, `scikit-learn`, `numpy`).

## Coding Style & Naming Conventions

- C++/CUDA: 4-space indentation, braces on the same line, `#pragma once` in headers.
- Naming: classes in `PascalCase` (e.g., `Autoencoder`), functions/vars in `snake_case` (e.g., `save_ppm`).
- Keep changes localized to the relevant variant directory (`src/gpu_v1.3/` etc.).
- Comment code in explanatory, pedagogical way with clear shape notations for mathematical objects.

## Testing Guidelines

- No automated test suite is currently included.
- For changes, prefer a smoke check: compile, run with `--keep-partial`, and confirm outputs are generated in `out-cpu/` or `out-gpu/`.

## Commit & Pull Request Guidelines

- Commit history uses short, imperative messages (often starting with “update …”); follow that convention and include the area touched (e.g., `update gpu_v1.3 training loop`).
- PRs should include: what changed, how to reproduce (exact command), GPU model/`-arch` used (if relevant), and screenshots for notebook-only changes.
- Do not commit generated outputs (`out-*/`, datasets, large intermediate binaries); keep them local.