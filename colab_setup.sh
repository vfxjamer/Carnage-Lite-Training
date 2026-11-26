#!/usr/bin/env bash
# Optional Colab build helper. The notebook handles Drive restore/sync and session supervision.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/Carnage-Lite}"
BUILD_DIR="$REPO_ROOT/build"

if [[ ! -f "$REPO_ROOT/CMakeLists.txt" ]]; then
  echo "Project not found at $REPO_ROOT. Run the notebook clone cell first."
  exit 1
fi

apt-get update -qq
apt-get install -y -qq build-essential cmake git libpython3-dev pkg-config

python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Select a Colab GPU runtime before building the training binary.")
print("GPU:", torch.cuda.get_device_name(0))
PY

nvidia-smi
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

echo "Build complete: $BUILD_DIR/Carnage"
echo "Use Carnage_colab.ipynb to restore checkpoints and run a supervised session."
