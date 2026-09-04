#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-dreamdojo:cu128-v2}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"

if ! docker info >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Cannot access the Docker daemon.

The current user must be added to the docker group by an administrator:
  sudo usermod -aG docker "$USER"

Log out and back in afterward, then rerun this script.
Do not make /var/run/docker.sock world-writable.
EOF
    exit 1
fi

echo "Building ${IMAGE} from ${ROOT}/Dockerfile"
DOCKER_BUILDKIT=1 docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg CUDA_NAME=cu128 \
    --build-arg STANDALONE=true \
    --tag "${IMAGE}" \
    "${ROOT}"

echo "Checking the DreamDojo CUDA environment..."
docker run --rm \
    --gpus all \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc '
        source /workspace/.venv/bin/activate
        python -c "
import os
import torch
import flash_attn
import h5py
import lightning
import pytorch3d
import torchcodec
import cosmos_predict2
import cosmos_predict2._src.predict2.action.configs.action_conditioned.config
print(\"torch:\", torch.__version__)
print(\"CUDA build:\", torch.version.cuda)
print(\"Python:\", os.path.realpath(\"/workspace/.venv/bin/python\"))
print(\"flash-attn: available\")
print(\"DreamDojo runtime extras: available\")
print(\"DreamDojo/Cosmos Predict2: available\")
"
        python -c "
import os
path = os.path.realpath(\"/workspace/.venv/bin/python\")
assert path.startswith(\"/opt/uv/python/\"), path
"
    '

echo "Built and verified Docker image: ${IMAGE}"
