#!/usr/bin/env bash
# Build, export and transfer the DreamDojo environment. No action is implicit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-dreamdojo:cu128-v2}"
OUTPUT="${OUTPUT:-${ROOT}/docker/dreamdojo-cu128-v2.sqsh}"
REMOTE_HOST="${REMOTE_HOST:-nvidia-cluster}"
REMOTE_DIR="${REMOTE_DIR:-/lustre/fsw/portfolios/healthcareeng/users/yunl/docker}"

build() {
    if ! docker info >/dev/null 2>&1; then
        echo 'Docker daemon unavailable. Ask an administrator for docker-group access; do not open the socket permissions.' >&2
        exit 1
    fi
    DOCKER_BUILDKIT=1 docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}" \
        --build-arg CUDA_NAME=cu128 --build-arg STANDALONE=true \
        --tag "$DOCKER_IMAGE" "$ROOT"
    docker run --rm --gpus all --entrypoint /workspace/.venv/bin/python \
        "$DOCKER_IMAGE" -c '
import os
import torch
import flash_attn
import h5py
import lightning
import pytorch3d
import torchcodec
import cosmos_predict2._src.predict2.action.configs.action_conditioned.config
path = os.path.realpath("/workspace/.venv/bin/python")
assert path.startswith("/opt/uv/python/"), path
print("DreamDojo runtime verified:", torch.__version__, torch.version.cuda, path)
'
}

export_sqsh() {
    local enroot_bin enroot_lib staging
    if command -v enroot >/dev/null 2>&1; then
        enroot_bin="$(command -v enroot)"
        enroot_lib="${ENROOT_LIBRARY_PATH:-}"
    elif [[ -x "$HOME/.local/enroot-4.2.0/usr/bin/enroot" ]]; then
        enroot_bin="$HOME/.local/enroot-4.2.0/usr/bin/enroot"
        enroot_lib="$HOME/.local/enroot-4.2.0/usr/lib/enroot"
    else
        echo 'Enroot is not installed.' >&2
        exit 1
    fi
    docker image inspect "$DOCKER_IMAGE" >/dev/null
    if [[ -L "$OUTPUT" || ( -e "$OUTPUT" && ! -f "$OUTPUT" ) ]]; then
        echo "Output must be a regular file, not a directory/symlink: $OUTPUT" >&2
        exit 1
    fi
    if [[ -e "$OUTPUT" && "${FORCE:-0}" != 1 ]]; then
        echo "Output exists: $OUTPUT. Choose another OUTPUT, or explicitly set FORCE=1." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$OUTPUT")"
    staging="$(mktemp -d "$(dirname "$OUTPUT")/.dreamdojo-image.XXXXXXXX")"
    # Validate before replacing anything. A failed import keeps its staging
    # directory for inspection and leaves the previous image untouched.
    echo "Export staging: $staging"
    ENROOT_LIBRARY_PATH="$enroot_lib" "$enroot_bin" import \
        --output "$staging/image.sqsh" "dockerd://${DOCKER_IMAGE}"
    unsquashfs -s "$staging/image.sqsh"
    mv -f "$staging/image.sqsh" "$OUTPUT"
    rmdir "$staging"
    sha256sum "$OUTPUT"
}

upload() {
    [[ -f "$OUTPUT" ]] || { echo "Missing image: $OUTPUT" >&2; exit 1; }
    local remote_path directory_q path_q local_sha remote_sha
    remote_path="${REMOTE_DIR}/$(basename "$OUTPUT")"
    printf -v directory_q '%q' "$REMOTE_DIR"
    printf -v path_q '%q' "$remote_path"
    local_sha="$(sha256sum "$OUTPUT" | awk '{print $1}')"
    ssh "$REMOTE_HOST" "mkdir -p $directory_q"
    rsync -avh --protect-args --partial --append-verify --info=progress2 \
        "$OUTPUT" "${REMOTE_HOST}:${remote_path}"
    remote_sha="$(ssh "$REMOTE_HOST" "sha256sum $path_q" | awk '{print $1}')"
    [[ "$local_sha" == "$remote_sha" ]] || { echo 'Image checksum mismatch.' >&2; exit 1; }
    echo "Verified: ${REMOTE_HOST}:${remote_path}"
}

case "${1:---help}" in
    build) build ;;
    export) export_sqsh ;;
    upload) upload ;;
    --help|-h)
        echo 'Usage: bash scripts/cluster/image.sh {build|export|upload}'
        echo 'Overrides: DOCKER_IMAGE, BASE_IMAGE, OUTPUT, REMOTE_HOST, REMOTE_DIR'
        echo 'Run each step explicitly. Export refuses existing OUTPUT unless FORCE=1.' ;;
    *) echo "Unknown action: $1" >&2; exit 2 ;;
esac
