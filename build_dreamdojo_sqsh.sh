#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-dreamdojo:cu128-v2}"
OUTPUT="${OUTPUT:-${ROOT}/docker/dreamdojo-cu128-v2.sqsh}"
LOCAL_ENROOT_ROOT="${HOME}/.local/enroot-4.2.0"

if command -v enroot >/dev/null 2>&1; then
    ENROOT_BIN="$(command -v enroot)"
    ENROOT_LIBRARY_PATH_VALUE="${ENROOT_LIBRARY_PATH:-}"
elif [[ -x "${LOCAL_ENROOT_ROOT}/usr/bin/enroot" ]]; then
    ENROOT_BIN="${LOCAL_ENROOT_ROOT}/usr/bin/enroot"
    ENROOT_LIBRARY_PATH_VALUE="${LOCAL_ENROOT_ROOT}/usr/lib/enroot"
else
    echo "Enroot is not installed." >&2
    exit 1
fi

if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
    echo "Missing Docker image: ${DOCKER_IMAGE}" >&2
    exit 1
fi

if [[ -e "${OUTPUT}" ]]; then
    if [[ "${FORCE:-0}" != "1" ]]; then
        echo "Output already exists: ${OUTPUT}" >&2
        echo "Set FORCE=1 to replace it." >&2
        exit 1
    fi
    rm -f "${OUTPUT}"
fi

mkdir -p "$(dirname "${OUTPUT}")"
echo "Converting ${DOCKER_IMAGE} to ${OUTPUT}"
ENROOT_LIBRARY_PATH="${ENROOT_LIBRARY_PATH_VALUE}" \
    "${ENROOT_BIN}" import \
    --output "${OUTPUT}" \
    "dockerd://${DOCKER_IMAGE}"

unsquashfs -s "${OUTPUT}"
sha256sum "${OUTPUT}"
