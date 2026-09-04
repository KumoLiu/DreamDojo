#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-yunl@nb-hel-cs-001-dc-01}"
REMOTE_DIR="${REMOTE_DIR:-/lustre/fsw/portfolios/healthcareeng/users/yunl/docker}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-dreamdojo-cu128-v2.sqsh}"
IMAGE="${ROOT}/docker/${IMAGE_NAME}"

if [[ ! -f "${IMAGE}" ]]; then
    echo "Missing image: ${IMAGE}" >&2
    exit 1
fi

LOCAL_SHA256="$(sha256sum "${IMAGE}" | awk '{print $1}')"

ssh "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"
rsync -avh --partial --append-verify --info=progress2 \
    "${IMAGE}" \
    "${REMOTE}:${REMOTE_DIR}/${IMAGE_NAME}"

REMOTE_SHA256="$(
    ssh "${REMOTE}" \
        "sha256sum '${REMOTE_DIR}/${IMAGE_NAME}'" |
        awk '{print $1}'
)"

if [[ "${LOCAL_SHA256}" != "${REMOTE_SHA256}" ]]; then
    echo "Checksum mismatch after transfer." >&2
    exit 1
fi

echo "Transfer verified:"
echo "${REMOTE}:${REMOTE_DIR}/${IMAGE_NAME}"
