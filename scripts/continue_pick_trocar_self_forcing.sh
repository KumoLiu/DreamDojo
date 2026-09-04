#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env_local.sh"

TEACHER_OUTPUT="datasets/g1_pick_trocar_iter2500_teacher_gen_10k"
WARMUP_OUTPUT="outputs/train/dreamdojo/pick_trocar_self_forcing/g1_pick_trocar_iter2500_warmup"
EMBEDDING_PATH="datasets/cr1_empty_string_text_embeddings.pt"

teacher_generation_running() {
    pgrep -f '/python -u -m cosmos_predict2\._src\.predict2\.action\.inference\.inference_gr00t_warmup.*g1_pick_trocar_iter2500_teacher_gen_10k' >/dev/null
}

while [[ ! -f "${TEACHER_OUTPUT}/actions/9999.json" ]]; do
    if ! teacher_generation_running; then
        echo "Teacher generation exited before producing sample 9999." >&2
        exit 1
    fi
    sleep 60
done

while teacher_generation_running; do
    sleep 10
done

python - <<'PY'
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

root = Path("datasets/g1_pick_trocar_iter2500_teacher_gen_10k")
expected = set(range(10_000))
for subdir, suffix in (
    ("actions", ".json"),
    ("images", ".png"),
    ("latents", ".pt"),
    ("videos", ".mp4"),
):
    found = {int(path.stem) for path in (root / subdir).glob(f"*{suffix}")}
    missing = expected - found
    if missing:
        raise RuntimeError(f"{subdir} is missing {len(missing)} samples, first: {min(missing)}")

embedding = Path("datasets/cr1_empty_string_text_embeddings.pt")
if not embedding.exists():
    downloaded = hf_hub_download(
        "nvidia/Cosmos-Predict2.5-2B",
        "robot/action-cond/cr1_empty_string_text_embeddings.pt",
    )
    shutil.copy2(downloaded, embedding)
print("Teacher targets and CR1 embedding are ready.")
PY

export IMAGINAIRE_OUTPUT_ROOT="$ROOT/outputs/train"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8

torchrun --nnodes=1 --nproc_per_node=4 \
    --master_port=12349 --master_addr=localhost --node_rank=0 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py -- \
    experiment=pick_trocar_iter2500_warmup_no_s3 \
    job.wandb_mode=disabled \
    2>&1 | tee results/g1_pick_trocar_iter2500_warmup.log

test -f "${WARMUP_OUTPUT}/checkpoints/iter_000020000/model/.metadata"

torchrun --nnodes=1 --nproc_per_node=4 \
    --master_port=12350 --master_addr=localhost --node_rank=0 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py -- \
    experiment=pick_trocar_iter2500_self_forcing_no_s3 \
    job.wandb_mode=disabled \
    2>&1 | tee results/g1_pick_trocar_iter2500_self_forcing.log
