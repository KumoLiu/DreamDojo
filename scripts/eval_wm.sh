#!/usr/bin/env bash
# Evaluate the HF post-train checkpoint on the three held-out validation sets.
#
# Reports teacher-forced and closed-loop MSE / PSNR / MAE against ground truth
# video, per dataset and overall.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-}" == --help ]]; then
  echo "Usage: CHECKPOINT=/path/model_ema_bf16.pt bash scripts/eval_wm.sh"
  echo "Optional: RUN_ROOT, ITERATION, EXPERIMENT, OUT_ROOT, HORIZON=auto, NUM_INFERENCE_STEPS=35"
  echo "GPU selection: EVAL_GPUS (indices/UUIDs), or CUDA_VISIBLE_DEVICES; EXCLUDE_GPUS filters discovered UUIDs."
  exit 0
fi
cd "$ROOT"
source "$ROOT/scripts/lib/env.sh"

# The final rank32 recipe is the default; cluster jobs pass an explicit run.
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora}"

ITERATION="${ITERATION:-18000}"
ITER_PADDED="$(printf "%09d" "$ITERATION")"
if [[ -n "${RUN_ROOT:-}" ]]; then
  CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/checkpoints/iter_${ITER_PADDED}/model_ema_bf16.pt}"
else
  CHECKPOINT="${CHECKPOINT:-$ROOT/../models/dreamdojo/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt}"
fi

# num_frames = 1 + 12 * num_chunks, and the loader subsamples video by 2, so an
# episode needs 2 * num_frames of its 30fps frames to fill the horizon without
# padding. Anything shorter gets its tail padded with a frozen copy of the last
# frame, which is trivially easy to predict and inflates PSNR.
#
# HORIZON=auto gives each episode the longest horizon its own length supports,
# so every episode is scored over its full duration with zero padding. This is
# the only way to see a complete rollout: val episodes are 54-248 video frames,
# so no single fixed horizon covers them all.
# HORIZON=<n> instead forces n chunks for every episode and drops episodes too
# short to fill it, which is the stricter setting for comparing runs.
HORIZON="${HORIZON:-auto}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-35}"

# Prefer explicit/scheduler-assigned devices. Machine-specific faulty GPUs
# belong in the caller's EVAL_GPUS/EXCLUDE_GPUS, not shared cluster defaults.
EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"
if [[ -n "${EVAL_GPUS:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<<"$EVAL_GPUS"
elif [[ -v CUDA_VISIBLE_DEVICES ]]; then
  IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPU_IDS < <(
    nvidia-smi -L 2>/dev/null | sed -n 's/^GPU [0-9]*: .*(UUID: \(GPU-[0-9a-f-]*\))$/\1/p' |
      while read -r uuid; do
        skip=""
        for bad in $(tr ',' ' ' <<<"$EXCLUDE_GPUS"); do
          [[ -n "$bad" && "$uuid" == *"$bad"* ]] && skip=1
        done
        [[ -z "$skip" ]] && echo "$uuid"
      done
  )
fi
NUM_GPUS="${#GPU_IDS[@]}"
if ((NUM_GPUS < 1)); then
  echo "No usable GPUs. nvidia-smi -L:" >&2
  nvidia-smi -L 2>&1 | sed 's/^/  /' >&2
  echo "  EXCLUDE_GPUS=$EXCLUDE_GPUS" >&2
  exit 1
fi

# A broken CUDA context makes every shard fail one episode at a time, which
# looks like 8 unrelated crashes. Fail fast instead, and check on a GPU from the
# pool so a context failure on the excluded card does not abort the run.
if ! CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
    python3 -c "import torch; torch.zeros(8, device='cuda')" 2>/dev/null; then
  echo "ERROR: cannot create a CUDA context on this node." >&2
  echo "  nvidia-smi sees $(nvidia-smi -L 2>/dev/null | wc -l) GPU(s)," \
       "$NUM_GPUS usable, but torch cannot initialise CUDA." >&2
  echo "  Check the driver and explicit EVAL_GPUS selection before retrying." >&2
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/eval/wm_val_iter${ITERATION}_h${HORIZON}_${RUN_TS}}"
LOG_DIR="$OUT_ROOT/logs"

VAL_SETS=(
  "teleop_success:datasets/g1_hf_pick_trocar_teleop_success_val"
  "rollouts_30k:datasets/g1_hf_pick_trocar_rollouts_30k_val"
  "rollouts_10k:datasets/g1_hf_pick_trocar_rollouts_10k_val"
)

test -f "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$LOG_DIR"

{
  echo "============================================================"
  echo "HF Post-Train Validation Eval"
  echo "Started: $(date -Is)"
  echo "============================================================"
  echo "  checkpoint          = $CHECKPOINT"
  echo "  experiment          = $EXPERIMENT"
  echo "  horizon             = $HORIZON"
  echo "  num_inference_steps = $NUM_INFERENCE_STEPS"
  echo "  gpus                = $NUM_GPUS (${GPU_IDS[*]})"
  echo "  excluded gpus       = ${EXCLUDE_GPUS:-<none>}"
  echo "  output              = $OUT_ROOT"
  echo
} | tee "$OUT_ROOT/00_setup.log"

# Build the (dataset, episode, num_chunks) work list, then deal it round-robin
# to GPUs. With single_base_index the loader maps sample index 1:1 onto episode
# order, so these indices pass straight through to --episode-indices.
mapfile -t WORK < <(
  for entry in "${VAL_SETS[@]}"; do
    name="${entry%%:*}"
    path="${entry##*:}"
    python3 - "$name" "$path" "$HORIZON" "$OUT_ROOT/00_setup.log" \
      "$OUT_ROOT/horizons.tsv" <<'PY'
import json
import sys

name, path, horizon, setup_log, manifest = sys.argv[1:6]
lengths = [
    json.loads(line)["length"]
    for line in open(f"{path}/meta/episodes.jsonl")
    if line.strip()
]

# num_frames = 1 + 12 * nc must satisfy 2 * num_frames <= length.
def max_chunks(length: int) -> int:
    return (length - 2) // 24

selected = []
if horizon == "auto":
    for i, length in enumerate(lengths):
        nc = max_chunks(length)
        if nc >= 1:
            selected.append((i, nc, length))
    note = "auto (per-episode full length, zero padding)"
else:
    nc = int(horizon)
    selected = [
        (i, nc, length) for i, length in enumerate(lengths) if max_chunks(length) >= nc
    ]
    note = f"fixed {nc} chunks ({1 + 12 * nc} frames)"

with open(setup_log, "a") as f:
    f.write(f"  {name}: {len(selected)}/{len(lengths)} episodes, horizon {note}\n")
    if selected:
        frames = [1 + 12 * nc for _, nc, _ in selected]
        f.write(
            f"    video frames: min={min(frames)} max={max(frames)}"
            f" mean={sum(frames) / len(frames):.0f}\n"
        )

with open(manifest, "a") as f:
    for i, nc, length in selected:
        f.write(f"{name}\t{i}\t{nc}\t{1 + 12 * nc}\t{length}\n")

for i, nc, _ in selected:
    print(f"{name}|{path}|{i}|{nc}")
PY
  done
)

echo "Total episodes to evaluate: ${#WORK[@]}" | tee -a "$OUT_ROOT/00_setup.log"

run_shard() {
  local gpu="$1"
  shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  for item in "$@"; do
    IFS='|' read -r name path episode chunks <<<"$item"
    episode_dir="$OUT_ROOT/$name/episode_$(printf '%03d' "$episode")"
    # Point OUT_ROOT at a previous partial run to resume it: episodes that
    # already produced metrics are not recomputed.
    if [[ -f "$episode_dir/metrics.json" ]]; then
      echo "skip (already done): $name episode $episode"
      continue
    fi
    python scripts/validate_pick_trocar_world_model.py \
      --checkpoint "$CHECKPOINT" \
      --experiment "$EXPERIMENT" \
      --dataset-path "$path" \
      --episode-indices "$episode" \
      --num-chunks "$chunks" \
      --num-inference-steps "$NUM_INFERENCE_STEPS" \
      --save-fps 15 \
      --output-dir "$episode_dir"
  done
}

pids=()
for ((g = 0; g < NUM_GPUS; g++)); do
  shard=()
  for ((i = g; i < ${#WORK[@]}; i += NUM_GPUS)); do
    shard+=("${WORK[i]}")
  done
  if ((${#shard[@]} == 0)); then
    continue
  fi
  run_shard "${GPU_IDS[g]}" "${shard[@]}" > "$LOG_DIR/gpu${g}.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

python3 - "$OUT_ROOT" <<'PY' | tee "$OUT_ROOT/metrics_summary.txt"
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = ["teleop_success", "rollouts_30k", "rollouts_10k"]
modes = ["teacher_forced", "closed_loop"]
metrics = ["mse", "psnr", "mae"]

# Horizon varies per episode under HORIZON=auto, so report it alongside the
# metrics: a PSNR is only meaningful next to the number of frames it averages.
frames_by_episode = {}
manifest = root / "horizons.tsv"
if manifest.exists():
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        name, episode, _chunks, frames, _length = line.split("\t")
        frames_by_episode[(name, int(episode))] = int(frames)

summary, all_records, all_frames = {}, [], []
for name in names:
    records, frames = [], []
    for path in sorted((root / name).glob("episode_*/metrics.json")):
        for record in json.loads(path.read_text()):
            records.append(record)
            nf = frames_by_episode.get((name, int(record["episode_index"])))
            if nf:
                frames.append(nf)
    if not records:
        continue
    all_records.extend(records)
    all_frames.extend(frames)
    summary[name] = {
        "episodes": len(records),
        "mean_frames": round(st.mean(frames), 1) if frames else None,
        **{
            mode: {m: st.mean(r[mode][m] for r in records) for m in metrics}
            for mode in modes
        },
    }

if all_records:
    summary["overall"] = {
        "episodes": len(all_records),
        "mean_frames": round(st.mean(all_frames), 1) if all_frames else None,
        **{
            mode: {m: st.mean(r[mode][m] for r in all_records) for m in metrics}
            for mode in modes
        },
    }

(root / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

hdr = f"{'dataset':<16}{'eps':>5}{'frames':>8}  " + "  ".join(
    f"{mode[:2].upper()}-{m.upper():<5}" for mode in modes for m in metrics
)
print(hdr)
print("-" * len(hdr))
for name, s in summary.items():
    mean_frames = "-" if s["mean_frames"] is None else f"{s['mean_frames']:.0f}"
    row = f"{name:<16}{s['episodes']:>5}{mean_frames:>8}  "
    row += "  ".join(
        f"{s[mode][m]:<8.3f}" for mode in modes for m in metrics
    )
    print(row)
print("\nTE = teacher forced, CL = closed loop")
print("frames = mean evaluated video frames per episode (15 fps)")
PY

python3 scripts/combine_eval_videos.py "$OUT_ROOT" 2>&1 \
  | tee "$LOG_DIR/combine.log" || echo "Overview video generation failed" >&2

{
  echo
  echo "Finished: $(date -Is)"
  echo "Shard failures: $failed"
  echo "Summary: $OUT_ROOT/metrics_summary.json"
  echo "Overview videos: $OUT_ROOT/overview/"
} | tee -a "$OUT_ROOT/00_setup.log"

exit "$failed"
