# Pick-Trocar backup inventory

This file records the minimal artifacts retained before releasing the local
machine. Paths are relative to the repository root.

## Required code

- Teacher post-training and loss ablation:
  - `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py`
  - `configs/2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_ema_both.yaml`
  - `configs/2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_ema_both_no_motion_consistency.yaml`
- Stratified Teacher Generation and 97-frame distillation:
  - `scripts/build_teacher_gen_sampling_manifest.py`
  - `scripts/validate_teacher_gen_dataset.py`
  - `scripts/run_pick_trocar_stratified_97f_pipeline.sh`
  - `cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_pick_trocar_self_forcing.py`
  - `cosmos_predict2/_src/predict2/interactive/configs/data.py`
- Teacher and causal held-out evaluation:
  - `scripts/validate_pick_trocar_world_model.py`
  - `scripts/eval_distilled_holdout.py`
  - `scripts/combine_holdout_eval_5x6.py`
  - `scripts/combine_distilled_holdout_eval.py`
  - `scripts/run_local_holdout_full_eval.sh`

## Retained resumable checkpoints

- Teacher:
  `outputs/train/dreamdojo/pick_trocar_headcam_rollout_holdout_5s5f_ema_both/g1_pick_trocar_headcam_2b_rollout_holdout_5s5f_ema_both/checkpoints/iter_000003000`
- Selected stratified Warmup:
  `outputs/train/dreamdojo/pick_trocar_self_forcing_97f/g1_pick_trocar_iter2500_warmup_stratified/checkpoints/iter_000003000`
- Final 97-frame Self-Forcing:
  `outputs/train/dreamdojo/pick_trocar_self_forcing_97f/g1_pick_trocar_iter2500_self_forcing_97f/checkpoints/iter_000003000`
- Original initialization:
  `migrated_checkpoints/g1_pick_trocar_headcam_2b_iter_000010000_ema_both`

The historical iter-2500 teacher checkpoint was already absent at cleanup time.
Configs therefore use the retained iter-3000 teacher for future continuation.

## Retained generated metadata

`datasets/g1_pick_trocar_iter2500_teacher_gen_10k_stratified/sample_manifest.jsonl`
is retained. The large `latents`, `actions`, `images`, and `videos` subfolders
were deleted because they can be regenerated from the manifest and teacher.

The manifest contains 10,000 unique windows: 40 from every one of the 250
training episodes, split evenly between uniform temporal coverage and
left-arm/left-hand action-weighted sampling.

## Evaluation outputs

Primary retained reports:

- `results/pick_trocar_distilled_holdout_97f`
- `results/holdout_5s5f_full_eval_rerun_20260825_1406_iter3000`

Evaluation CLI indices are positions in the 10-episode holdout view, not source
episode IDs:

`0→0, 1→19, 2→21, 3→27, 4→53, 5→55, 6→127, 7→154, 8→198, 9→236`.

## Container

`docker/dreamdojo-cu128-v2.sqsh` is the latest validated container image.
`Dockerfile`, `.dockerignore`, and the build/transfer scripts are retained so
the image can also be rebuilt.
