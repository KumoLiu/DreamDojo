# DreamDojo 实验上下文与备份说明

本文档记录本机截至 2026-08-25 的 DreamDojo / G1 pick-trocar 实验，
用于在新机器上恢复实验上下文。路径均相对于 `~/DreamDojo`，除非特别说明。

## 1. 实验目标

目标是使用 Unitree G1 的 pick-trocar head-camera 数据，对 DreamDojo
2B 模型进行 post-training，并进一步将非因果 Teacher 蒸馏为可以长时域
自回归生成的 causal student。

主要研究问题：

1. G1 post-training 模型在 held-out case 上的 teacher-forced 与 closed-loop
   表现。
2. temporal / motion consistency loss 是否会加剧 hallucination。
3. Warmup 和 Self-Forcing 是否能改善 causal rollout 的长期漂移。
4. 49 帧和 97 帧 rollout 对长时域误差的影响。

## 2. 数据与 holdout 划分

原始数据集：

```text
datasets/g1_pick_trocar_rollout_all_260_headcam
```

共 260 个 episode。使用以下 manifest 固定划分：

```text
datasets/g1_pick_trocar_rollout_all_260_headcam_split_manifest.json
```

- 训练集：250 个 episode
- held-out 集：10 个 episode
- held-out success：21, 53, 127, 154, 198
- held-out failure：0, 19, 27, 55, 236

注意：评估脚本中的 `--episode-indices` 是 holdout view 中的**位置索引**，
不是原始 episode ID：

```text
holdout position 0,1,2,3,4,5,6,7,8,9
source episode    0,19,21,27,53,55,127,154,198,236
```

### `data_split: full` 与数据泄漏的关系

这里的 `full` 不是“使用原始 260 个 episode”，而是“使用当前 dataset
view 中的全部 episode”：

```text
g1_pick_trocar_rollout_all_260_headcam_train
  └─ full = 250 个 train episode

g1_pick_trocar_rollout_all_260_headcam_eval_5s5f
  └─ full = 10 个 held-out episode
```

拆分脚本是 `scripts/split_rollout_holdout.py`。它将以下 10 个固定评估
episode 从训练逻辑中排除：

```text
eval success: 21, 53, 127, 154, 198
eval failure: 0, 19, 27, 55, 236
```

两个 view 共享原始 `data/` 和 `videos/` 文件（通过 symlink），但各自具有
不同的 `meta/episodes.jsonl`、`episodes_stats.jsonl`、`info.json` 和统计
信息。因此数据文件物理上没有被复制或删除；隔离发生在 dataset metadata
和 episode index 层面。只要 dataloader 按 view 的 metadata 建立索引，
train view 就不会采到这 10 个 held-out episode。

需要区分以下实验：

| 实验 | 使用的数据 | 是否包含这 10 个 held-out case |
|---|---|---|
| `rollout_all_official` | 原始 260 episode 数据集 | 是，存在泄漏 |
| `rollout_all_ema_both` | 原始 260 episode 数据集 | 是，存在泄漏 |
| `holdout_5s5f_official` | 250 episode train view | 否 |
| `holdout_5s5f_ema_both` | 250 episode train view | 否 |
| `no_motion_consistency` | 250 episode train view | 否 |
| `rollout_mix` | 200 headcam train + 120 rollout 数据 | 有200 headcam val 作为 validation，但没有这组 10-case holdout eval |
| `real_mix` | 200 headcam + real train 数据 | 有 200 headcam val + real val 作为 validation，但没有这组 10-case holdout eval |

所以早期 `rollout_all_*` 结果只能作为探索性结果，不能和 holdout 版本的
10-case 指标进行严格的无泄漏对比。`rollout_mix` 和 `real_mix` 虽然各自
配置了 validation，但 validation 分别是普通 headcam / real 数据，并不是
这组固定的 10 个 held-out case。最终建议使用 `holdout_5s5f_*` 相关模型
和结果进行 10-case 对比。

目前没有删除这套拆分：

- 原始 260 episode 数据集仍保留；
- 250 episode train view 仍保留；
- 10 episode eval view 仍保留；
- split manifest 仍保留；
- 旧的全量实验 checkpoint 仍保留，但已标记为可能存在泄漏。

如果未来需要在物理文件层面也彻底隔离，需要将 train/eval 数据复制到两个
独立目录，而不是仅使用 symlink view。目前训练和评估流程依赖的是逻辑
episode 列表隔离。

保留的数据视图：

```text
datasets/g1_pick_trocar_rollout_all_260_headcam_train
datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f
```

## 3. Post-training 实验

### 3.1 原始 / 官方相关实验

配置文件名中的字段含义如下：

```text
2b_480_640_g1_pick_trocar_headcam_rollout_all_official
│  │   │  │            │          │       │
│  │   │  │            │          │       └─ 官方初始化 / 实验命名
│  │   │  │            │          └──────── rollout-all 数据
│  │   │  │            └────────────────── head camera
│  │   │  └─────────────────────────────── G1 embodiment
│  │   └────────────────────────────────── pick-trocar task
│  └───────────────────────────────────── 480×640 resolution
└──────────────────────────────────────── 2B model
```

这些 post-training 配置通常使用 12-action chunk、13-frame training window、
batch size 4、learning rate `4e-5`、3000 iterations，并且每 500 iterations
保存 checkpoint。`data_split: full` 表示在指定数据集 view 中使用完整数据；
`single_base_index: false` 表示从 episode 内的多个合法窗口采样。

相关配置：

```text
configs/2b_480_640_g1_pick_trocar_headcam.yaml
configs/2b_480_640_g1_pick_trocar_headcam_rollout_all_official.yaml
configs/2b_480_640_g1_pick_trocar_headcam_rollout_all_ema_both.yaml
configs/2b_480_640_g1_pick_trocar_headcam_rollout_mix.yaml
configs/2b_480_640_g1_pick_trocar_headcam_real_mix.yaml
```

主要输出目录：

```text
outputs/train/dreamdojo/pick_trocar_headcam/
outputs/train/dreamdojo/pick_trocar_headcam_rollout_mix/
```

这些实验用于探索 G1 head-camera、rollout-all、EMA 和 rollout mix。
它们属于历史实验，当前没有作为 97F Self-Forcing 主线的初始化入口。

#### `headcam_rollout_all_official`

```text
初始化：
checkpoints/DreamDojo/2B_G1_post-train/iter_000050000/

训练数据：
datasets/g1_pick_trocar_rollout_all_260_headcam

验证数据：
datasets/g1_pick_trocar_200_headcam_val
```

这是“官方 G1 2B post-train checkpoint + 全部 260 个 rollout episode”的
基线。`official` 表示使用官方已有 checkpoint 作为初始化，不表示数据来自
官方数据集。

#### `headcam_rollout_all_ema_both`

```text
初始化：
migrated_checkpoints/g1_pick_trocar_headcam_2b_iter_000010000_ema_both

训练数据：
datasets/g1_pick_trocar_rollout_all_260_headcam

验证数据：
datasets/g1_pick_trocar_200_headcam_val
```

它与 official 版本使用相同的 260 episode 数据，区别是从本机迁移的
iter 10000 `ema_both` 权重开始，而不是从官方 iter 50000 开始。

#### `headcam_rollout_mix`

```text
初始化：
migrated_checkpoints/g1_pick_trocar_headcam_2b_iter_000010000_ema_both

训练数据：
datasets/g1_pick_trocar_200_headcam_train
datasets/g1_pick_trocar_rollout_120_headcam

dataset_mixing_weights: [190, 120]

验证数据：
datasets/g1_pick_trocar_200_headcam_val
```

`rollout_mix` 表示混合普通 teleoperation / demonstration 数据和 rollout
数据，而不是只使用 rollout-all 数据。权重 `[190, 120]` 近似按照两个数据源
的 episode 数量采样，即约 190 份普通 teleop 数据对 120 份 rollout 数据，
不是两个目录各采样 50%。

#### `headcam_real_mix`

```text
初始化：
outputs/train/dreamdojo/pick_trocar_headcam/
  g1_pick_trocar_headcam_2b/checkpoints/iter_000010000

训练数据：
datasets/g1_pick_trocar_200_headcam_train
datasets/g1_pick_trocar_real_train_headcam
datasets/g1_pick_trocar_real_train_fail_contact_headcam

dataset_mixing_weights: [0.8, 0.1, 0.1]

验证数据：
datasets/g1_pick_trocar_200_headcam_val
datasets/g1_pick_trocar_real_val_headcam
```

`real_mix` 是把真实机器人数据，尤其是 `fail_contact` 样本，加入普通
head-camera 数据，用于测试真实接触和失败状态是否能减少抓取阶段漂移。

#### `g1_pick_trocar_real_train_fail_contact_headcam` 的来源

这个数据集不是手工复制出来的，而是由以下脚本生成：

```text
scripts/prepare_real_rollout_lerobot.py
```

默认输入和输出：

```text
输入真实 rollout：
/localhome/local-yunl/real_rollouts

输出根目录：
/localhome/local-yunl/DreamDojo/datasets
```

脚本会扫描输入下的 `*/episode_*`，读取每个 episode 的 `meta.json`、
`states.npy`、`actions.npy` 和 `color_0.mp4`，并按照固定的真实数据
validation episode ID 划分 train / val。当前固定的 validation IDs 是：

```text
1, 8, 9, 10, 13, 15, 27
```

随后从真实 train episode 中筛选 `success == false` 的 episode，生成：

```text
datasets/g1_pick_trocar_real_train_fail_contact_headcam
```

因此该目录的含义是：

- 来源：真实 Unitree G1 rollout；
- camera：head camera；
- split：真实数据 train split，不包含固定 real validation IDs；
- 内容：train split 中所有 `success == false` 的失败接触 episode；
- 当前统计：52 个 episode、9919 frames、30 FPS。

同一次脚本运行还会生成：

```text
datasets/g1_pick_trocar_real_train_headcam
datasets/g1_pick_trocar_real_val_headcam
datasets/g1_pick_trocar_200_headcam_train
datasets/g1_pick_trocar_200_headcam_val
datasets/g1_pick_trocar_split_manifest.json
```

脚本还会检查 real train 和 real validation 的 episode ID 不重叠。

相关输出目录：

```text
outputs/train/dreamdojo/pick_trocar_headcam/
outputs/train/dreamdojo/pick_trocar_headcam_real_mix/
outputs/train/dreamdojo/pick_trocar_headcam_rollout_mix/
```

### 3.2 Held-out EMA post-training

配置：

```text
configs/2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_ema_both.yaml
```

输出：

```text
outputs/train/dreamdojo/pick_trocar_headcam_rollout_holdout_5s5f_ema_both/
```

当前保留的 Teacher checkpoint：

```text
outputs/train/dreamdojo/pick_trocar_headcam_rollout_holdout_5s5f_ema_both/g1_pick_trocar_headcam_2b_rollout_holdout_5s5f_ema_both/checkpoints/iter_000003000
```

历史实验命名中多处仍使用 `iter2500`，但实际可用的 Teacher checkpoint
是 iter 3000。原先尝试使用的 iter 2500 checkpoint 已不存在，因此后续
配置和评估应以 iter 3000 为准。

训练数据和验证数据分别是：

```text
训练：
datasets/g1_pick_trocar_rollout_all_260_headcam_train

验证：
datasets/g1_pick_trocar_rollout_all_260_headcam_eval_5s5f
```

也就是说，这个实验与 rollout-all 全量版本的关键差别不是模型结构，而是
使用固定的 250 / 10 train-held-out 划分，避免在 10 个最终评估 case 上训练。

### 3.3 `headcam_rollout_holdout_5s5f_official`

配置：

```text
configs/2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_official.yaml
```

这是 held-out 划分下的 official 初始化对照：

- 初始化：`checkpoints/DreamDojo/2B_G1_post-train/iter_000050000/`
- 训练集：250 个 rollout episode
- 验证 / 最终 held-out：固定 10 个 episode
- 13-frame window、12-action chunk
- learning rate `4e-5`，训练 3000 iterations

其作用是与从 `ema_both` 初始化的 held-out 实验进行公平比较。

### 3.4 关闭 motion consistency loss 的对照实验

代码修改：

```text
cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py
```

新增可配置字段：

```text
model.config.motion_consistency_loss_weight
```

默认值为 `0.1`，设置为 `0.0` 时关闭该项 loss。

配置：

```text
configs/2b_480_640_g1_pick_trocar_headcam_rollout_holdout_5s5f_ema_both_no_motion_consistency.yaml
```

输出：

```text
outputs/train/dreamdojo/pick_trocar_headcam_rollout_holdout_5s5f_ema_both_no_motion_consistency/
```

相关评估结果：

```text
results/no_motion_consistency_eval/
results/no_motion_consistency_eval_iter3000_retry/
```

一次 iter 3000 closed-loop 评估的汇总结果约为 MAE 24.37、PSNR 15.28 dB。
该实验是 hallucination / temporal consistency loss 的 ablation，不属于
97F Self-Forcing 主线。

## 4. Self-Forcing 蒸馏实验

完整流程是：

1. Teacher Generation：Teacher 预计算 denoising targets。
2. Warmup：causal student 学习匹配 Teacher 输出。
3. Self-Forcing：使用 student 自己的 autoregressive rollout 继续训练，
   让模型适应自身误差，降低长时域误差累积。

详细通用说明见：

```text
docs/DISTILL.md
```

### 4.1 旧版 49-frame 实验

输出：

```text
outputs/train/dreamdojo/pick_trocar_self_forcing/
```

包含：

```text
g1_pick_trocar_iter2500_warmup
g1_pick_trocar_iter2500_self_forcing
```

特点：

- rollout 长度为 49 帧
- Warmup 训练到约 20,000 iterations
- Self-Forcing 训练到约 3,000 iterations
- 主要用于验证蒸馏流程是否能运行

旧版 49F 评估结果目录已经删除：

```text
results/pick_trocar_distilled_holdout_49f
```

旧版 checkpoint 目前仍保留，属于历史实验，后续可继续清理。

### 4.2 Stratified Teacher Generation

为解决原始 Teacher Generation 只覆盖部分训练 episode 的问题，新增：

```text
scripts/build_teacher_gen_sampling_manifest.py
scripts/validate_teacher_gen_dataset.py
```

manifest：

```text
datasets/g1_pick_trocar_iter2500_teacher_gen_10k_stratified/sample_manifest.jsonl
```

采样特征：

- 共 10,000 个 unique windows
- 覆盖全部 250 个训练 episode
- 每个 episode 40 个 sample
- 一半为均匀时间采样
- 一半提高左臂 / 左手动作变化的采样权重

Teacher Generation 的大缓存（latents、images、actions、videos）已删除，
因为可以依据 manifest 和 Teacher checkpoint 重新生成。当前只保留 manifest。

### 4.3 新版 97-frame 实验

主流程脚本：

```text
scripts/run_pick_trocar_stratified_97f_pipeline.sh
scripts/finish_pick_trocar_97f_evaluation.sh
```

配置：

```text
cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_pick_trocar_self_forcing.py
cosmos_predict2/_src/predict2/interactive/configs/data.py
```

输出：

```text
outputs/train/dreamdojo/pick_trocar_self_forcing_97f/
```

Warmup：

```text
g1_pick_trocar_iter2500_warmup_stratified
```

Self-Forcing：

```text
g1_pick_trocar_iter2500_self_forcing_97f
```

当前保留的可继续训练 checkpoint：

```text
outputs/train/dreamdojo/pick_trocar_self_forcing_97f/g1_pick_trocar_iter2500_warmup_stratified/checkpoints/iter_000003000
outputs/train/dreamdojo/pick_trocar_self_forcing_97f/g1_pick_trocar_iter2500_self_forcing_97f/checkpoints/iter_000003000
```

97F 的含义：

- 1 个初始 frame
- 后续 8 个 12-action chunks
- 共 97 帧
- 15 FPS 下约 6.47 秒逻辑时长

扩展到 97 帧的目的不是声称已经覆盖完整任务，而是让训练和评估暴露
更长时间的 closed-loop 误差累积。任务后半段仍可能没有完全覆盖，因此
97F 结果主要用于判断长期稳定性，而不是完整任务成功率。

## 5. 评估脚本与结果

Teacher 评估：

```text
scripts/validate_pick_trocar_world_model.py
scripts/run_local_holdout_full_eval.sh
scripts/combine_holdout_eval_5x6.py
```

因果 student 评估：

```text
scripts/eval_distilled_holdout.py
scripts/combine_distilled_holdout_eval.py
```

主要结果：

```text
results/pick_trocar_distilled_holdout_97f
results/holdout_5s5f_full_eval
results/holdout_5s5f_full_eval_rerun_20260825_1406_iter3000
```

97F closed-loop 汇总（10 个 held-out cases）：

```text
Teacher 2500  : MAE 20.47, PSNR 16.30 dB
Warmup 3000   : MAE 29.87, PSNR 13.57 dB
Self-Forcing  : MAE 31.84, PSNR 13.11 dB
```

结论：

- Teacher 明显最好。
- Warmup 在 10 个 case 中有 9 个优于 Self-Forcing 97F。
- Self-Forcing 没有改善当前设置下的图像 fidelity。
- 主要视觉问题是约 2 秒以后几何位置和接触关系逐渐漂移。
- 低 training loss 不等价于 closed-loop 长时域稳定。
- 左手抓取阶段是最容易出现漂移和 hallucination 的区域。

## 6. 代码修改清单

核心模型和推理改动：

```text
cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py
cosmos_predict2/_src/predict2/action/inference/inference_gr00t_warmup.py
cosmos_predict2/_src/predict2/interactive/inference/action_video2world.py
cosmos_predict2/_src/predict2/interactive/configs/data.py
cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_pick_trocar_self_forcing.py
```

新增 / 修改的主要脚本：

```text
scripts/build_teacher_gen_sampling_manifest.py
scripts/validate_teacher_gen_dataset.py
scripts/split_rollout_holdout.py
scripts/validate_pick_trocar_world_model.py
scripts/eval_distilled_holdout.py
scripts/combine_holdout_eval_5x6.py
scripts/combine_distilled_holdout_eval.py
scripts/run_local_holdout_full_eval.sh
scripts/run_pick_trocar_stratified_97f_pipeline.sh
scripts/finish_pick_trocar_97f_evaluation.sh
scripts/select_distilled_checkpoint.py
```

基础设施相关修改：

```text
Dockerfile
bin/entrypoint.sh
.dockerignore
.gitignore
```

这些修改用于固定 CUDA / Python 运行环境、允许 Warmup checkpoint 的
`net.max_frames` 覆盖，并支持本地及 Slurm 运行。

## 7. Canvas 分析记录

本机生成过两个 Canvas：

### Pick-Trocar 97F 分析

文件：

```text
.cursor/projects/localhome-local-yunl-DreamDojo/canvases/pick-trocar-97f-analysis.canvas.tsx
```

内容：

- 10 个 held-out episodes
- 97 frames、15 FPS、约 6.47 秒
- Teacher / Warmup / Self-Forcing 的 MAE、MSE、PSNR 对比
- 逐 case 的 MAE 差异
- 误差随时间增长的曲线
- 结论是 Teacher 最好，Warmup 大多数 case 优于 Self-Forcing
- 主要失败模式是 2 秒以后几何和接触漂移

### Training convergence review

文件：

```text
.cursor/projects/localhome-local-yunl-DreamDojo/canvases/training-convergence-review.canvas.tsx
```

内容：

- 分析 iter 140–3000 的 training loss
- 约 iter 2180 达到平滑 loss 最低点
- 3000 iter 附近已经进入平台期
- 但配置没有 validation loss，因此不能仅凭 training loss 证明泛化收敛
- 建议先停在 iter 3000，用固定 holdout rollout 做模型选择

Canvas 文件属于 Cursor 本机工作区资源，不一定会随 Git 仓库同步。
本文档保留了它们的关键结论。

## 8. 当前备份建议

当前主线应保留：

```text
Teacher iter 3000
Warmup 97F iter 3000
Self-Forcing 97F iter 3000
migrated_checkpoints/g1_pick_trocar_headcam_2b_iter_000010000_ema_both
```

原始初始化权重不是当前三模型推理的必需品，但只有约 8.1GB，建议保留，
便于以后重新开展 post-training。

不需要传输：

```text
.venv/
.cache/
旧版中间 checkpoint
旧版 Teacher Generation 大缓存
```

运行环境建议在新机器重新安装，不要直接复制 `.venv`。如目标机器支持
相同的 Apptainer / Singularity 环境，可额外备份：

```text
docker/dreamdojo-cu128-v2.sqsh
```

## 9. 新机器恢复顺序

1. 复制代码、配置、scripts、docs、datasets 和 results。
2. 确认三个主线 checkpoint 的 `model/.metadata` 存在。
3. 安装新机器的 Python / CUDA / NATTEN 环境。
4. 阅读本文档和 `docs/PICK_TROCAR_BACKUP.md`。
5. 先运行 held-out inference smoke test，再继续训练。
6. 若继续 Self-Forcing，使用 97F Warmup iter 3000 作为 student 初始化，
   使用 Teacher iter 3000 作为 teacher。

新机器上的 Agent 可使用以下提示恢复上下文：

```text
请先完整阅读 docs/EXPERIMENT_CONTEXT.md 和 docs/PICK_TROCAR_BACKUP.md，
再根据其中的实验记录、checkpoint 路径和代码修改继续工作。
```
