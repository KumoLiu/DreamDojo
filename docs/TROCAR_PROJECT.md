# Pick-trocar 项目：数据、世界模型与奖励分类器

更新于2026-09-17。本文是本地任务的唯一项目说明，覆盖数据、pipeline、
实验结论、分类器及cluster运行。上游通用教程仍保留在docs/。
RL配置、最佳policy与真机结果以 [RLinf项目文档](../../RLinf/docs/dreamdojo/README.md) 为准，
不在两处维护完整实验清单。

## 1. 当前结论

- DreamDojo 2B基于官方G1 post-train权重做LoRA任务适配，能够用真实动作驱动交接/放置，
  但闭环接触阶段仍有器械漂移、形变和遮挡问题。
- 学习率、rank、训练长度、batch、数据混合与motion-consistency均做过对比。
  LR3e-4 / rank32是效率较好的保留方案；rank64的像素指标略好，但没有证明物理质量更可靠。
- 此处的 `scratch_lr3e-4` 指从官方预训练权重重新适配，不是随机初始化。
- RL使用冻结的非蒸馏WM；WM15表示推理去噪步数，不是15帧训练片段。
- 整体MSE改善有很大部分来自背景；不可据此声称器械/接触物理已解决。
- 正式RL使用冻结的v2三阶段分类器；只保留这一版本，未部署的v3已移出仓库归档。

## 2. 数据与适配

HF来源为 `nvidia/s2r_data_dev`，本地原始数据根
`/localhome/local-yunl/data/`。实际WM训练清单如下。

| 原始数据目录 | train episodes | frames | validation episodes |
| --- | ---: | ---: | ---: |
| pick_trocar_teleop_success_train | 225 | 34,021 | 25 |
| pick_trocar_rollouts_30k_bs256_train | 151 | 19,056 | 17 |
| pick_trocar_rollouts_10k_bs32_train | 115 | 19,312 | 13 |
| 合计 | 491 | 72,389 | 55 |

两个rollout集合来自不同SFT checkpoint，包含失败片段，有助于覆盖偏离演示的状态。
55例不进入WM训练，但已被多次模型选择使用，应称验证集而不是未使用盲测集。
30k验证 `episode_000002.mp4` 已从496帧裁剪为145帧，视频此前已同步cluster。
2026-09-17已将20个配套修正文件上传至HF（1个视频、15个Parquet、4个元数据文件），
固定版本为 [4721087378293026b06f5cd28e8cd8102cde5b95](https://huggingface.co/datasets/nvidia/s2r_data_dev/commit/4721087378293026b06f5cd28e8cd8102cde5b95)。
17条验证episode共2,081帧；上传前修正了遗漏的统计count及后续episode的index统计。
17条视频/动作对齐校验通过，远端41个文件hash与本地一致，20个改动文件重新下载校验通过。
跨机器复评应使用该revision重新拉取，不能混用旧缓存；本次未同步cluster的新统计字段。

原始G1-Dex3为28D，WM dataloader要求43D G1布局。使用
`scripts/adapt_g1_dex3_to_dreamdojo.py` 填充腿/腰维度并匹配
`shared_meta/G1_stats.json`。目标名必须含 `g1`，否则embodiment推断可能出错。

| 原始目录后缀 | DreamDojo datasets/ 下适配目录 |
| --- | --- |
| teleop_success_train | g1_hf_pick_trocar_teleop_success_train |
| teleop_success_validation | g1_hf_pick_trocar_teleop_success_val |
| rollouts_30k_bs256_train | g1_hf_pick_trocar_rollouts_30k_train |
| rollouts_30k_bs256_val | g1_hf_pick_trocar_rollouts_30k_val |
| rollouts_10k_bs32_train | g1_hf_pick_trocar_rollouts_10k_train |
| rollouts_10k_bs32_val | g1_hf_pick_trocar_rollouts_10k_val |

适配示例（目标应为新目录，先确认已有数据，不要重复覆盖）：

```bash
python scripts/adapt_g1_dex3_to_dreamdojo.py \
  --src /localhome/local-yunl/data/pick_trocar_teleop_success_train \
  --dst /absolute/new/g1_hf_pick_trocar_teleop_success_train \
  --camera-key observation.images.cam_head
```

RLinf在线环境仍读取28D原始目录，不直接读取这些43D适配目录。
Reward的有效train数为490，因为分类器另有标注有效性筛选；不与WM491混淆。

## 3. Pipeline与保留的WM配置

Teleop → GR00T N1.7 SFT → 两个SFT checkpoint真机rollout → 合并数据。
合并数据分别用于DreamDojo LoRA适配、带阶段标注的分类器训练；随后在RLinf中
冻结WM和分类器，用GRPO更新GR00T policy，最后对RL与SFT checkpoint做真机对照。
本repo负责WM和分类器；GR00T在线动作桥接、KIR、GRPO与policy评测归RLinf。

官方初始化：`checkpoints/DreamDojo/2B_G1_post-train/iter_000050000`。
主配方：13-frame clips，数据混合0.34/0.33/0.33，LoRA rank32，
global batch32，LR3e-4，18,000次迭代，EMA用于推理。
精确训练状态与覆盖项以对应config / sweep receipt为准。

RL保留的两份cluster权重，根为
`/lustre/fsw/portfolios/healthcareeng/users/yunl/checkpoints/DreamDojo/`：

| 变体 | 权重相对路径 | experiment |
| --- | --- | --- |
| scratch_r32 | lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt | dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora |
| r64 | lora_r32_lr3e-4_r64_18k/checkpoints/iter_000018000/model_ema_bf16.pt | dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora_lr3e-4_r64 |

目录名沿用历史命名；r64那行实际adapter rank为64，不按目录前缀猜测。
本地仅保留最终scratch rank32任务配置，官方10份通用配置保持不动。
r64及其他消融配置已归档，历史权重未删除；加载旧模型需从备份恢复匹配配置，不能套用rank32。
DreamDojo训练迭代 `iter_000018000` 与RL policy代数 `global_step_220` 不是同一个计数。

### 最终使用的WM是怎么训练出来的

真机第220代policy对应的冻结WM是上表`scratch_r32`，不是rank64。
以下设置已核对cluster原训练目录保存的`config.yaml`、`launch_info.yaml`及Slurm日志，
不是从当前基础YAML的默认值推测。训练目录为
`outputs/train/dreamdojo/hf_teleop_rollout_posttrain_lora/lora_r32_scratch_lr3e-4_18k/`
（相对cluster用户根）。

| 项目 | 最终WM实际设置 |
| --- | --- |
| 初始化 | 官方DreamDojo `2B_G1_post-train/iter_000050000`；新建LoRA，不继承之前LoRA的训练状态 |
| 数据 | 上节491条train；teleop / 30k rollout / 10k rollout按0.34 / 0.33 / 0.33采样 |
| 视频 / 动作 | head camera，480×640，13帧；G1每隔2个原始30fps帧采样；28D→43D→384D conditioning |
| LoRA | rank32、alpha32；q/k/v/output投影及MLP layer1/layer2；非全参数微调、非蒸馏 |
| Batch / 并行 | 8 GPU，每卡4，累积1次，global batch32；FSDP，context parallel1 |
| Optimizer | fused AdamW，基础LR3e-4，betas0.9/0.999，weight decay0.1，grad clip0.1 |
| Scheduler | linear，warmup100；cycle400,000，倍率f_max0.99 / f_min0.4；不是18k衰减到零 |
| 训练目标 | action-conditioned rectified flow；motion-consistency权重0.1，沿用原channel差分，不是后来的temporal实验 |
| 迭代 / Seed | 18,000次optimizer更新，WM seed0；不同于RL policy的seed1234 |
| 保存 / 选择 | 每500次保存；训练内validation关闭，55例用于独立评估；最后采用iter18000的EMA BF16导出 |

三个Slurm段为1019862、1020515、1021603；续跑从本run的完整checkpoint恢复。
`scratch`仅表示重新从官方权重进行任务适配，不是从随机参数训练2B模型。
日志确认在18,000次完成。最后导出的`model_ema_bf16.pt`是推理权重，
不是带optimizer/scheduler的完整续训checkpoint，也不是GR00T policy。
RL期间WM保持冻结；train15 / eval35是调用WM时的去噪步数，不是它的训练片段长度。

复用配方时在cluster的DreamDojo根目录提交，**先将RUN_NAME改为新的唯一名字**：

```bash
sbatch --job-name=dd_scratch_r32_reproduction \
  --export=ALL,EXPERIMENT=dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora,RUN_NAME=scratch_r32_reproduction,NUM_GPUS=8,MAX_ITER=18000,LOAD_TRAINING_STATE=false \
  dreamdojo_sweep_train.slurm
```

唯一的任务配置为[最终rank32 YAML](../configs/2b_480_640_g1_hf_teleop_rollout_posttrain_lora.yaml)。
历史基础文件默认LR1e-4 / 3k，训练时通过覆盖项变为LR3e-4 / 18k；
现在已将最终LR、迭代数、seed、optimizer、scheduler和EMA设置写入YAML；
MC直接使用上游固定的channel差分、权重0.1，与最终训练相同，已撤下消融开关，
不再依赖历史sweep或隐藏覆盖项。原experiment名保留，便于RLinf加载已有rank32权重。
原始训练记录保存在本地`outputs/hf_publish/wm_scratch_r32_lr3e-4_iter18000/provenance/`，
包含内部路径，本次不上传；HF仅上传推理权重。
原数据、配套LAM/文本编码器/VAE和兼容的DreamDojo代码仍需另行准备。

### 最终WM的Hugging Face归档

2026-09-17已上传至私有模型仓库
[world_models/lora_r32_scratch_lr3e-4_18k](https://huggingface.co/nvidia/s2r_models_dev/tree/ca865cc10c4d32dbd3ea4361cff508529298de6b/yunl/rl/world_models/lora_r32_scratch_lr3e-4_18k)。
固定revision：`ca865cc10c4d32dbd3ea4361cff508529298de6b`。
唯一上传文件为`model_ema_bf16.pt`，4,395,116,963 bytes，SHA256：
`2ea3050704a31e40e3f90cd9457676d809622d62dcf46bc010b03603e66fe4b4`。
cluster、本地副本与HF LFS元数据的大小/hash一致；原仓库549个文件未改动。
这是完整网络及未合并LoRA参数的推理导出，不是仅adapter，也不包含optimizer/scheduler。

登录有权限的HF账号后下载：

```bash
hf download nvidia/s2r_models_dev \
  yunl/rl/world_models/lora_r32_scratch_lr3e-4_18k/model_ema_bf16.pt \
  --revision ca865cc10c4d32dbd3ea4361cff508529298de6b \
  --local-dir /absolute/model_download
```

本地副本：`/localhome/local-yunl/models/dreamdojo/lora_r32_scratch_lr3e-4_18k/checkpoints/iter_000018000/model_ema_bf16.pt`。
CPU安全加载检查确认rank32、280个LoRA A和280个LoRA B张量；未另做GPU推理。
上传回执及本地检查在`outputs/hf_publish/wm_scratch_r32_lr3e-4_iter18000/`，
该目录是忽略版本控制的产物，不是新的源码入口。

## 4. WM消融的主要发现

同55例、closed-loop、`HORIZON=auto`、0–255像素尺度MSE。
下表仅选关键结果；完整27次WM评测表在旧SLURM_SWEEPS归档原文，
逐episode结果仍在 `outputs/eval/`（不在本轮清理范围内）。

| 对照 | 训练迭代 | CL-MSE（低更好） | 能支持的结论 |
| --- | ---: | ---: | --- |
| Full finetune / LoRA | 3k | 1285.3 / 1158.5 | 当时LoRA更好；LR等设置不同，不是纯参数化消融 |
| LR1e-4 baseline | 9k → 18k | 1020.5 → 929.1 | 增加训练长度有收益 |
| scratch LR2e-4 / 3e-4 / 4e-4，r32 | 18k | 882.4 / 874.3 / 910.6 | 2–3e-4附近较好；3e-4不是唯一可靠最优值 |
| LR3e-4 rank64 | 18k | 860.0 | 像素分数略好，不能据此判定更物理正确 |
| scratch gb64（LR1e-4） | 18k | 875.8 | 与gb32/LR3e-4接近，但消耗两倍样本 |
| scratch mix0.6/0.2/0.2 | 18k | 937.0 | 多采teleop无一致优势 |
| rank16系列 | 18k | 1011.0–1054.4 | 当时预算下未优于rank32；组合设置需看原config |
| clip25 | 18k | 1091.0 | 同迭代更差且每步约1.8倍耗时 |
| LR1e-5 | 9k | 1890.4 | 同9k预算明显不足，不能与18k主模型当等预算比较 |
| LR3e-4 MC off / time / time-w0.5 | 18k | 885.1 / 918.7 / 871.2 | 未看到稳定优于原方案的证据 |

motion-consistency原实现对B C T H W的dim1求差，实际是channel而非时间。
后续把MC关掉、改时间轴、降低权重均做过实验；不是只“修正一个维度”就消除了漂移。
MC消融分支现已撤回上游，仅保留最终训练的固定channel/0.1损失；
若要重现其他MC实验，需要从备份同时恢复代码和配置。

不要只按训练迭代比较不同batch：batch翻倍同时翻倍样本/计算。
也不要只按late/early比值判断漂移，更差的early基线可能让比值虚假变好。
Teacher forcing每块重置真实图像，closed loop消费自己的预测，二者都用真实未来动作时，
差距主要诊断累积误差，而非证明policy能力。

## 5. 如何解读WM评测

两种模式都可用同一组真实动作，区别在图像条件：teacher forcing每块重置真实图像；
closed loop使用上块预测。闭环更接近RL中的使用方式，但二者都不能代替policy真机评测。

| 指标 | 用途与限制 |
| --- | --- |
| MSE / MAE | 0–255像素尺度；背景可能掩盖手、器械和接触区域的错误 |
| PSNR | MSE的单调变换，不是独立的第二项证据 |
| moving_region_mse / motion_weighted_mse | 更关注运动区域；运动区域不等同于任务物体 |
| mse_early / mse_late / late_over_early | 同时看绝对误差和比值；较差的early分母会让比值虚假改善 |
| SSIM / LPIPS | 补充结构与感知质量；仍不衡量动作服从性或物理正确性 |

`video_metrics.py`计算逐episode指标，`compare_eval_runs.py`做配对比较
（含Wilcoxon signed-rank与逐case胜负数）。比较必须使用相同数据版本、case、
去噪步数、horizon和初始化协议。不同batch同迭代不等于同样本预算。

感知指标默认不启用；可用`rescore_eval_videos.py`从保存的comparison视频补算，
再用`compare_eval_runs.py --filename metrics_extended.json`比较。
视频经过有损编码，补算值不等同于原始帧指标，不应混为同一协议。
LPIPS的late/early比值在此前小规模比较中较少受误差基线影响，
但不能据此声称它一般性地独立于误差水平或能证明物理稳定。
成功/失败分组工具保留为`split_eval_by_outcome.py`，不把未执行的分组算作已有结果。

## 6. 三阶段奖励分类器

### 正式使用的v2

Milestone指任务阶段，不是训练checkpoint。模型输出picked / handed / placed三个独立概率。
ResNet18输入4帧channel堆叠，历史偏移0/4/8/16@30fps，只看当前和过去。
训练使用masked BCE；不确定的head跳过loss，不做左右翻转。

阶段标注来自`meta/episodes.jsonl`的`source_annotation.cleaned_phase_frames`及审阅修订：

- teleop train225条中ep98无标注，排除；ep211 placement越界1帧已修正。
  有效224条、34,000帧；val25条中ep11无可监督放置事件，排除后24条。
- 加266条rollout，合计490条train、72,368帧。失败和部分成功是必要负例。
- 未发生的阶段用null；拿起后掉落不继续标为picked；过渡/不清晰状态用mask。
- `reviewed_labels_v3.json`是v2的标签修订号，不是四头v3模型。

必须保留的资产：`outputs/milestone/v2/best.pt`、`reviewed_labels_v3.json`、
`drops_manual.json`、`auto_labels_v2_realnew.npz`、`auto_labels_v1_realnew.npz`
（后四项均在`outputs/milestone/`）。v2权重SHA256：
`2dd843e6a78a72a2af626bbdcf7bc9df182ac68f0119d45f1d585e17f93080d7`。

复现示例使用新name，勿覆盖既有v2；需保持原split、mask和实际reward预处理：

```bash
.venv/bin/python -m scripts.milestone.extract_frames
.venv/bin/python -m scripts.milestone.train --name v2_reproduction --epochs 8 \
  --extra-labels outputs/milestone/reviewed_labels_v3.json
.venv/bin/python -m scripts.milestone.calibrate_reward \
  --checkpoint outputs/milestone/v2_reproduction/best.pt --refresh
```

训练参数已按最终checkpoint核对：8 epochs、batch128、LR3e-4、weight decay1e-4、
seed0；ImageNet ResNet18初始化、OneCycleLR，按teleop验证mAP保存best（epoch3，0起算）。
`train.py`默认使用最终标签，必须指定新`--name`，已有目录会拒绝覆盖。
仅保留抽帧、标签/数据加载、模型、训练、推理、解码及reward校准这一条代码路径。
旧版本专用训练/评测/网页及一次性标注、裁剪脚本已删除；修订结果保留在标签数据中。
`review_v1_val/review_sheet.csv`是沿用的30条人工校准标签，不需要v1模型，仍保留。

### RL实际reward与效果

各头阈值0.8；pick/hand需最近15 tick中13个通过，place需2/2。
RLinf将15fps生成帧重复两次匹配30Hz历史，place两票可能来自同一生成帧。
阶段历史锁存0→3，每个新阶段一次+1、累计最多3；掉落不会撤回已发reward。
多个头可同tick推进；KIR已pick起点初始化stage1、return0，不奖励既有进展。
单帧分类、锁存stage、人工判定成功不能混用。

旧30条审阅视频经校准30/30判定正确；额外120条真实视频（60成功/60失败）119/120正确，
没有把其中60条失败判成完整成功。这是已录真实视频上的分类器结果，
不是policy成功率，也不是WM生成域准确率。
生成域仍有未pick误判pick、器械复制/形变及双手重叠误判交接；历史锁存可能放大误奖。

独立视频评分入口为`python -m scripts.milestone.reward --video VIDEO --checkpoint CKPT`；
该CLI不等同于带15→30Hz与KIR处理的RLinf批量adapter。

### 分类器实验结论（简要）

初版v0只用teleop，容易把掉落误判成交接；加入真实失败rollout和审阅标签形成v1。
最终v2进一步加入掉落时刻、部分head的mask和标签修订，使“拿起后掉落”不再一直标正，
并用含失败案例的30条真实视频校准确认窗口。正式RL始终冻结这份v2。

2026-09-14试过v3：在v2上加入AI稀疏审阅的WM生成帧和validity head。
阶段召回改善，但留出稀疏点中异常只检出3/10，同时误拦7/171正常点；
未证明解决生成域误奖，因此未接入RL。稀疏AI标签也不能视为人工逐帧真值。
旧版本代码、测试和v0/v1/v3产物已退出当前repo；不再维护复核网页。
归档可恢复，但RL130原视频此前已删除，无法从归档完整重建原始输入。

## 7. 本地与cluster运行入口

保留的是可复用工具，不保留绑定某个旧job的等待/续评包装。

| 入口 | 用途 |
| --- | --- |
| sync_to_slurm.sh | 检查并同步所选代码/资产；执行前查看清单，避免重复传已有数据 |
| dreamdojo_sweep_train.slurm | 任意数量数据集、独立run目录、Slurm训练 |
| dreamdojo_sweep_eval.slurm | 选定WM、转换checkpoint、统一评测/视频 |
| launch_local.sh | 指定唯一任务experiment和新的job.name进行本地训练 |
| scripts/submit_sweep.py | 通用工具，读取用户另行准备的新sweep，支持dry-run与--only；不附旧实验表 |
| scripts/lib/chain.sh | DreamDojo自动续跑 |
| scripts/validate_pick_trocar_world_model.py | teacher-forced / closed-loop评估 |
| scripts/video_metrics.py / compare_eval_runs.py | 逐case指标与配对比较 |
| scripts/analyze_train_loss.py / plot_train_loss.py | 跨续跑段训练曲线 |

本地训练使用最终YAML，必须先选择新的job.name；下列命令会启动训练：

```bash
NPROC=8 bash launch_local.sh \
  dreamdojo_2b_480_640_g1_hf_teleop_rollout_posttrain_lora \
  job.name=my_new_scratch_r32_run
```

旧消融YAML、sweep表及五个绑定旧配置的启动包装已移出仓库并备份。
通用`submit_sweep.py`仍可读取外部新spec；去掉`--dry-run`会实际提交。
`--dry-run`只打印命令，不生成文件；实际提交才在被git忽略的
`outputs/sweeps/<sweep>/<run>.overrides`写入参数，通过`OVERRIDES_FILE`传给Slurm。
这是运行时传参文件，避免逗号/空格被sbatch错误拆分，不作为源码保留；
对应排队、训练和续跑结束前不要删除。实际训练参数以run保存的config / launch_info为准。
历史最终run通过`OVERRIDES=optimizer.lr=0.0003`提交；现已固化进最终YAML。
推理加载权重不需要历史`.overrides`文件。
逗号复杂值使用独立config，避免sbatch --export / Hydra的双重解析。

Cluster文件根为`/lustre/fsw/portfolios/healthcareeng/users/yunl/`，
代码位于`code/DreamDojo`，数据在`datasets/`，权重在`checkpoints/`，产物在`outputs/`。
常用路径变量：
`USER_BASE`、`REPO_ROOT`、`DATASET_ROOT`、`CHECKPOINT_ROOT`、
`OUTPUT_ROOT`、`EVAL_ROOT`、`CONTAINER_IMAGE`。
训练模板将数据集改写为 `/data/<basename>` 并核验 `meta/info.json`。
WM环境历史镜像为 `dreamdojo-cu128-v2.sqsh`；
RLinf使用另一个独立SQSH，不能混为同一运行环境。

评估使用与checkpoint结构匹配的13-frame配置；
`HORIZON=auto`不为长固定窗口重复padding；新协议设置新的 `EVAL_TAG` 避免覆盖结果。
本地其他rank/LR配置已归档；cluster旧代码及原运行快照本次未改动。
RLinf本地源码的默认WM权重与experiment已成对改为最终scratch rank32，
配方为policy noise0.3、LR5e-6、WM train15/eval35、KIR关闭、冻结v2。
主YAML/cluster为8卡、64 env×2、global batch128/micro8；本地包装避开GPU4，
使用7卡、56 env×2、global batch112/micro8，每代仍20次更新，但采样量不同。
旧cluster快照及SQSH尚未同步这些默认值。历史r64必须先恢复匹配配置，不能只换权重路径。

自动续跑由`dreamdojo_sweep_train.slurm`调用`scripts/lib/chain.sh`；
`CHAIN_TIMEOUT`默认3.9h，超时后在`MAX_RUNS`内提交后继job，
使用同一run目录与`checkpoints/latest_checkpoint.txt`恢复，正常结束不继续提交。
新实验必须用独立run名；不要让两个job写同一个checkpoint目录。

## 8. 代码边界、检查与整理记录

2026-09-17已通过官方Git远端核实：`NVIDIA/DreamDojo`的`main`为
`02f119b759d5c7f84a399fdeea3c6e82e7ed6cff`。以下审查以它为基线，
不是最初的initial commit，也不是本地最近一次cleanup提交。
上游自己的蒸馏、多机器人支持和10份官方YAML不作为本地冗余修改回退。

当前仅保留下列原有Python文件补丁，以及新加的文本缓存模块：

| 文件/模块 | 保留原因 |
| --- | --- |
| external/lam/model.py | 明确LAM路径，缺文件立即失败、严格加载，避免未加载权重仍继续 |
| inference/video2world.py、inference/text_embedding_cache.py | 批量输入、可选文本缓存与offload |
| models/text2world_model_rectified_flow.py | 仅保留两处batch采样修复；训练损失已恢复上游 |
| action/models/action_conditioned_video2world_rectified_flow_model.py | 可选guidance=0跳过无用分支 |
| imaginaire/lazy_config/lazy.py | resolver重复注册兼容 |
| imaginaire/utils/checkpoint_db.py | 固定可用tokenizer revision；上游旧revision的tokenizer.pth经查询返回404 |

表中inference/models/action路径均相对`cosmos_predict2/_src/predict2/`，
imaginaire相对`cosmos_predict2/_src/`。Tokenizer保留revision
`85f8ae7bfe8f5525c8d103429524dcf12f98bf7b`，没有改成浮动main。
Dockerfile/entrypoint保留已验证的依赖安装、Pyxis Python路径与可跳过sync开关；
这些属于环境支持。`.dockerignore`不再排除官方`docker/`源码。

已撤回的不必要上游修改：

| 文件/功能 | 处理与依据 |
| --- | --- |
| groot_dreams/data/dataset.py | 完全恢复上游；移除sample_start/sample_end过滤、恢复统计量提示。最终491 train+55 val均无这些字段，采样集合不变 |
| action/inference/inference_gr00t_warmup.py | 完全恢复上游teacher-generation入口；撤掉额外dataset_path参数和完成样本提前返回，不属于最终非蒸馏WM流程 |
| interactive/inference/action_video2world.py | 完全恢复上游；撤掉最终流程未使用的experiment_opts扩展 |
| MC消融开关 | 撤回可调weight/temporal分支及最终YAML的相应字段，保留上游固定channel/0.1行为；不改变最终模型的训练目标 |

旧`scripts/prepare_real_rollout_lerobot.py`依赖已撤回的contact-only采样字段，
不用于最终HF数据，已移除并备份，避免继续产出不生效的区间标记；没有删除任何真实数据。
`groot_dreams/data/transform/state_action.py`已按要求恢复上游原版：
直接导入`pytorch3d.transforms`，缺少依赖时立即报错，不再作为可选依赖。
本地DreamDojo/RLinf两个venv目前都缺少pytorch3d；实际WM启动前需补装。
本轮仅做CPU/结构检查，没有伪造依赖完成模型加载，也没有安装依赖、启动训练或修改cluster镜像。
RLinf侧`toolkits/world_model/dreamdojo_validation.py`验证跨repo动作桥接/条件一致性，
不迁入本repo；这里的`validate_pick_trocar_world_model.py`仍用于独立WM评估。

分类器仅保留`tests/milestone/test_v2.py`这一个CPU安全测试文件：
掉落/mask、因果帧堆叠、三头checkpoint格式、最终默认参数及防覆盖。
不依赖旧实验产物、不下载预训练模型；使用现有环境，不执行uv sync：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /localhome/local-yunl/RLinf/.venv/bin/python -m pytest \
  --noconftest -p no:cacheprovider -c /dev/null -q tests/milestone
```

CI保持不变；RLinf端动作桥接、推理、LAM、续跑等检查按其项目文档运行。
仅保留最终任务YAML与官方配置；原消融配置和旧sweep表可从仓库外备份恢复。
原版通用蒸馏框架、上游教程也保留，不与已撤下的任务专用蒸馏混淆。

2026-09-17：三份本地项目文档合并；退休9个旧迁移/固定run脚本及一次性页面，
移动分类器测试并更新导出引用。改前15个文件与hash清单保存在
`/localhome/local-yunl/code_cleanup_archive/20260917_dreamdojo_repo.o2slvvrp/`。
本次不删除数据、权重、输出，不同步cluster，不重建镜像。

同日按要求进一步精简任务配置：保留最终1份，删除其他28份任务YAML、旧sweep表和
五个旧配置专用入口。备份及校验记录：
`/localhome/local-yunl/code_cleanup_archive/20260917_final_wm_config.3lyl6wr9/`。

同日统一RLinf最优默认配置、只保留正式v2分类器；14个旧实验/标注脚本及v3测试移除，
v0/v1/v3输出移到仓库外（可恢复），v2权重和正式标签未改，原v3网页6010已停止。
备份：`/localhome/local-yunl/code_cleanup_archive/20260917_best_defaults_classifier.jp3dxhnS/`。

同日按官方main完成上游补丁审查和精简；原文件、现有暂存/未暂存差异及验证记录在
`/localhome/local-yunl/code_cleanup_archive/20260917_upstream_audit.gZE0UCcb/`。

更早的四篇长流水账逐字原文在
`/localhome/local-yunl/code_cleanup_archive/20260916_project_closeout.bA84wJ/DreamDojo/files/docs/`。
项目汇总及历史归档位置见 [统一项目文档](../../RLinf/docs/dreamdojo/README.md#cluster)。
代码、配置、数据、原始结果的关系优先于旧文档中的“接下来/目前最好”等阶段性描述。
