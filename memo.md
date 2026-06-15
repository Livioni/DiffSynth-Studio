# 本地改动备忘录

1. rollout a video

```bash
python examples/wanworldmodel/rollout.py --task beat_block_hammer --episode episode55 --num_inference_steps 500
```

## 常用工作流

1. 生成 robot action 归一化统计：

```bash
python examples/wanworldmodel/tools/compute_robot_metadata.py \
  --root world_model_data/robotwin_aloha \
  --output_path world_model_data/robotwin_aloha/metadata.json
```

2. 下载周期评估需要的 LPIPS/FID 权重：

```bash
python examples/wanworldmodel/tools/download_eval_metric_weights.py \
  --download_source modelscope
```

3. 可选：先统计各 task 的运动强弱，排查低运动窗口：

```bash
python examples/wanworldmodel/tools/stat_robotwin_motion.py \
  --dataset_base_path world_model_data/robotwin_aloha/train_set,world_model_data/robotwin_aloha/robotwin_aloha_fail \
  --num_frames 25 \
  --stride 6
```

4. 训练 no-language world model：

```bash
bash examples/wanworldmodel/model_training/train_robotwin_aloha_no_lang.sh
```

5. 推理单个 eval window：

```bash
python examples/wanworldmodel/infer.py \
  --eval_dataset_base_path world_model_data/robotwin_aloha/val_set \
  --task adjust_bottle \
  --episode episode0 \
  --start_frame 0 \
  --checkpoint_path outputs/WanWorldModel_film_no_language_abs_action_fix/step-20000.safetensors \
  --output_dir outputs/inference
```

6. 看数据集可视化：

```bash
python -m worldmodel_rerun.app \
  --roots world_model_data/robotwin_aloha/train_set,world_model_data/robotwin_aloha/val_set,world_model_data/robotwin_aloha/robotwin_aloha_fail \
  --port 7860 \
  --rerun-web-port 9090
```

打开打印出的 Web UI 地址，选 root/task/episode 后点 Open Rerun。

## 新增脚本

### `examples/wanworldmodel/model_training/train_robotwin_aloha.sh`

用途：带语言条件的 RoboTwin ALOHA world model 训练包装脚本。

特点：

- 调用 `examples/wanworldmodel/train.py`。
- 默认使用 `world_model_data/robotwin_aloha`。
- 默认 `world_model_video_camera=head_camera`。
- 默认 `num_frames=25`、`height=240`、`width=320`、`world_model_stride=12`。
- 默认训练模块是 `dit,action_embedder`。
- 默认 action 注入方式是 `film`。
- 默认会加载 T5 text encoder/tokenizer，因此可以使用 episode instruction/prompt。
- 输出目录是 `outputs/WanWorldModel_film`。

运行：

```bash
bash examples/wanworldmodel/model_training/train_robotwin_aloha.sh
```

如果 LPIPS/FID 权重缺失，先跑：

```bash
python examples/wanworldmodel/tools/download_eval_metric_weights.py --download_source modelscope
```

### `examples/wanworldmodel/model_training/train_robotwin_aloha_no_lang.sh`

用途：无语言条件的 RoboTwin ALOHA world model 训练包装脚本。

特点：

- 不加载 T5 text encoder/tokenizer。
- 传入 `--disable_language_condition`，DiT 会跳过 context attention。
- 当前 action 注入方式是 `film`，适合 no-language；`context` 不适合 no-language。
- 默认训练 root 是：
  - `world_model_data/robotwin_aloha/train_set`
  - `world_model_data/robotwin_aloha/robotwin_aloha_fail`
- 默认 eval root 是 `world_model_data/robotwin_aloha/val_set`。
- 默认 `num_processes=6`。
- 默认 `num_frames=25`、`height=256`、`width=320`、`world_model_stride=6`。
- 输出目录是 `outputs/WanWorldModel_film_no_language_abs_action_fix`。
- 当前工作区里这个脚本相对已跟踪版本有两处未提交改动：
  - `eval_steps=200`
  - `eval_max_samples=32`

运行：

```bash
bash examples/wanworldmodel/model_training/train_robotwin_aloha_no_lang.sh
```

断点恢复时，把脚本里的：

```bash
resume_training_checkpoint="latest"
```

或者指定：

```bash
resume_training_checkpoint="outputs/.../training_checkpoints/step-20000"
```

### `examples/wanworldmodel/train.py`

用途：Wan world model 的真实训练入口。

核心功能：

- 支持 `dataset_type=world_model|unified|auto`。
- `world_model` 模式直接使用 `WorldModelDataset` 读取 RoboTwin ALOHA 数据。
- 自动把训练视频首帧作为 `input_image` 条件。
- 从 `robot_data` 拼出 action tensor，默认维度 14。
- 支持 action 归一化：
  - `--action_metadata_path`
  - `--action_metadata_key robot_statistics`
  - `--action_normalization_mode standard|scale_only`
- 支持 no-language 训练：
  - `--disable_language_condition`
  - 此时不加载 text encoder/tokenizer，且跳过 context attention。
- 支持周期 eval：
  - 生成 `x_pred`
  - 保存/记录 `x_gt`、`x_pred`、`x_reconst`
  - 计算 `val/mse`、`val/psnr`、`val/ssim`、`val/lpips`、`val/fid`
- 支持 warmup cosine LR：
  - `--lr_scheduler warmup_cosine`
  - `--lr_warmup_steps`
  - `--lr_cosine_min_ratio`
- 支持完整训练状态 checkpoint：
  - `--resume_training_checkpoint latest`
  - `--training_checkpoint_dir`
  - `--disable_training_checkpoint`

关键参数：

```bash
--dataset_base_path world_model_data/robotwin_aloha/train_set,world_model_data/robotwin_aloha/robotwin_aloha_fail
--dataset_type world_model
--world_model_tasks adjust_bottle,open_laptop
--world_model_video_camera head_camera
--world_model_stride 6
--world_model_include_failed
--height 256
--width 320
--num_frames 25
--extra_inputs input_image
--trainable_models dit,action_embedder
--action_dim 14
--action_injection_method film
--action_metadata_path world_model_data/robotwin_aloha/metadata.json
--disable_language_condition
```

### `examples/wanworldmodel/infer.py`

用途：加载训练好的 world model checkpoint，对 eval dataset 中某个 window 做推理。

输出：

- `*_pred.mp4`：生成视频。
- `*_gt.mp4`：对应 ground truth 视频。
- `*_input.png`：首帧输入图。
- `*_action.npy`：该 window 的 action。
- `*.json`：本次推理 metadata。

先列出可用样本：

```bash
python examples/wanworldmodel/infer.py \
  --eval_dataset_base_path world_model_data/robotwin_aloha/val_set \
  --list_samples
```

no-language checkpoint 推理：

```bash
python examples/wanworldmodel/infer.py \
  --eval_dataset_base_path world_model_data/robotwin_aloha/val_set \
  --task adjust_bottle \
  --episode episode0 \
  --start_frame 0 \
  --checkpoint_path outputs/WanWorldModel_film_no_language_abs_action_fix/step-20000.safetensors \
  --action_metadata_path world_model_data/robotwin_aloha/metadata.json \
  --action_injection_method film \
  --output_dir outputs/inference
```

如果 checkpoint 是带语言条件训练的，要加：

```bash
--enable_language_condition
```

如果 checkpoint 里推不出 action embedder 维度，手动传：

```bash
--action_dim 14
--action_embedder_hidden_dim <hidden_dim>
```

### `examples/wanworldmodel/action_utils.py`

用途：把 `WorldModelDataset` 返回的 robot dict 拼成 action tensor。

拼接顺序：

1. `left.action.arm_joint`
2. `left.action.gripper`
3. `right.action.arm_joint`
4. `right.action.gripper`

输出形状通常是 `[T, 14]`。

### `examples/wanworldmodel/tools/compute_robot_metadata.py`

用途：扫描 RoboTwin ALOHA episode 的 `robot_data/*.npy`，计算 action/state 的 mean/std，写入 `metadata.json`。

这个 metadata 后面用于 action normalization。

默认写入 key：

```text
robot_statistics
```

常用：

```bash
python examples/wanworldmodel/tools/compute_robot_metadata.py \
  --root world_model_data/robotwin_aloha \
  --output_path world_model_data/robotwin_aloha/metadata.json
```

只统计部分 task：

```bash
python examples/wanworldmodel/tools/compute_robot_metadata.py \
  --root world_model_data/robotwin_aloha/train_set \
  --tasks adjust_bottle,open_laptop \
  --output_path world_model_data/robotwin_aloha/metadata.json
```

严格模式会在缺文件或同组帧数不一致时报错：

```bash
--strict
```

### `examples/wanworldmodel/tools/download_eval_metric_weights.py`

用途：下载并校验 world model eval 所需的 LPIPS/FID 权重。

默认下载：

- FID
- LPIPS vgg

常用：

```bash
python examples/wanworldmodel/tools/download_eval_metric_weights.py \
  --metric all \
  --lpips_net vgg \
  --download_source modelscope
```

只下载 FID：

```bash
python examples/wanworldmodel/tools/download_eval_metric_weights.py --metric fid
```

强制重新下载：

```bash
python examples/wanworldmodel/tools/download_eval_metric_weights.py --force
```

### `examples/wanworldmodel/tools/stat_robotwin_motion.py`

用途：按 task 或 root/task 统计 world model window 的图像运动量和 action 变化量，方便找低运动数据、失败样本分布或任务难度差异。

注意：这是当前未跟踪文件，但脚本已在工作区。

默认统计：

- 图像相邻帧 L1：
  - `image_frame_l1_mean`
  - `image_frame_l1_max`
  - `image_first_last_l1`
- action L2：
  - `action_l2_mean`
  - `action_l2_max`
  - `action_delta_l2_mean`
  - `action_delta_l2_max`
  - `action_first_last_l2`
- 低运动比例：
  - image frame low
  - image first-last low
  - action low
  - image/action 同时 low

常用：

```bash
python examples/wanworldmodel/tools/stat_robotwin_motion.py \
  --dataset_base_path world_model_data/robotwin_aloha/train_set,world_model_data/robotwin_aloha/robotwin_aloha_fail \
  --camera head_camera \
  --num_frames 25 \
  --stride 6 \
  --output_csv outputs/robotwin_motion_stats_by_task.csv \
  --output_json outputs/robotwin_motion_stats_by_task.json
```

只算 action，跳过图像读取：

```bash
python examples/wanworldmodel/tools/stat_robotwin_motion.py --metrics action
```

按 root/task 分开统计 train 和 fail：

```bash
python examples/wanworldmodel/tools/stat_robotwin_motion.py --group_by root_task
```

每个 group 最多采样 1000 个 window：

```bash
python examples/wanworldmodel/tools/stat_robotwin_motion.py --max_windows_per_group 1000 --seed 0
```

输出逐 window 明细：

```bash
python examples/wanworldmodel/tools/stat_robotwin_motion.py \
  --window_csv outputs/robotwin_motion_stats_windows.csv
```

### `worldmodel_rerun/app.py`

用途：WorldModelDataset 的 Web + Rerun 可视化工具。

功能：

- Web 页面选择 root/task/episode。
- 后台把 episode 渲染成 `.rrd`。
- 自动启动 `rerun --serve-web`。
- 在浏览器里看：
  - 多 camera RGB
  - depth
  - left/right action 曲线
  - left/right state 曲线
  - episode meta/prompt

运行：

```bash
python -m worldmodel_rerun.app \
  --roots world_model_data/robotwin_aloha/train_set,world_model_data/robotwin_aloha/val_set,world_model_data/robotwin_aloha/robotwin_aloha_fail \
  --cameras head_camera,left_camera,right_camera,third_view \
  --port 7860 \
  --rerun-web-port 9090 \
  --depth-meter 1000
```

默认生成的 `.rrd` 放在：

```text
/tmp/worldmodel_rerun_recordings
```

依赖：需要能找到 `rerun` 可执行文件。

### `upload_ms.py`

用途：把 RoboTwin ALOHA task 文件夹逐个打包成 `.tar.gz`，上传到 ModelScope dataset repo。

默认 roots：

- `world_model_data/robotwin_aloha/train_set`
- `world_model_data/robotwin_aloha/val_set`

默认会：

- 每个 task 单独打包。
- 上传前尝试列出远端文件。
- 远端已有则跳过。
- 上传成功后删除本地临时 archive。
- 有 `pigz` 时用并行压缩。

干跑看计划：

```bash
python upload_ms.py --dry-run
```

只上传某些 task：

```bash
python upload_ms.py --only adjust_bottle open_laptop
```

带 root-level 文件，例如 `metadata.json`：

```bash
python upload_ms.py --include-root-files
```

建议使用环境变量传 ModelScope token，不要把 token 写进命令历史：

```bash
export MODELSCOPE_SDK_TOKEN=...
python upload_ms.py --repo-id <namespace>/<dataset_repo>
```

备注：当前源码参数里有默认 token；这个 memo 不记录具体 token，后续最好把源码默认值改成 `None` 或只读环境变量。

### `wan_image_text_infer.py`

用途：普通 Wan2.2-TI2V-5B image + text to video 推理脚本，不使用 robot action，也不使用 world model checkpoint。

可以当 baseline 或 sanity check。

运行：

```bash
python wan_image_text_infer.py \
  --input_image path/to/frame_000000.png \
  --prompt "A robot arm manipulates an object on a table." \
  --output outputs/wan_image_text.mp4 \
  --height 480 \
  --width 640 \
  --num_frames 25
```

显存相关：

```bash
--vram_limit_gb 20
--vram_reserve_gb 2
--disable_vram_management
--no_tiled
```

### `.vscode/launch.json`

用途：VS Code debug 配置。

包含：

- Wan2.2-TI2V-5B 普通训练 direct debug。
- Wan2.2-TI2V-5B accelerate launch。
- WanWorldModel direct debug。
- WanWorldModel accelerate launch。

注意：`.gitignore` 里把 `/.vscode` 的 ignore 注释掉了，所以这个配置现在会进入版本管理。

## 新增核心模块

### `diffsynth/core/data/world_model_dataset.py`

新增 `WorldModelDataset`。

目标数据结构大致是：

```text
root/
  task_name/
    episode0/
      meta.json
      camera_data/
        images/<camera>/frame_*.png
        depths/<camera>/frame_*.npy 或 frame_*.png
        intrinsics/<camera>/intrinsic.npy
        extrinsics/<camera>/frame_*.npy
      robot_data/
        left_arm_joint_action.npy
        left_gripper_action.npy
        right_arm_joint_action.npy
        right_gripper_action.npy
        left_endpose.npy
        left_endpose_gripper.npy
        right_endpose.npy
        right_endpose_gripper.npy
```

重要参数：

- `root`：可以是单个 root，也可以是逗号分隔的多个 root。
- `tasks`：只加载指定 task；为空加载全部 task。
- `cameras`：要加载的 camera。
- `num_frames`：每个样本窗口长度。
- `stride`：滑动窗口步长；为空时等于 `num_frames`。
- `include_depth`：是否加载 depth。
- `include_camera_params`：是否加载 intrinsics/extrinsics。
- `include_failed`：是否包含 `meta.json` 标记失败的 episode。
- `repeat`：虚拟重复数据集长度。
- `max_data_items`：限制样本数，eval 时常用。

每个 item 返回：

- `task`
- `episode`
- `episode_path`
- `prompt`
- `frame_indices`
- `cameras`
- `robot`
- `meta`

### `diffsynth/pipelines/wan_world_model.py`

新增 `WanWorldModelPipeline`。

输入：

- `input_image`：首帧图像条件。
- `prompt`：文本条件，可关闭。
- `action`：robot action 序列，支持 tensor/list/dict。
- Wan 推理参数：`height`、`width`、`num_frames`、`seed`、`cfg_scale`、`num_inference_steps`、`sigma_shift`、`tiled`。

action 相关：

- `action_dim`：默认训练脚本用 14。
- `action_injection_method`：
  - `none`：不用 action。
  - `context`：把 action embedding 写进 text context token；需要 language/context attention。
  - `additive`：加到 latent token。
  - `cross_attention`：额外 action cross attention。
  - `adaln`：加到 AdaLN modulation。
  - `film`：用 action 生成 gamma/beta 调制 token。
- `action_metadata_path`：action mean/std metadata。
- `action_normalization_mode`：
  - `standard`：`(action - mean) / std`
  - `scale_only`：`action / std`

no-language 模式：

- `use_text_condition=False`
- 训练脚本参数是 `--disable_language_condition`
- 推理脚本默认就是 no-language；带语言 checkpoint 要显式 `--enable_language_condition`
- no-language 下不要用 `action_injection_method=context`

### `diffsynth/models/wan_video_dit.py` 和 `diffsynth/pipelines/wan_video.py`

给 Wan DiT 增加 action 条件能力：

- `normalize_action_injection_method()`
- `WanModel.action_embedding_dim()`
- `WanModel.configure_action_injection()`
- `DiTBlock` 支持：
  - additive
  - cross attention
  - AdaLN
  - FiLM
- `disable_context_attention` 可以跳过 text cross attention。
- `model_fn_wan_video()` 增加 `action_emb`，并把 per-frame action embedding 对齐到 latent tokens。

限制：

- sliding-window Wan inference 暂不支持 `action_emb`。
- 某些依赖 context 的 adapter 和 `disable_context_attention` 不兼容。

## 训练框架新增能力

### 随机视频起点

`UnifiedDataset.default_video_operator()` 新增 `random_start`。

默认行为从“固定取视频前缀”变成“随机取合法起点的 clip”。

关闭：

```bash
--disable_video_random_start
```

已接入：

- `examples/wanvideo/model_training/train.py`
- `examples/ltx2/model_training/train.py`
- `examples/mova/model_training/train.py`

### warmup cosine LR

`diffsynth/diffusion/runner.py` 新增：

```bash
--lr_scheduler warmup_cosine
--lr_warmup_steps 1000
--lr_cosine_min_ratio 0.1
```

`constant` 仍可用。

### 完整训练状态 checkpoint

不只是保存 `.safetensors`，还保存 optimizer/scheduler/RNG/global step 等 accelerator state。

目录默认：

```text
<output_path>/training_checkpoints/step-<global_step>/
```

里面有：

- accelerator state
- `training_state.json`
- `latest` marker 在 `<output_path>/training_checkpoints/latest`

恢复：

```bash
--resume_training_checkpoint latest
```

或：

```bash
--resume_training_checkpoint outputs/.../training_checkpoints/step-20000
```

限制：完整训练状态 checkpoint 不支持 `--enable_model_cpu_offload`。

### 只保留最新 checkpoint

新增：

```bash
--keep_latest_checkpoint_only
```

会清理旧的：

- `step-*.safetensors`
- `epoch-*.safetensors`
- old training checkpoint directories

### 日志增强

新增：

- `--log_steps`
- `train/loss`
- `train/learning_rate`
- TensorBoard video logging
- WandB video logging
- WandB run name 默认用 `output_path` basename

world model eval 会记录：

- scalar metrics
- eval videos

## 当前默认训练设置提醒

### language 版本

`train_robotwin_aloha.sh`：

- `num_processes=4`
- `output_path=./outputs/WanWorldModel_film`
- `dataset_base_path=world_model_data/robotwin_aloha`
- `height=240`
- `width=320`
- `num_frames=25`
- `world_model_stride=12`
- `trainable_models=dit,action_embedder`
- `action_injection_method=film`
- `enable_wandb_log=true`
- `eval_dataset_base_path=world_model_data/robotwin_aloha_testset/custom_aloha_clean`
- `eval_steps=2000`
- `eval_max_samples=2`

### no-language 版本

`train_robotwin_aloha_no_lang.sh`：

- `num_processes=6`
- `output_path=./outputs/WanWorldModel_film_no_language_abs_action_fix`
- `dataset_base_path=world_model_data/robotwin_aloha/train_set,world_model_data/robotwin_aloha/robotwin_aloha_fail`
- `dataset_metadata_path=world_model_data/robotwin_aloha/metadata.json`
- `height=256`
- `width=320`
- `num_frames=25`
- `world_model_stride=6`
- `use_language_condition=false`
- `model_id_with_origin_paths` 不加载 T5 text encoder
- `tokenizer_path=""`
- `trainable_models=dit,action_embedder`
- `action_injection_method=film`
- `eval_dataset_base_path=world_model_data/robotwin_aloha/val_set`
- `eval_steps=200`
- `upload_video_steps=1000`
- `eval_max_samples=32`
- `enable_wandb_log=true`

## 容易忘的坑

- no-language 模式不要用 `action_injection_method=context`。
- no-language 推理脚本默认关闭语言条件；语言 checkpoint 推理时要加 `--enable_language_condition`。
- 周期 eval 需要 LPIPS/FID 权重；缺权重先跑 `download_eval_metric_weights.py`。
- action normalization 依赖 `metadata.json` 里的 `robot_statistics`。
- `dataset_base_path` 在 `WorldModelDataset` 里可以用逗号分隔多个 root。
- `world_model_stride` 决定窗口数量，`stride=6` 比 `stride=25` 产生更多重叠样本。
- `WorldModelDataset` 默认只索引长度至少为 `num_frames` 的 episode。
- `include_failed=false` 会跳过 meta 里标记失败的 episode。
- 完整训练 checkpoint 和 model CPU offload 不兼容。
- `outputs/` 已加入 `.gitignore`。
- `upload_ms.py` 当前源码里有默认 token，使用时最好改成环境变量，避免泄露。
