#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/models"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false

# 缺少 LPIPS/FID 权重时先运行一次：
# python examples/wanworldmodel/tools/download_eval_metric_weights.py --download_source modelscope

# =========================
# accelerate
# =========================
num_processes=6   # 启动的训练进程数，通常等于 GPU 数。

# =========================
# path
# =========================
output_path="./outputs/WanWorldModel_film_no_language"  # 模型、日志、评估视频和断点输出目录。
log_path="${output_path}/train.log"  # shell stdout/stderr 保存路径。

# =========================
# dataset
# =========================
dataset_base_path="world_model_data/robotwin_aloha/train_set,world_model_data/robotwin_aloha/robotwin_aloha_fail"  # 训练数据根目录，多个 root 用逗号分隔。
dataset_metadata_path="world_model_data/robotwin_aloha/metadata.json"  # unified 数据集 metadata 路径；world_model 模式通常留空。
dataset_type="world_model"  # 数据集类型：auto、unified 或 world_model。
dataset_repeat=5  # 每个 epoch 内虚拟重复数据集的次数，用于增加训练步数。
dataset_num_workers=0  # 训练 DataLoader worker 数。
data_file_keys="video"  # unified 数据集读取的数据字段，world_model 模式基本不使用。

# =========================
# world model dataset
# =========================
world_model_tasks=""  # 只加载指定 task，逗号分隔；留空表示加载全部 task。
world_model_cameras=""  # 要加载的 camera 列表，逗号分隔；留空只加载 video camera。
world_model_video_camera="head_camera"  # 作为训练视频输入的 camera。
world_model_stride="12"  # 固定长度窗口滑动步长；留空默认等于 num_frames。
world_model_include_depth=false  # 是否额外加载 depth。
world_model_include_camera_params=false  # 是否额外加载相机内外参。
world_model_include_failed=true  # 是否包含失败 episode。

# =========================
# video size / sampling
# =========================
height=240  # 训练视频帧高度；留空启用动态分辨率。
width=320  # 训练视频帧宽度；留空启用动态分辨率。
max_pixels=1048576  # 动态分辨率时每帧最大像素数。
num_frames=25  # 每个训练视频 window 的帧数。
video_random_start=true  # unified 视频是否随机起始帧；world_model 固定 window 不依赖它。

# =========================
# model loading
# =========================
model_paths=""  # 本地模型路径 JSON；和 model_id_with_origin_paths 二选一或组合。
model_id_with_origin_paths="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"  # 模型仓库和权重匹配规则；no-language 训练不加载 T5 text encoder。
tokenizer_path=""  # no-language 训练不需要 tokenizer。
extra_inputs="input_image"  # 额外传给模型的输入字段，逗号分隔。
fp8_models=""  # 使用 FP8 加载的模型名或路径，逗号分隔。
offload_models=""  # split training 中需要 offload 的模型，逗号分隔。
resume_from_checkpoint=""  # 只加载模型权重的 checkpoint 文件；不恢复 optimizer/step。
initialize_model_on_cpu=false  # 是否先在 CPU 初始化模型。

# =========================
# language conditioning
# =========================
use_language_condition=false  # false 时传入 --disable_language_condition，不加载 text encoder/tokenizer，也跳过 DiT context attention。
text_context_length=512  # 仅 use_language_condition=true 时使用。

# =========================
# template model
# =========================
template_model_id_or_path=""  # template model 的 ID 或本地路径。
enable_lora_hot_loading=false  # 是否启用 LoRA 热加载，仅部分 image-to-lora 模型可用。

# =========================
# training
# =========================
task="sft"  # 训练任务类型，如 sft、sft:train、direct_distill。
trainable_models="dit,action_embedder"  # 参与训练的模块名，逗号分隔。
learning_rate=1e-4  # 学习率。
lr_scheduler="warmup_cosine"  # 学习率调度：constant 或 warmup_cosine。
lr_warmup_steps=1000  # 线性 warmup step 数。
lr_cosine_min_ratio=0.1  # cosine decay 最终学习率比例。
weight_decay=0.01  # AdamW 权重衰减。
num_epochs=5  # 训练 epoch 数。
customized_optimizer=""  # 自定义 optimizer 类路径，如 bitsandbytes.optim.Adam8bit。
find_unused_parameters=true  # DDP 是否查找未使用参数。
log_steps=10  # 每隔多少 step 记录训练 loss/lr；<=0 表示每 step。

# =========================
# gradient
# =========================
gradient_accumulation_steps=1  # 梯度累积步数。
use_gradient_checkpointing=true  # 是否启用梯度检查点以省显存。
use_gradient_checkpointing_offload=false  # 是否把梯度检查点相关数据 offload 到 CPU。
max_timestep_boundary=1.0  # 采样 timestep 上边界比例。
min_timestep_boundary=0.0  # 采样 timestep 下边界比例。

# =========================
# action conditioning
# =========================
action_dim=14  # 机器人 action 向量维度；设空会禁用 action 条件。
action_embedder_hidden_dim=""  # action embedder 隐藏层维度；留空使用 DiT hidden dim。
action_injection_method="film"  # action 注入方式：none/context/additive/cross_attention/adaln/film。
action_metadata_path="world_model_data/robotwin_aloha/metadata.json"  # action 归一化统计 metadata 路径。
action_metadata_key="robot_statistics"  # metadata 中 action 统计信息的 key。
action_normalization_eps=1e-6  # action 归一化 std 的最小值。
action_normalization_mode="standard"  # action 归一化模式：standard=(action-mean)/std，scale_only=action/std。

# =========================
# eval
# =========================
eval_dataset_base_path="world_model_data/robotwin_aloha/val_set"  # 评估数据根目录；设空跳过 eval dataset。
eval_steps=2000  # 每隔多少 step 做一次 eval；<=0 禁用周期 eval。
eval_num_inference_steps=50  # eval 推理的 diffusion steps。
eval_max_samples=4  # eval 最多评估多少个 window。
eval_dataset_num_workers=0  # eval DataLoader worker 数。
eval_num_videos_to_log=4  # 记录到日志平台的 eval 样本视频数量。
eval_video_fps=4  # eval 视频保存和日志展示 FPS。
eval_metric_batch_size=4  # LPIPS/FID 计算 batch size。

# =========================
# output / checkpoints
# =========================
remove_prefix_in_ckpt="pipe.dit."  # 保存训练权重时移除的 state_dict key 前缀。
save_steps=20000  # 每隔多少 step 保存权重和完整训练断点；留空则按 epoch 保存权重。
save_training_checkpoint=true  # 是否在 save_steps 时保存 optimizer/scheduler/RNG/global step 等完整断点。
resume_training_checkpoint=""  # 从完整训练断点恢复；可填 latest 或 step 目录。
training_checkpoint_dir=""  # 完整训练断点目录；留空默认 output_path/training_checkpoints。

# =========================
# offload training
# =========================
enable_model_cpu_offload=false  # 是否启用训练时模型层级 CPU offload。
enable_optimizer_cpu_offload=false  # 开启 model offload 时，是否把 optimizer 也放 CPU。
cpu_offload_split_threshold=""  # 大模块递归拆分阈值，单位 MB；留空表示叶子模块粒度。

# =========================
# logging
# =========================
enable_tensorboard_log=true  # 是否启用 TensorBoard。
enable_swanlab_log=false  # 是否启用 SwanLab。
swanlab_project="DiffSynth-Studio"  # SwanLab project 名。
enable_wandb_log=true  # 是否启用 Weights & Biases。
wandb_project="WanWorldModel"  # WandB project 名。

add_arg() {
  train_args+=("$1" "$2")
}

add_optional_arg() {
  if [[ -n "$2" ]]; then
    train_args+=("$1" "$2")
  fi
}

add_flag() {
  if [[ "$2" == true ]]; then
    train_args+=("$1")
  fi
}

add_inverse_flag() {
  if [[ "$2" != true ]]; then
    train_args+=("$1")
  fi
}

mkdir -p "${output_path}"

accelerate_args=(
  --num_processes "${num_processes}"
)

train_args=(
  examples/wanworldmodel/train.py
)

# Dataset
add_arg --dataset_base_path "${dataset_base_path}"
add_optional_arg --dataset_metadata_path "${dataset_metadata_path}"
add_arg --dataset_type "${dataset_type}"
add_arg --dataset_repeat "${dataset_repeat}"
add_arg --dataset_num_workers "${dataset_num_workers}"
add_arg --data_file_keys "${data_file_keys}"

# World Model Dataset
add_optional_arg --world_model_tasks "${world_model_tasks}"
add_optional_arg --world_model_cameras "${world_model_cameras}"
add_arg --world_model_video_camera "${world_model_video_camera}"
add_optional_arg --world_model_stride "${world_model_stride}"
add_flag --world_model_include_depth "${world_model_include_depth}"
add_flag --world_model_include_camera_params "${world_model_include_camera_params}"
add_flag --world_model_include_failed "${world_model_include_failed}"

# Video Size / Sampling
add_optional_arg --height "${height}"
add_optional_arg --width "${width}"
add_arg --max_pixels "${max_pixels}"
add_arg --num_frames "${num_frames}"
add_inverse_flag --disable_video_random_start "${video_random_start}"

# Model Loading
add_optional_arg --model_paths "${model_paths}"
add_optional_arg --model_id_with_origin_paths "${model_id_with_origin_paths}"
add_optional_arg --tokenizer_path "${tokenizer_path}"
add_arg --extra_inputs "${extra_inputs}"
add_optional_arg --fp8_models "${fp8_models}"
add_optional_arg --offload_models "${offload_models}"
add_optional_arg --resume_from_checkpoint "${resume_from_checkpoint}"
add_flag --initialize_model_on_cpu "${initialize_model_on_cpu}"

# Language Conditioning
add_inverse_flag --disable_language_condition "${use_language_condition}"
if [[ "${use_language_condition}" == true ]]; then
  add_arg --text_context_length "${text_context_length}"
fi

# Template Model
add_optional_arg --template_model_id_or_path "${template_model_id_or_path}"
add_flag --enable_lora_hot_loading "${enable_lora_hot_loading}"

# Training
add_arg --task "${task}"
add_optional_arg --trainable_models "${trainable_models}"
add_arg --learning_rate "${learning_rate}"
add_arg --lr_scheduler "${lr_scheduler}"
add_arg --lr_warmup_steps "${lr_warmup_steps}"
add_arg --lr_cosine_min_ratio "${lr_cosine_min_ratio}"
add_arg --weight_decay "${weight_decay}"
add_arg --num_epochs "${num_epochs}"
add_optional_arg --customized_optimizer "${customized_optimizer}"
add_flag --find_unused_parameters "${find_unused_parameters}"
add_arg --log_steps "${log_steps}"

# Gradient
add_arg --gradient_accumulation_steps "${gradient_accumulation_steps}"
add_flag --use_gradient_checkpointing "${use_gradient_checkpointing}"
add_flag --use_gradient_checkpointing_offload "${use_gradient_checkpointing_offload}"
add_arg --max_timestep_boundary "${max_timestep_boundary}"
add_arg --min_timestep_boundary "${min_timestep_boundary}"

# Action Conditioning
add_optional_arg --action_dim "${action_dim}"
add_optional_arg --action_embedder_hidden_dim "${action_embedder_hidden_dim}"
add_arg --action_injection_method "${action_injection_method}"
add_optional_arg --action_metadata_path "${action_metadata_path}"
add_arg --action_metadata_key "${action_metadata_key}"
add_arg --action_normalization_eps "${action_normalization_eps}"
add_arg --action_normalization_mode "${action_normalization_mode}"

# Eval
add_arg --eval_dataset_base_path "${eval_dataset_base_path}"
add_arg --eval_steps "${eval_steps}"
add_arg --eval_num_inference_steps "${eval_num_inference_steps}"
add_arg --eval_max_samples "${eval_max_samples}"
add_arg --eval_dataset_num_workers "${eval_dataset_num_workers}"
add_arg --eval_num_videos_to_log "${eval_num_videos_to_log}"
add_arg --eval_video_fps "${eval_video_fps}"
add_arg --eval_metric_batch_size "${eval_metric_batch_size}"

# Output / Checkpoints
add_arg --output_path "${output_path}"
add_arg --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}"
add_optional_arg --save_steps "${save_steps}"
add_inverse_flag --disable_training_checkpoint "${save_training_checkpoint}"
add_optional_arg --resume_training_checkpoint "${resume_training_checkpoint}"
add_optional_arg --training_checkpoint_dir "${training_checkpoint_dir}"

# Offload Training
add_flag --enable_model_cpu_offload "${enable_model_cpu_offload}"
add_flag --enable_optimizer_cpu_offload "${enable_optimizer_cpu_offload}"
add_optional_arg --cpu_offload_split_threshold "${cpu_offload_split_threshold}"

# Logging
if [[ "${enable_tensorboard_log}" == true ]]; then
  add_flag --enable_tensorboard_log true
else
  add_flag --disable_tensorboard_log true
fi
add_flag --enable_swanlab_log "${enable_swanlab_log}"
add_arg --swanlab_project "${swanlab_project}"
add_flag --enable_wandb_log "${enable_wandb_log}"
add_arg --wandb_project "${wandb_project}"

accelerate launch "${accelerate_args[@]}" "${train_args[@]}" 2>&1 | tee "${log_path}"
