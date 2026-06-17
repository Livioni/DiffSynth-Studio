#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/models"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false

num_processes=4
num_machines=1
mixed_precision="bf16"
dynamo_backend="no"

output_path="./outputs/WanWorldActionModel_robotwin_aloha"
log_path="${output_path}/train.log"

dataset_base_path="world_model_data/robotwin_aloha"
dataset_repeat=5
dataset_num_workers=0
max_data_items=""

world_model_tasks=""
world_model_cameras=""
world_model_video_camera="head_camera,left_camera,right_camera"
world_model_stride="12"
world_model_include_failed=true

height=192
width=256
max_pixels=1048576
num_frames=9
action_horizon=33
action_dim=20
action_max_seq_len=60
statistics_path=""
quat_order="xyzw"

model_paths=""
model_id_with_origin_paths=""
tokenizer_path=""
world_action_checkpoint_path=""
fp8_models=""
offload_models=""
resume_from_checkpoint=""
initialize_model_on_cpu=false

task="sft"
trainable_models="dit"
learning_rate=1e-5
lr_scheduler="warmup_cosine"
lr_warmup_steps=1000
lr_cosine_min_ratio=0.1
weight_decay=0.01
num_epochs=5
customized_optimizer=""
find_unused_parameters=true
log_steps=10
lambda_video=1.0
lambda_action=1.0

gradient_accumulation_steps=2
use_gradient_checkpointing=true
use_gradient_checkpointing_offload=false
max_timestep_boundary=1.0
min_timestep_boundary=0.0

remove_prefix_in_ckpt="pipe.dit."
save_steps=10000
keep_latest_checkpoint_only=true
save_training_checkpoint=true
resume_training_checkpoint=""
training_checkpoint_dir=""

enable_model_cpu_offload=false
enable_optimizer_cpu_offload=false
cpu_offload_split_threshold=""

enable_tensorboard_log=true
enable_swanlab_log=false
swanlab_project="DiffSynth-Studio"
enable_wandb_log=true
wandb_project="WanWorldActionModel"

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
  --num_machines "${num_machines}"
  --mixed_precision "${mixed_precision}"
  --dynamo_backend "${dynamo_backend}"
)

train_args=(
  examples/wanworldactionmodel/train.py
)

add_arg --dataset_base_path "${dataset_base_path}"
add_arg --dataset_repeat "${dataset_repeat}"
add_arg --dataset_num_workers "${dataset_num_workers}"
add_optional_arg --max_data_items "${max_data_items}"

add_optional_arg --world_model_tasks "${world_model_tasks}"
add_optional_arg --world_model_cameras "${world_model_cameras}"
add_arg --world_model_video_camera "${world_model_video_camera}"
add_optional_arg --world_model_stride "${world_model_stride}"
add_flag --world_model_include_failed "${world_model_include_failed}"

add_arg --height "${height}"
add_arg --width "${width}"
add_arg --max_pixels "${max_pixels}"
add_arg --num_frames "${num_frames}"
add_arg --action_horizon "${action_horizon}"
add_arg --action_dim "${action_dim}"
add_arg --action_max_seq_len "${action_max_seq_len}"
add_optional_arg --statistics_path "${statistics_path}"
add_arg --quat_order "${quat_order}"

add_optional_arg --model_paths "${model_paths}"
add_optional_arg --model_id_with_origin_paths "${model_id_with_origin_paths}"
add_optional_arg --tokenizer_path "${tokenizer_path}"
add_optional_arg --world_action_checkpoint_path "${world_action_checkpoint_path}"
add_optional_arg --fp8_models "${fp8_models}"
add_optional_arg --offload_models "${offload_models}"
add_optional_arg --resume_from_checkpoint "${resume_from_checkpoint}"
add_flag --initialize_model_on_cpu "${initialize_model_on_cpu}"

add_arg --task "${task}"
add_arg --trainable_models "${trainable_models}"
add_arg --learning_rate "${learning_rate}"
add_arg --lr_scheduler "${lr_scheduler}"
add_arg --lr_warmup_steps "${lr_warmup_steps}"
add_arg --lr_cosine_min_ratio "${lr_cosine_min_ratio}"
add_arg --weight_decay "${weight_decay}"
add_arg --num_epochs "${num_epochs}"
add_optional_arg --customized_optimizer "${customized_optimizer}"
add_flag --find_unused_parameters "${find_unused_parameters}"
add_arg --log_steps "${log_steps}"
add_arg --lambda_video "${lambda_video}"
add_arg --lambda_action "${lambda_action}"

add_arg --gradient_accumulation_steps "${gradient_accumulation_steps}"
add_flag --use_gradient_checkpointing "${use_gradient_checkpointing}"
add_flag --use_gradient_checkpointing_offload "${use_gradient_checkpointing_offload}"
add_arg --max_timestep_boundary "${max_timestep_boundary}"
add_arg --min_timestep_boundary "${min_timestep_boundary}"

add_arg --output_path "${output_path}"
add_arg --remove_prefix_in_ckpt "${remove_prefix_in_ckpt}"
add_optional_arg --save_steps "${save_steps}"
add_flag --keep_latest_checkpoint_only "${keep_latest_checkpoint_only}"
add_inverse_flag --disable_training_checkpoint "${save_training_checkpoint}"
add_optional_arg --resume_training_checkpoint "${resume_training_checkpoint}"
add_optional_arg --training_checkpoint_dir "${training_checkpoint_dir}"

add_flag --enable_model_cpu_offload "${enable_model_cpu_offload}"
add_flag --enable_optimizer_cpu_offload "${enable_optimizer_cpu_offload}"
add_optional_arg --cpu_offload_split_threshold "${cpu_offload_split_threshold}"

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
