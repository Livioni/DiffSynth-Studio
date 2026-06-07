#!/usr/bin/env bash
set -o pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/models"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false

# Run this once if LPIPS/FID metric weights are missing.
# python examples/wanworldmodel/tools/download_eval_metric_weights.py --download_source modelscope

mkdir -p outputs/WanWorldModel_action_additive

accelerate launch --num_processes 4 examples/wanworldmodel/train.py \
  --dataset_base_path "world_model_data/robotwin_aloha" \
  --data_file_keys "video" \
  --height 240 \
  --width 320 \
  --num_frames 25 \
  --dataset_repeat 10 \
  --dataset_num_workers 0 \
  --log_steps 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --tokenizer_path "models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./outputs/WanWorldModel_action_additive" \
  --trainable_models "dit,action_embedder" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --action_dim 14 \
  --action_injection_method "additive" \
  --action_metadata_path "world_model_data/robotwin_aloha/metadata.json" \
  --eval_dataset_base_path "world_model_data/robotwin_aloha_testset/custom_aloha_clean" \
  --eval_steps 2000 \
  --eval_num_inference_steps 50 \
  --eval_max_samples 2 \
  --eval_num_videos_to_log 4 \
  --eval_metric_batch_size 4 \
  --enable_wandb_log \
  --save_steps 10000 \
  --wandb_project "WanWorldModel" 2>&1 | tee "outputs/WanWorldModel_action_additive/train.log"
