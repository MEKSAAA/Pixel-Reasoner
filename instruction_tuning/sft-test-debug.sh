#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=pr_sft_qwen25vl3b_dbg
export MODEL_DIR=/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct
# export DATA_JSON=/NEW_EDS/miaojw/datasets/pixel_reasoner_sft/release_toolpart_images_only.json
export DATA_JSON=/NEW_EDS/miaojw/projects/CLEAN/rewrite_training_samples.json
export WANDB_DISABLED=true

unset TRANSFORMERS_NO_FLASH_ATTENTION

# 单卡调试模式
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

args=(
  --output_dir "output/${RUN_NAME}"
  --model_name_or_path "${MODEL_DIR}"
  --datasetpath "${DATA_JSON}"
  --eval_strategy no
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 4
  --learning_rate 1e-6
  --num_train_epochs 1
  --bf16
  --torch_dtype bfloat16
  --gradient_checkpointing true
  --attn_implementation flash_attention_2
  --save_only_model true
  --freeze_vision_modules true
  --report_to none
  --dataloader_num_workers 2
  --dataloader_persistent_workers False
)

# 单卡调试 - 直接运行Python脚本
python instruction_tuning/sft_tool.py "${args[@]}"

