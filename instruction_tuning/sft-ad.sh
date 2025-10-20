#!/usr/bin/env bash
set -euo pipefail
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
export NCCL_IB_DISABLE=1


export DEBUG_MODE="true"
RUN_NAME=ad_sft_qwen25vl7b_v4         # 保持与八卡相同名称（如需区分可自行加后缀）
export LOG_PATH="logs/debug_log_${RUN_NAME}.txt"

MODEL_DIR=/NEW_EDS/miaojw/models/Qwen2.5-VL-7B-Instruct
DATA_JSON=/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/mvtec_agent_tool_train.json

export CUDA_VISIBLE_DEVICES=5
export WANDB_DISABLED=true      
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_NO_FLASH_ATTENTION

########################################
# 关键等效设置说明：
# 8卡全局batch = 8 * per_device(1) * grad_accum(2) = 16
# 单卡等效：per_device 1，grad_accum 16
########################################

python instruction_tuning/sft_tool.py \
    --output_dir output/${RUN_NAME} \
    --model_name_or_path ${MODEL_DIR} \
    --datasetpath ${DATA_JSON} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --eval_strategy no \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --learning_rate 1e-6 \
    --data_seed 49 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 8 \
    --run_name ${RUN_NAME} \
    --save_strategy epoch \
    --save_steps 100 \
    --save_only_model true \
    --freeze_vision_modules true \
    2>&1 | tee "${LOG_PATH}"
