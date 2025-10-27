set -euo pipefail
export DEBUG_MODE="true"
RUN_NAME=ad_sft_qwen25vl7b        # 保持与八卡相同名称（如需区分可自行加后缀）
export LOG_PATH="logs/debug_log_${RUN_NAME}.txt"

MODEL_DIR=/NEW_EDS/miaojw/models/Qwen2.5-VL-7B-Instruct
DATA_JSON=/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/train_cross345.json


export CUDA_VISIBLE_DEVICES=3,4,6,7
export WANDB_DISABLED=true      
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_NO_FLASH_ATTENTION

# NCCL 配置：解决超时和通信问题
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_TIMEOUT=180
# export NCCL_BLOCKING_WAIT=1

python -m torch.distributed.run --nproc_per_node=4 \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="127.0.0.1" \
  --master_port="29510" \
  instruction_tuning/sft_tool.py \
  --deepspeed instruction_tuning/local_scripts/zero2.json \
  --output_dir output/${RUN_NAME} \
  --model_name_or_path ${MODEL_DIR} \
  --datasetpath ${DATA_JSON} \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --eval_strategy no \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 5 \
  --bf16 \
  --torch_dtype bfloat16 \
  --learning_rate 2e-5 \
  --data_seed 49 \
  --report_to none \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --num_train_epochs 10 \
  --run_name ${RUN_NAME} \
  --save_strategy epoch \
  --save_steps 100 \
  --save_only_model true \
  --freeze_vision_modules true \
  2>&1 | tee "${LOG_PATH}"