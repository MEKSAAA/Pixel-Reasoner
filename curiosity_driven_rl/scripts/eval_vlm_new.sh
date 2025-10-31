set -x

RAY_MASTER_NODE_ADDRESS="0.0.0.0"
RAY_MASTER_NODE_PORT=$(shuf -n 1 -i 30000-65535)
WORLD_SIZE=1
NODE_RANK=0
GPUS_PER_NODE=8

MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
export WANDB_MODE="offline"
export WANDB_API_KEY="null"
# export NCCL_SOCKET_IFNAME=ens2f5
# export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export HOST_IP=0.0.0.0
export VLLM_HOST_IP=0.0.0.0

working_dir=${working_dir:-"/NEW_EDS/miaojw/projects/Pixel-Reasoner"}
cd $working_dir
export HF_ENDPOINT=https://hf-mirror.com
nnode=$WORLD_SIZE
testdata=${testdata:-"none"}
num_vllm=${num_vllm:-"4"}
num_gpus=${num_gpus:-"4"}
tp=${tp:-"1"}
actor_ngpus=${actor_ngpus:-"1"}
nsamples=${nsamples:-"1"}
temperature=${temperature:-"0.6"}
factor=${factor:-"1"}
eval_bsz=${eval_bsz:-"8"}
MIN_PIXELS=${MIN_PIXELS:-"$(( 512 * 28 * 28))"}
MAX_PIXELS=${MAX_PIXELS:-"$(( 5120 * 28 * 28))"}
tag=${tagname} # -n${nsamples}
rule_reward=${rule:-"none"}
sys=${sys:-"default"}
lr=${lr:-"10"}
algo=${algo:-"group"}
dataver=${dataver:-"none"}
util=${util:-"0.7"}

numref=0

maxlen=${maxlen:-"8192"}
policy=${policy:-"/path/to/policy"}
save_name="${tag}" # rbsize 1024->256
DATASET=${testdata}
MODEL_CPK_NAME=${save_name}
PRETRAIN_MODEL=${policy}
savefolder=${savefolder:-"eval_results"}
SAVE_PATH=$working_dir/${savefolder}/$save_name
mkdir -p "${SAVE_PATH}"



post_args=""
if [ $nnode -gt 1 ]; then
    if [ $nnode -gt 3 ]; then
        post_args=(--ref_num_nodes 0
            --ref_num_gpus_per_node 8 
            --actor_num_nodes 16
            --actor_num_gpus_per_node 1 
            --vllm_num_engines 16 
            --vllm_tensor_parallel_size 1
            --micro_train_batch_size 4 
            --train_batch_size 256 
            --micro_rollout_batch_size 1
            --rollout_batch_size 1024
        )
    else
        post_args=(--ref_num_nodes 0
            --ref_num_gpus_per_node 8 
            --actor_num_nodes 8
            --actor_num_gpus_per_node 1 
            --vllm_num_engines 8 
            --vllm_tensor_parallel_size 1
            --micro_train_batch_size 4 
            --train_batch_size 256 
            --micro_rollout_batch_size 8
            --rollout_batch_size 1024
    )
    fi
else 
    post_args=(--ref_num_nodes 0
            --ref_num_gpus_per_node 8 
            --actor_num_nodes 0
            --actor_num_gpus_per_node ${actor_ngpus} 
            --vllm_num_engines ${num_vllm}
            --vllm_tensor_parallel_size ${tp}
            --adam_offload
            --micro_train_batch_size 4 
            --train_batch_size 256 
            --micro_rollout_batch_size 1
            --rollout_batch_size 1024
    )
fi

LD_LIBRARY_PATH_VALUE=$nvj_path:$LD_LIBRARY_PATH

export RAY_TMPDIR=/NEW_EDS/miaojw/ray_tmp
export TMPDIR=/NEW_EDS/miaojw/ray_tmp
export RAY_USAGE_STATS_ENABLED=0

# RUNTIME_ENV_JSON="{\"pip\": [\"Qwen-Agent\"], \"env_vars\": {\"MAX_PIXELS\": \"$MAX_PIXELS\", \"MIN_PIXELS\": \"$MIN_PIXELS\", \"RAY_DEBUG\": \"legacy\", \"LD_LIBRARY_PATH\": \"$LD_LIBRARY_PATH_VALUE\"}}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"MAX_PIXELS\": \"$MAX_PIXELS\",
    \"MIN_PIXELS\": \"$MIN_PIXELS\",
    \"RAY_DEBUG\": \"legacy\",
    \"LD_LIBRARY_PATH\": \"$CONDA_PREFIX/lib:${NVJITLINK_DIR}:/usr/local/cuda/lib64\",
    \"LD_PRELOAD\": \"$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1\",
    \"TRANSFORMERS_NO_TORCHVISION\": \"1\"
  }
}"



ray_output=$(ray start --head --num-gpus ${num_gpus} --temp-dir ${RAY_TMPDIR} --port=8399)


ray status
ray job submit --address="http://127.0.0.1:8265" \
--runtime-env-json="$RUNTIME_ENV_JSON" \
-- python3 -m openrlhf.cli.eval_ray \
--vllm_enable_sleep \
--vllm_gpu_memory_utilization ${util} \
--vllm_sync_backend gloo \
--enable_prefix_caching \
--pretrain $PRETRAIN_MODEL \
--save_path $SAVE_PATH \
--n_samples_per_prompt ${nsamples} \
--max_epochs 1 \
--num_episodes 3 \
--prompt_max_len 10000 \
--max_samples 100000 \
--generate_max_len ${maxlen} \
--advantage_estimator ${algo} \
--zero_stage 3 \
--bf16 \
--actor_learning_rate ${lr}e-7 \
--rule_reward ${rule_reward} \
--temperature 1.0 \
--top_p 0.95 \
--init_kl_coef 0.0 \
--aux_loss_coef 0.05 \
--entropy_loss_coef 0.0 \
--prompt_data $DATASET \
--input_key question \
--apply_chat_template \
--normalize_reward \
--data_version ${dataver} \
--flash_attn \
--gradient_checkpointing \
--ckpt_path $SAVE_PATH \
--save_steps 5 \
--max_ckpt_num 5 \
--save_hf_ckpt \
--disable_ds_ckpt \
--use_wandb $WANDB_API_KEY \
--wandb_run_name $save_name \
--system_prompt ${sys} \
--use_kl_estimator_k3 \
--wandb_project vlm-rl-eval \
--buffer_norm 0 \
--train_vlm \
--training_mode eval_only \
--eval_batch_size_pergpu ${eval_bsz} \
--eval_data ${testdata} \
${post_args[@]} 