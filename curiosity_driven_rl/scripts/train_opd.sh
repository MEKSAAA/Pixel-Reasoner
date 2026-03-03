# reward是perceptual+behavioral
export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export temperature=1.0

export trainver="/NEW_EDS/miaojw/projects/AgentIAD/train300-opd-med-general-sys.json"
export testver="/NEW_EDS/miaojw/projects/AgentIAD/test-btmri-general-sys.json"

export filter=True # filtering zero advantages
export algo=group # default for grpo
export lr=10
export MAX_PIXELS=4014080 # =[max_image_token]x28x28
export sys=anomaly_vcot # system prompt version
export mode=no_eval # [no_eval, eval_only, train]
export policy=/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl/saves/iad-grpo-3b/global_step6_hf
export ref_policy=/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/sft_qwen25vl3b0301_btmri
export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
export save_name="opd-3b-0301"

unset ABLATION_MODE

export rbuffer=128  
export bsz=128
export evalsteps=1
export mbsz=1 
export tp=1 # 
export repeat=1 # data repeat
export nepoch=3 # data epoch
export logp_bsz=1 # must be 1
export maxlen=10000 # generate_max_len
export nsamples=8
export tagname=Train

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RAY_TMPDIR=/NEW_EDS/miaojw/miaojw/ray_tmp
export TMPDIR=/NEW_EDS/miaojw/ray_tmp
export TRITON_CACHE_DIR=/NEW_EDS/miaojw/tmp/triton
export VLLM_ENFORCE_EAGER=1            # 等价于 --enforce-eager
export VLLM_WORKER_DISABLE_CUDA_GRAPH=1  # 部分版本生效；两条都给上

# 删掉同步阻塞（它会把 warmup/capture 放大成“卡住”）
unset CUDA_LAUNCH_BLOCKING

# 单机 8 卡，避免 RDMA/网卡探测带来的 NCCL 阻塞
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="^lo,docker0"
export NCCL_P2P_LEVEL=NVL               # 优先 NVLink
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

export LAOZHANG_API_KEY="sk-CR8zGpArpDAF1IkyE55181D2285f47Ee867244548262977d"
export ZHIPU_API_KEY="03ef23a3717f4890b3458c8434b302a1.KrauNDJb3Jvrr3KV"
export DASHSCOPE_API_KEY="sk-71dbfa5602194275a39f296f745332d7"


unset PYTORCH_CUDA_ALLOC_CONF
bash ./scripts/train_vlm_opd.sh