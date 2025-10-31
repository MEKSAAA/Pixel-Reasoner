export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export temperature=1.0
export trainver="/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/train366grpo-md-new.json"
export testver="/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/test6400md.json"
export filter=True # filtering zero advantages
export algo=group # default for grpo
export lr=10
export MAX_PIXELS=4014080 # =[max_image_token]x28x28
export sys=anomaly_vcot # system prompt version
export mode=no_eval # [no_eval, eval_only, train]
export policy=/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/md_sft_qwen25vl3b1029/checkpoint-600
export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
export save_name="grpo-3b-600"


export rbuffer=512 # replay buffer size
export bsz=256 # global train batch size

export evalsteps=1
export mbsz=2 

export tp=1 # vllm tp, 1 for 7B
export repeat=1 # data repeat
export nepoch=3 # data epoch
export logp_bsz=1 # must be 1
export maxlen=10000 # generate_max_len
export nsamples=8
export tagname=Train

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

export WANDB_MODE=disabled
unset PYTORCH_CUDA_ALLOC_CONF
bash ./scripts/train_vlm_single.sh