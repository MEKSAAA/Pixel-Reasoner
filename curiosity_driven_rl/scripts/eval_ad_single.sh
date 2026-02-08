set -x
benchmark=VisA
export CUDA_VISIBLE_DEVICES=4，6

export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"

export policy="/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct"
# export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/ad_sft_qwen25vl3b_v1/checkpoint-176"
# export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/ad_sft_qwen25vl7b_1024/checkpoint-176"

export savefolder=eval
export RAY_IGNORE_UNHANDLED_ERRORS=1

export testdata="/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/VisA_test_tool.json"
export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib" # in case the system cannot fiind the nvjit library
export RAY_TMPDIR=/data/miaojw/ray_tmp
# export sys=vcot # define the system prompt
export sys=anomaly_vcot
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution

export eval_bsz=32 # vllm will processes this many queries 
export tagname=eval_VisA_3b176_vcot_1026
export num_vllm=2
export num_gpus=2
# export tp=1                # tensor parallel
export util=0.7            # vLLM 显存利用率，卡小就调低些 (0.6~0.7)
export maxlen=8192         # 你脚本默认 8192；不够就再调
export actor_ngpus=0       # eval_only 下本就不用 actor，多卡才用

bash ${working_dir}/scripts/eval_vlm_new.sh