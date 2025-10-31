benchmark=3b
export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"

export CUDA_VISIBLE_DEVICES=0
# export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/cross_sft_qwen25vl3b1029/checkpoint-220"
export policy="/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct"
export savefolder=eval
export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
############
export sys=anomaly_notool # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=128 # vllm will processes this many queries 
export tagname=eval_3b_notool_1030

export testdata="/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/test8366baseline.json"
export num_vllm=1
export num_gpus=1
export util=0.7
export actor_ngpus=0  

bash ${working_dir}/scripts/eval_vlm_new.sh