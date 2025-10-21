benchmark=mvtec
export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"

export policy="/NEW_EDS/miaojw/models/Qwen2.5-VL-7B-Instruct"

export savefolder=eval/eval_ad_qwen25vl7b
export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib" # in case the system cannot fiind the nvjit library
export LD_LIBRARY_PATH=/data/miaojw/envs/cur11/lib/python3.10/site-packages/nvidia/nvjitlink/lib
############
export sys=anomaly_notool # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=64 # vllm will processes this many queries 
export tagname=eval_ad_qwen25vl7b

export testdata="/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/mvtec_agent_test_notool.json"
export num_vllm=8
export num_gpus=8

export actor_ngpus=0  

# export FLASH_ATTENTION_FORCE_DISABLE=1

bash ${working_dir}/scripts/eval_vlm_new.sh