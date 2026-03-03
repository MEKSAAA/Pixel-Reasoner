benchmark=3b
export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl/saves/opd-3b-0301/global_step5_hf"
# export policy="/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct"
export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/sft-iad-notype-qa"
export savefolder=eval
export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
############
export sys=anomaly_vcot_notype_qa # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=128 # vllm will processes this many queries 
export tagname=eval_iad_notype_qa

export testdata="/NEW_EDS/miaojw/projects/AgentIAD/test_notype_iad_qa_sft.json"
export num_vllm=8
export num_gpus=8
export util=0.7
export actor_ngpus=0  

export LAOZHANG_API_KEY="sk-CR8zGpArpDAF1IkyE55181D2285f47Ee867244548262977d"
export ZHIPU_API_KEY="03ef23a3717f4890b3458c8434b302a1.KrauNDJb3Jvrr3KV"
export DASHSCOPE_API_KEY="sk-71dbfa5602194275a39f296f745332d7"

bash ${working_dir}/scripts/eval_vlm_new.sh