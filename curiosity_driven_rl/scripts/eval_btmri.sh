benchmark=3b
export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/3b-sft-med"
export savefolder=eval
export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
############
export sys=anomaly_vcot_med # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=128 # vllm will processes this many queries 

export tagname=eval_3bbrmrisft_btmri

export testdata="/NEW_EDS/miaojw/projects/agentiad-03/binary-brain/test-btmri.json"
export num_vllm=8
export num_gpus=8
export util=0.7
export actor_ngpus=0  

export LAOZHANG_API_KEY="sk-CR8zGpArpDAF1IkyE55181D2285f47Ee867244548262977d"
export ZHIPU_API_KEY="03ef23a3717f4890b3458c8434b302a1.KrauNDJb3Jvrr3KV"

bash ${working_dir}/scripts/eval_vlm_new.sh