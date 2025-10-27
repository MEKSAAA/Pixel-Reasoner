benchmark=test8028-crossdataset
export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"

#export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/ad_sft_qwen25vl7b_1024/checkpoint-88"
export CUDA_VISIBLE_DEVICES=4,6
export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/ad_sft_qwen25vl3b_v1/checkpoint-176"
export savefolder=eval
export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
############
export sys=anomaly_vcot # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=128 # vllm will processes this many queries 
export tagname=eval_crossdataset_3b176_vcot

export testdata="/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/test8028-crossdataset.json"
export num_vllm=2
export num_gpus=2
export util=0.2 
export actor_ngpus=0  

bash ${working_dir}/scripts/eval_vlm_new.sh