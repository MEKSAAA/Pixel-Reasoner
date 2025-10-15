benchmark=vstar
export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"
export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/PixelReasoner-WarmStart/checkpoint-1225"
export savefolder=tooleval
# export nvj_path="/path/to/nvidia/nvjitlink/lib" # in case the system cannot fiind the nvjit library
############
export sys=vcot # define the system prompt
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution
export eval_bsz=64 # vllm will processes this many queries 
export tagname=eval_vstar_bestmodel
export testdata="${working_dir}/data/${benchmark}.parquet"
export num_vllm=8
export num_gpus=8
bash ${working_dir}/scripts/eval_vlm_new.sh