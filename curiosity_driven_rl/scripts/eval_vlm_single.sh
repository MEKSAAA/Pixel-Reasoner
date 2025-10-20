set -x
export CUDA_VISIBLE_DEVICES=7
benchmark=vstar
export working_dir="/NEW_EDS/miaojw/projects/Pixel-Reasoner/curiosity_driven_rl"
# export policy="/NEW_EDS/miaojw/projects/Pixel-Reasoner/PixelReasoner-WarmStart/checkpoint-1225"
export policy="/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct"
export savefolder=notooleval_qwen25vl3b
# export savefolder=notoolval
export testdata="${working_dir}/data/${benchmark}.parquet"
# export testdata="/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/mvtec_agent_test.json"
# export nvj_path="/NEW_EDS/miaojw/miniconda3_new/envs/curiosity9/lib/python3.10/site-packages/nvidia/nvjitlink/lib" # in case the system cannot fiind the nvjit library
export LD_LIBRARY_PATH=/data/miaojw/envs/cur11/lib/python3.10/site-packages/nvidia/nvjitlink/lib
export RAY_TMPDIR=/data/miaojw/ray_tmp
# export sys=vcot # define the system prompt
export sys=notool
export MIN_PIXELS=401408
export MAX_PIXELS=4014080 # define the image resolution

export eval_bsz=1 # vllm will processes this many queries 
export tagname=eval_vstar_bestmodel
export num_vllm=1
export num_gpus=1
export tp=1                # tensor parallel
export util=0.3            # vLLM 显存利用率，卡小就调低些 (0.6~0.7)
export nsamples=1
export maxlen=1024         # 你脚本默认 8192；不够就再调
export actor_ngpus=0       # eval_only 下本就不用 actor，多卡才用

bash ${working_dir}/scripts/eval_vlm_new.sh