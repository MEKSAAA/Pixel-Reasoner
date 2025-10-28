## Data Preparation
### Download MMAD
- mkdir MMAD
- cd MMAD
- wget -O ALL_DATA.zip https://huggingface.co/datasets/jiang-cc/MMAD/resolve/refs%2Fpr%2F1/ALL_DATA.zip?download=true
- unzip ALL_DATA.zip
### Download MMAD-SEG-NEW
- cd MMAD
- 下载 MMAD-SEG-NEW.rar
- unrar MMAD-SEG-NEW.rar
### 修改json中的全局路径
- train338cross.json (cross-dataset训练文件)
- test8028cross.json (cross-dataset测试文件)
- train1600sft-md.json (multi-domain用于sft训练文件)
- train366grpo-md.json (multi-domain用于grpo训练文件)
- test6400md.json (multi-domain测试文件)
- 五个文件都需要把/NEW_EDS/miaojw/projects/MMAD替换成本地路径

## Instruction Tuning
### Installation
- Navigate to the `instruction_tuning` folder
- Follow the detailed setup guide in [installation instructions](instruction_tuning/install/install.md)

### Launch Training
- 均需修改 MODEL_DIR，DATA_JSON为本地路径
#### Multi-domain
- bash instruction_tuning/sft-ad-multi.sh
#### Cross-dataset
- bash instruction_tuning/sft-ad-cross.sh



## Evaluation
### Installation
- Install the openrlhf according to `curiosity_driven_rl/installation.md`.
### Evaluation Data
#### Multi-domain
- ./ad-dt/test6400md.json
#### Cross-dataset
- ./ad-dt/test8028cross.json
### Evaluation
- cd curiosity_driven_rl
- 评估训练后模型：设置sys=anomaly_vcot,运行 bash scripts/eval_ad_vcot.sh
- 需要更改以上两个bash中的working_dir，policy，nvj_path，LD_LIBRARY_PATH，testdata
