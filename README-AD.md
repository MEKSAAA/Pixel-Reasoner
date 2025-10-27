## Data Preparation
### Download MMAD
- mkdir MMAD
- cd MMAD
- wget -O ALL_DATA.zip https://huggingface.co/datasets/jiang-cc/MMAD/resolve/refs%2Fpr%2F1/ALL_DATA.zip?download=true
- unzip ALL_DATA.zip
### Download MMAD-SEG
- cd MMAD
- 下载 MMAD-SEG.rar
- unrar MMAD-SEG.rar
### Download train345cross-ROI
- cd MMAD
- 下载 train345cross-ROI.rar
- unrar train345cross-ROI.rar
### 修改json中的全局路径
- train_cross345.json (cross-dataset训练文件)
- test_cross8028.json (cross-dataset测试文件)
- train_multidomain_sft1600.json (multi-domain用于sft训练文件)
- train_multidomain_grpo366.json (multi-domain用于grpo训练文件)
- test_multidomain6400.json (multi-domain测试文件)
- 五个文件都需要把/NEW_EDS/miaojw/projects/MMAD替换成本地路径

## Instruction Tuning
### Installation
- Navigate to the `instruction_tuning` folder
- Follow the detailed setup guide in [installation instructions](instruction_tuning/install/install.md)
### Training Data
#### Multi-domain
- ./ad-dt/train_multidomain_sft1600.json
#### Cross-dataset
- ./ad-dt/train_cross345.json
### Launch Training
- 单卡训练 bash instruction_tuning/sft-ad.sh
- 多卡训练 bash instruction_tuning/sft-multi-ad.sh
- 均需修改 MODEL_DIR，DATA_JSON为本地路径

## Evaluation
### Installation
- Install the openrlhf according to `curiosity_driven_rl/installation.md`.
### Evaluation Data
#### Multi-domain
- ./ad-dt/test_multidomain6400.json
#### Cross-dataset
- ./ad-dt/test_cross8028.json
### Evaluation
- cd curiosity_driven_rl
- 评估训练后模型：设置sys=anomaly_vcot,运行 bash scripts/eval_ad_vcot.sh
- 需要更改以上两个bash中的working_dir，policy，nvj_path，LD_LIBRARY_PATH，testdata
