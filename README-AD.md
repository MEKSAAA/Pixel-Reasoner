## Instruction Tuning
### Installation
- Navigate to the `instruction_tuning` folder
- Follow the detailed setup guide in [installation instructions](instruction_tuning/install/install.md)
### Training Data
- ./ad-dt/mvtec_agent_tool_train.json
- 需要下载mvtec数据：wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
- 将mvtec_agent_tool_train.json文件里的/NEW_EDS/miaojw/datasets/mvtec_dataset全部替换成本地路径
- 下载mvtec_seg_white_bg_precise_train_sampled.rar并解压到本地，将mvtec_agent_tool_train.json文件里的/NEW_EDS/miaojw/projects/Accurate-WinCLIP-pytorch/results/mvtec_seg_white_bg_precise_train_sampled替换成本地文件夹路径
### Launch Training
- 单卡训练 bash instruction_tuning/sft-ad.sh
- 多卡训练 bash instruction_tuning/sft-multi-ad.sh
- 均需修改 MODEL_DIR，DATA_JSON为本地路径

## Evaluation
### Installation
- Install the openrlhf according to `curiosity_driven_rl/installation.md`.
### Training Data
- ./ad-dt/mvtec_agent_test.json
- 将mvtec_agent_test.json文件里的/NEW_EDS/miaojw/datasets/mvtec_dataset全部替换成下载的mvtec数据集的本地路径
### Evaluation
- cd curiosity_driven_rl
- 评估未训练模型：设置sys=anomaly_notool,运行 bash scripts/eval_ad_notool.sh
- 评估训练后模型：设置sys=anomaly_vcot,单卡运行 bash scripts/eval_ad_vcot.sh
- 需要更改以上两个bash中的working_dir，policy，nvj_path，LD_LIBRARY_PATH，testdata
