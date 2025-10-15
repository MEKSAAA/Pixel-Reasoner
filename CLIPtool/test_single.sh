#!/bin/bash
# 单样本测试脚本示例

python single_image_test.py \
  --image /NEW_EDS/miaojw/datasets/mvtec_dataset/capsule/test/squeeze/015.png \
  --class_name capsule \
  --output ./output


python single_image_test.py \
  --image /NEW_EDS/miaojw/datasets/mvtec_dataset/bottle/train/good/000.png \
  --class_name bottle \
  --output ./output