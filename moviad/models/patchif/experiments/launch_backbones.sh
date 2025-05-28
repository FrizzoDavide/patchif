#!/bin/bash

script_path="backbones.py"
model_name="patchcore"
dataset_name="mvtec"
category="pill"
backbone="mcunet-in3"
device_num=2

python $script_path \
  --dataset_name $dataset_name \
  --model_name $model_name \
  --category $category \
  --backbone $backbone \
  --device_num $device_num
