#!/bin/bash

script_path="patchif_test.py"
model_name="patchcore"
dataset_name="mvtec"
dataset_path="/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec"
category="pill"
backbone="mobilenet_v2"
ad_layers=("features.4 features.7 features.10")
device_num=2

python $script_path \
  --dataset_name $dataset_name \
  --dataset_path $dataset_path \
  --model_name $model_name \
  --category $category \
  --backbone $backbone \
  --ad_layers ${ad_layers[@]} \
  --device_num $device_num
