#!/bin/bash

script_path="train_vad.py"
# model_name="patchcore"
model_name="padim"
dataset_name="mvtec"
backbone="mobilenet_v2"
ad_layers=("features.4 features.7 features.10")
category="pill"
# category="hazelnut"
dataset_path="/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec"
device_num=2

python $script_path \
  --dataset_name $dataset_name \
  --model_name $model_name \
  --category $category \
  --backbone $backbone \
  --ad_layers ${ad_layers[@]} \
  --dataset_path $dataset_path \
  --device_num $device_num \
  --save_model

