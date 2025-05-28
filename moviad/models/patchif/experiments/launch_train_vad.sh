#!/bin/bash

script_path="train_vad.py"
# model_name="patchcore"
model_name="padim"
dataset_name="mvtec"
backbone="mobilenet_v2"
category="pill"
# category="hazelnut"
device_num=2

python $script_path \
  --dataset_name $dataset_name \
  --model_name $model_name \
  --category $category \
  --backbone $backbone \
  --device_num $device_num \
  --save_model

