#!/bin/bash

script_path="train_vad.py"

model_name="patchif"
ad_model_type="eif"
n_estimators=100
dataset_name="mvtec"
backbone="mobilenet_v2"
category="pill"
# category="hazelnut"
device_num=2

python $script_path \
  --dataset_name $dataset_name \
  --model_name $model_name \
  --ad_model_type $ad_model_type \
  --n_estimators $n_estimators \
  --category $category \
  --backbone $backbone \
  --device_num $device_num \
  --save_model

