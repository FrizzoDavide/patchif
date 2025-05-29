#!/bin/bash

script_path="train_vad.py"

#NOTE: model parameters
model_name="patchif"
backbone="mobilenet_v2"
ad_model_type="if"
n_estimators=100

#NOTE: dataset parameters
dataset_name="mvtec"
category="pill"
# category="hazelnut"

#NOTE: device parameters
device_num=2

python $script_path \
  --dataset_name $dataset_name \
  --model_name $model_name \
  --ad_model_type $ad_model_type \
  --n_estimators $n_estimators \
  --category $category \
  --backbone $backbone \
  --device_num $device_num \
  --train_model \
  --anomaly_map

