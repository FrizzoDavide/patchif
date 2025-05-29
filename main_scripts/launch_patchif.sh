#!/bin/bash

script_path="main_patchif.py"

# model parameters
backbone="mobilenet_v2"
ad_model="eif"
n_estimators=100

# dataset parameters
categories=("pill")
contamination_ratio=0.1

# device
device_num=2

# experiment parameters
seeds=(0 1 2)

python $script_path \
  --train \
  --test \
  --seeds "${seeds[@]}"\
  --backbone_model_name $backbone \
  --ad_model $ad_model \
  --n_estimators $n_estimators \
  --plus \
  --categories "${categories[@]}" \
  --contaminate \
  --contamination_ratio $contamination_ratio \
  --device $device_num



