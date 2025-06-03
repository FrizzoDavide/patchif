#!/bin/bash

script_path="main_patchif.py"

# model parameters
backbone="mobilenet_v2"
ad_model="eif"
n_estimators=100
max_nodes=10000

# dataset parameters
categories=("pill")
contamination_ratio=0.1

# device
device_num=2

# experiment parameters
# seeds=(0 1 2)
seeds=(0)

# experiment name
# exp_name="03-06-2025_17-28-49_PatchIF_eif_mobilenet_v2_n_estimators_100_contamination_0.1"

python $script_path \
  --train \
  --test \
  --seeds "${seeds[@]}"\
  --backbone_model_name $backbone \
  --ad_model_name $ad_model \
  --n_estimators $n_estimators \
  --max_nodes $max_nodes \
  --plus \
  --categories "${categories[@]}" \
  --contaminate \
  --contamination_ratio $contamination_ratio \
  --save_model \
  --save_metrics \
  --device $device_num \
  --file_pos 0



