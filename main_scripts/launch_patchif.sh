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
n_runs=1

# experiment name

# NOTE: PatchIF_EIF+
# exp_name="04-06-2025_07-52-28_PatchIF_eif_mobilenet_v2_n_estimators_100_contamination_0.1"

# NOTE: PatchIF_IF
# exp_name="04-06-2025_08-33-01_PatchIF_if_mobilenet_v2_n_estimators_100_contamination_0.1"

#NOTE: PatchIF_EIF
exp_name="04-06-2025_08-44-19_PatchIF_eif_mobilenet_v2_n_estimators_100_contamination_0.1"

python $script_path \
  --exp_name $exp_name \
  --test \
  --n_runs $n_runs \
  --backbone_model_name $backbone \
  --ad_model_name $ad_model \
  --n_estimators $n_estimators \
  --max_nodes $max_nodes \
  --categories "${categories[@]}" \
  --contaminate \
  --contamination_ratio $contamination_ratio \
  --save_metrics \
  --device $device_num \
  --file_pos 0



