#!/bin/bash

script_path="main_patchif.py"


# model parameters
backbone="wide_resnet50_2"
ad_model="if"
n_estimators=100
max_nodes=10000

# dataset parameters
categories=("pill")
contamination_ratio=0.1

# device
device_num=2

# experiment parameters
n_runs=5
mode="${1:-'train'}"

#NOTE: mode variable:
# - 'train' to train the model and test it on the test set
# - 'test' to only test the model on the test set (using a previously trained model)
# - 'anomaly_maps' to produce the anomaly maps of a previously trained model

# experiment name

# NOTE: PatchIF_EIF+
# exp_name="04-06-2025_07-52-28_PatchIF_eif_mobilenet_v2_n_estimators_100_contamination_0.1"
# best_seed=0

# NOTE: PatchIF_IF
# mobilenet_v2
# exp_name="04-06-2025_12-21-18_PatchIF_if_mobilenet_v2_n_estimators_300_contamination_0.1"
# wide_resnet50_2
exp_name="04-06-2025_14-50-06_PatchIF_if_wide_resnet50_2_n_estimators_100_contamination_0.1"
best_seed=4

#NOTE: PatchIF_EIF
# exp_name="04-06-2025_08-44-19_PatchIF_eif_mobilenet_v2_n_estimators_100_contamination_0.1"
# best_seed=0

if [ "$mode" = "train" ]; then

  python $script_path \
    --train \
    --test \
    --n_runs $n_runs \
    --backbone_model_name $backbone \
    --ad_model_name $ad_model \
    --n_estimators $n_estimators \
    --max_nodes $max_nodes \
    --categories "${categories[@]}" \
    --contaminate \
    --contamination_ratio $contamination_ratio \
    --save_model \
    --save_metrics \
    --device $device_num \
    --file_pos 0

elif [ "$mode" = "test" ]; then

  python $script_path \
    --exp_name $exp_name \
    --test \
    --inference \
    --n_runs 1 \
    --best_seed $best_seed \
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

elif [ "$mode" = "anomaly_maps" ]; then

  python $script_path \
    --exp_name $exp_name \
    --test \
    --inference \
    --n_runs 1 \
    --best_seed $best_seed \
    --backbone_model_name $backbone \
    --ad_model_name $ad_model \
    --n_estimators $n_estimators \
    --max_nodes $max_nodes \
    --categories "${categories[@]}" \
    --contaminate \
    --contamination_ratio $contamination_ratio \
    --save_figures \
    --device $device_num \
    --file_pos 0

else
  echo "Invalid mode. Please use 'train', 'test', or 'anomaly_maps'."
  exit 1
fi



