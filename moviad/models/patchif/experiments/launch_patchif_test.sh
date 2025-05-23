#!/bin/bash

# Export the conda environment path to PATH
# export PATH="/home/davide_frizzo/anaconda3/envs/moviad/bin/:$PATH"

script_path="patchif_test.py"
backbone="mobilenet_v2"
ad_layers=("features.4 features.7 features.10")
category="pill"
dataset_path="/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec"
device_num=1

python $script_path \
  --backbone $backbone \
  --ad_layers ${ad_layers[@]} \
  --device_num $device_num \
  --dataset_path $dataset_path \
  --category $category

