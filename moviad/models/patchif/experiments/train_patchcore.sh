#!/bin/bash

# Let's move in the root folder of the moviad library
cd ~/patchif/main_scripts

script_path="main_patchcore.py"
mode="train"
dataset_path="../moviad/home/datasets/mvtec"
ad_layers=("features.4 features.7 features.10")
save_path="../moviad/models/patchif/experiments"

python $script_path \
  --mode $mode \
  --dataset_path $dataset_path \
  --category "pill" \
  --backbone "mobilenet_v2" \
  --ad_layers $ad_layers \
  --device "cuda:0" \
  --save_path $save_path
