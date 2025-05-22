#!/bin/bash

script_path="patchif_test.py" 
backbone="mobilenet_v2"
ad_layers=("features.4 features.7 features.10")
category="pill"
dataset_path="../../../datasets/mvtec/"
device_num=0


python $script_path \
  --backbone $backbone \
  --ad_layers ${ad_layers[@]} \
  --device_num $device_num \
  --dataset_path $dataset_path \
  --category $category

