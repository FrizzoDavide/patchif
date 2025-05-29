#!/bin/bash

script_path="contaminate_data.py"

dataset_name="mvtec"
category="pill"

#NOTE: device parameters
device_num=2

python $script_path \
  --dataset_name $dataset_name \
  --category $category \
  --device_num $device_num
