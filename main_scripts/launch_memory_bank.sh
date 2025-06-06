#!/bin/bash

script_path="memory_bank.py"

python $script_path \
  --dataset_name "mvtec" \
  --category "pill" \
  --backbone "mobilenet_v2" \
  --file_pos 0 \
  --decomposition "umap" \
  --n_components 3 \
  --n_neighbors 15 \
  --min_dist 0.1
