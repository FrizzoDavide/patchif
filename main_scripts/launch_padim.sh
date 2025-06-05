#!/bin/bash

# backbones:
# - wide_resnet50_2 ["layer1", "layer2","layer3"]
# - mobilenet_v2 [4,7,10] [7,10,13] [10,13,16]
# - phinet_1.2_0.5_6_downsampling [4,5,6] [5,6,7] [6,7,8]
# - mcunet-in3 [3,6,9] [6,9,12] [9,12,15]
# - micronet-m1 [1,2,3] [2,3,4] [3,4,5]

# you can provide a directory to save the metrics like: --results_dirpath ./padim_metrics

python main_padim.py \
    --debug \
    --train \
    --test \
    --backbone_model_name "mobilenet_v2" \
    --device "cuda:2" \
    --seeds 0 \
    --categories "pill" \
    --ad_layers_idxs 4 7 10
