"""
Python script to print out all the statistics about model sizes and FLOPs for the different
backbones we can use for the vad models, just to get and idea of what we are working on
"""

import os
from pathlib import Path
import argparse
import setproctitle
import ipdb
import torch
import gc
from torchvision.transforms import transforms
from tqdm import tqdm

from moviad.models.patchcore.patchcore import PatchCore
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor, TORCH_BACKBONES, OTHERS_BACKBONES

from moviad.utilities.manage_files import generate_path, get_most_recent_file

parser = argparse.ArgumentParser()

parser.add_argument("--model_name",type=str,default="patchcore",help="Dataset name")
parser.add_argument("--backbone", type=str, default="mobilenet_v2", help="Model backbone")
parser.add_argument("--dataset_name",type=str,default="mvtec",help="Dataset name")
parser.add_argument("--category", type=str, default="pill", help="MVTec dataset category")
parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
parser.add_argument("--device_num", type=int, default=0, help="Number of the CUDA device to use")

args = parser.parse_args()

setproctitle.setproctitle(f"backbones_exp_{args.model_name}_{args.backbone}")

device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

feature_extractor = CustomFeatureExtractor(
    model_name = args.backbone,
    layers_idx = args.ad_layers,
    device = device,
    frozen = True,
    quantized = False,
    calibration_dataloader=None
)

model = PatchCore(
    device = device,
    input_size = (224, 224),
    feature_extractor = feature_extractor,
    num_neighbors = 9,
    apply_quantization = False,
    k = 10000
)
model.to(device)

model_dirpath = generate_path(
    basepath = os.getcwd(),
    folders = [
        "models",
        args.dataset_name,
        args.category,
        model.name,
        args.backbone,
    ]
)

try:
    model_path = get_most_recent_file(dirpath = model_dirpath, file_pos = 0)
    print('#'* 50)
    print(f"Loading the model state dict from: {model_path}")
    print('#'* 50)
except FileNotFoundError:
    print(f"No model found in {model_dirpath}. Are you sure you trained the model?")

model.load_model(model_path)

sizes,total_size = model.get_model_size_and_macs()

print('#'* 50)
print("------MODEL STATISTICS------")
print(f"Total size: {total_size}")

print("----FEATURE EXTRACTOR SIZE----")
print(f"sizes: {sizes['feature_extractor']}")
print("----MEMORY BANK SIZE----")
print(f"sizes: {sizes['memory_bank']}")
print('#'* 50)

