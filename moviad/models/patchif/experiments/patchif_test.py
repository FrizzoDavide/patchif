"""
Python script to test the first things needed for the patchif project.
- Try to obtain the memory bank of patch features from the PatchCore model
"""

# general imports
import os
from pathlib import Path
import argparse
import ipdb
import torch
import gc
from torchvision.transforms import transforms
from tqdm import tqdm

# moviad imports
#NOTE: AD model → CFA model
# from moviad.models.cfa.cfa import CFA

#NOTE: AD model → PatchCore
from moviad.models.patchcore.patchcore import PatchCore

#NOTE: Trainer → TrainerPatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore

#NOTE: Datasets → MVTec and RealIad
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum

#NOTE: Feature Extractor → CustomFeatureExtractor
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

#NOTE: TaskType
from moviad.utilities.configurations import TaskType

parser = argparse.ArgumentParser()

parser.add_argument("--backbone", type=str, help="Model backbone")
parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
parser.add_argument("--device_num", type=int, default=0, help="Number of the CUDA device to use")
parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
parser.add_argument("--category", type=str, help="Dataset category to test")

args = parser.parse_args()

device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print('#'* 50)
print(f"Using device: {device}")
print('#'* 50)

print('#'* 50)
print("Building feature extractor")
print('#'* 50)

#NOTE: Define the feature extractor (i.e. pre trained CNN) using the CustomFeatureExtractor class
feature_extractor = CustomFeatureExtractor(
    model_name = args.backbone,
    layers_idx = args.ad_layers,
    device = device,
    frozen = True,
    quantized = False,
    calibration_dataloader=None
)


#TODO: Define and load the dataset using the MVTecDataset class

print('#'* 50)
print("Loading MVTec dataset")
print('#'* 50)

dataset = MVTecDataset(
    task = TaskType.SEGMENTATION,
    root = args.dataset_path,
    category = args.category,
    split = args.split,
    norm = True,
    img_size = (224,224),
    gt_mask_size = None,
    preload_imgs = True
)

ipdb.set_trace()

#TODO: Define the model using the PatchCore class

#TODO: Define the trainer using the TrainerPatchCore class

#TODO: Train the model using the trainer class → at this point after the model is trained
# I guess that I should be able to access the memory bank of patch features

