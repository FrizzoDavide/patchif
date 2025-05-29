"""
Python script to test the methods to contaminate the ataset with anomalies
in the MVTecDataset
"""

# general imports
import os
import argparse
import setproctitle
import ipdb
import torch

#NOTE: Dataset → MVTec
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset, CATEGORIES
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.exp_configurations import DATASET_NAMES, DATASET_PATHS

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name",type=str,default="mvtec",help="Dataset name")
parser.add_argument("--category", type=str, default="pill", help="Dataset category to test")
parser.add_argument("--device_num", type=int, default=0, help="Number of the CUDA device to use")

args = parser.parse_args()

setproctitle.setproctitle(f"contaminate_dataset")

assert args.dataset_name in DATASET_NAMES, f"Dataset {args.dataset_name} not supported. Supported datasets: {DATASET_NAMES}"

device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

print('#'* 50)
print("------DATASET CONTAMINATION EXPERIMENT------")
print(f"Dataset: {args.dataset_name}")
print(f"Category: {args.category}")
print(f"device: {device}")

print('#'* 50)
print(f"Defining training and test dataloaders from {args.dataset_name} dataset")
print('#'* 50)

#NOTE: Define and load the dataset using the MVTecDataset class

train_dataset = MVTecDataset(
    task = TaskType.SEGMENTATION,
    root = DATASET_PATHS[args.dataset_name],
    category = args.category,
    split = Split.TRAIN,
    norm = True,
    img_size = (224,224),
    gt_mask_size = None,
    preload_imgs = True
)

train_dataset.load_dataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

print('#'* 50)
print(f"-- TRAIN DATASET INFORMATION --")
print(f"Length: {len(train_dataset)}")
print(f"Contamination ratio: {train_dataset.compute_contamination_ratio()}")
print('#'* 50)

test_dataset = MVTecDataset(
    task = TaskType.SEGMENTATION,
    root = DATASET_PATHS[args.dataset_name],
    category = args.category,
    split = Split.TEST,
    norm = True,
    img_size = (224,224),
    gt_mask_size = None,
    preload_imgs = True
)

test_dataset.load_dataset()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

print('#'* 50)
print(f"-- TEST DATASET INFORMATION --")
print(f"Length: {len(test_dataset)}")
print(f"Test dataset contamination ratio: {test_dataset.compute_contamination_ratio()}")
print('#'* 50)

#NOTE: Contaminate the training dataset with anomalies from the test dataset

contamination_set_size = train_dataset.contaminate(
    source = test_dataset,
    ratio = 0.1,
    seed = 42
)

print('#'* 50)
print(f"-- CONTAMINATED TRAIN DATASET INFORMATION --")
print(f"Length: {len(train_dataset)}")
print(f"Contamination ratio: {train_dataset.compute_contamination_ratio()}")
print('#'* 50)

print('#'* 50)
print(f"-- TEST DATASET INFORMATION AFTER CONTAMINATION --")
print(f"Length: {len(test_dataset)}")
print(f"Test dataset contamination ratio: {test_dataset.compute_contamination_ratio()}")
print('#'* 50)

