"""
Python script to try out the patchif idea → load the memory bank of an already trained
PatchCore model and try to feed it in input to and ExtendedIsolationForest model and fit it
"""

# general imports
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
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset, CATEGORIES
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor, TORCH_BACKBONES, OTHERS_BACKBONES
from moviad.utilities.manage_files import generate_path, get_current_time, get_most_recent_file, open_element, save_element

from exiffi_core.model import ExtendedIsolationForest as EIF

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name",type=str,default="mvtec",help="Dataset name")
parser.add_argument("--dataset_path", type=str, help="Path of the directory where the dataset is stored")
parser.add_argument("--model_name",type=str,default="patchcore",help="Dataset name")
parser.add_argument("--category", type=str, default="pill", help="Dataset category to test")
parser.add_argument("--backbone", type=str, default="mobilenet_v2", help="Model backbone")
parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
parser.add_argument("--device_num", type=int, default=0, help="Number of the CUDA device to use")
# EIF parameters
parser.add_argument("--n_estimators", type=int, default=100, help="Number of base estimators in the ensemble")
parser.add_argument("--max_samples", type=int, default=256, help="Number of samples to draw from X to train each base estimator")

args = parser.parse_args()


setproctitle.setproctitle(f"patchif_exp_{args.dataset_name}_{args.category}_{args.model_name}_{args.backbone}")

MODEL_NAMES = (
    "patchcore",
    "cfa",
    "padim",
    "stfpm",
    "ganomaly",
    "fastflow",
    "past",
    "rd4ad",
    "simplenet"
)

DATASET_NAMES = (
    "mvtec", # MVTec dataset
    "iad", # RealIad dataset
    "visa", # VisA dataset
    "miic", # MIIC dataset
)

assert args.dataset_name in DATASET_NAMES, f"Dataset {args.dataset_name} not supported. Supported datasets: {DATASET_NAMES}"
assert args.category in CATEGORIES, f"Dataset {args.dataset_name} not supported. Supported datasets: {CATEGORIES}"
assert args.model_name in MODEL_NAMES, f"Model {args.model_name} not supported. Supported models: {MODEL_NAMES}"
assert args.backbone in TORCH_BACKBONES or args.backbone in OTHERS_BACKBONES, f"Backbone {args.backbone} not supported. Supported backbones: {TORCH_BACKBONES + OTHERS_BACKBONES}"

device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

#NOTE: Define the feature extractor (i.e. pre trained CNN) using the CustomFeatureExtractor class

feature_extractor = CustomFeatureExtractor(
    model_name = args.backbone,
    layers_idx = args.ad_layers,
    device = device,
    frozen = True,
    quantized = False,
    calibration_dataloader=None
)

#NOTE: Define the PatchCore model using the PatchCore class

patchcore = PatchCore(
    device = device,
    input_size = (224, 224),
    feature_extractor = feature_extractor,
    num_neighbors = 9,
    apply_quantization = False,
    k = 10000
)
patchcore.to(device)

#NOTE: Define and load the test dataset using the MVTecDataset class

test_dataset = MVTecDataset(
    task = TaskType.SEGMENTATION,
    root = args.dataset_path,
    category = args.category,
    split = Split.TEST,
    norm = True,
    img_size = (224,224),
    gt_mask_size = None,
    preload_imgs = True
)

test_dataset.load_dataset()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

#NOTE: Define the model directory path where the model will be saved

model_dirpath = generate_path(
    basepath = os.getcwd(),
    folders = [
        "models",
        args.dataset_name,
        args.category,
        patchcore.name,
        args.backbone,
    ]
)

try:
    model_path = get_most_recent_file(model_dirpath,file_pos=0)
    print('#'* 50)
    print(f"Loading the model state dict from: {model_path}")
    print('#'* 50)
except FileNotFoundError:
    print("Model directory not found. Please train the model first or check the path.")

#NOTE: Evaluate the model on the test set

patchcore = PatchCore(
    device = device,
    input_size = (224, 224),
    feature_extractor = feature_extractor,
    num_neighbors = 9,
    apply_quantization = False,
    k = 1000
)
patchcore.load_model(model_path)

print('#'* 50)
print("Successfully loaded the model state dict")
print(f"Memory bank shape: {patchcore.memory_bank.shape}")
print('#'* 50)

#NOTE: Creat and instance of EIF

eif = EIF(
    plus = True,
    n_estimators = args.n_estimators,
    max_samples = args.max_samples,
    use_centroid_split = True
)
ipdb.set_trace()

#NOTE: Fit the EIF model on the memory bank of the PatchCore model

print('#'* 50)
print("Fitting the EIF model on the memory bank of the PatchCore model")
print('#'* 50)

eif.fit(patchcore.memory_bank.cpu().numpy())

#TODO: After having fitted the model I have to compute the anomaly scores passing in
# input the embeddings of the test set images, not the image directly. In order to obtain
# the embeddings I have to use the forward method of the PatchCore model but in the `train`
# mode (where embeddings are computed).


