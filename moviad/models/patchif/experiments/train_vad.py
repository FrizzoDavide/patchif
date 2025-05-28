"""
Python script to test the first things needed for the patchif project.
- Try to obtain the memory bank of patch features from the PatchCore model
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

# moviad imports
#NOTE: AD model → CFA model
# from moviad.models.cfa.cfa import CFA

#NOTE: AD model → PatchCore
from moviad.models.patchcore.patchcore import PatchCore

#NOTE: AD model → PaDiM
from moviad.models.padim.padim import Padim

#NOTE: AD model → PatchIF
from moviad.models.patchif.patchif import PatchIF

#NOTE: Trainer → TrainerPatchCore
from moviad.trainers.trainer import TrainerResult
from moviad.trainers.trainer_patchcore import TrainerPatchCore

#NOTE: Trainer → TrainerPadim
from moviad.trainers.trainer_padim import PadimTrainer

#NOTE: Trainer → TrainerPatchIF
from moviad.trainers.trainer_patchif import TrainerPatchIF

#NOTE: Datasets → MVTec and RealIad
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset, CATEGORIES
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum

#NOTE: Feature Extractor → CustomFeatureExtractor
from moviad.trainers.trainer_stfpm import TrainerSTFPM
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor, TORCH_BACKBONES, OTHERS_BACKBONES

#NOTE:: Import utility  functions from manage_files.py
from moviad.utilities.manage_files import generate_path, get_current_time, get_most_recent_file, open_element, save_element

#NOTE: TaskType
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator
from moviad.utilities.exp_configurations import MODEL_NAMES, DATASET_NAMES, DATASET_PATHS, AD_LAYERS

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name",type=str,default="mvtec",help="Dataset name")
parser.add_argument("--model_name",type=str,default="patchcore",help="Dataset name")
parser.add_argument("--category", type=str, default="pill", help="Dataset category to test")
parser.add_argument("--ad_model_type", type=str, default="eif", help="Type of AD model, eif or if")
parser.add_argument("--backbone", type=str, default="mobilenet_v2", help="Model backbone")
parser.add_argument("--device_num", type=int, default=0, help="Number of the CUDA device to use")
parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators for the IF/EIF model, for patchif")
parser.add_argument("--save_model", action='store_true', help="Flag to save the model")
parser.add_argument("--train_model", action='store_true', help="Flag to train the model, otherwise load the state dict of an already saved model")
parser.add_argument("--anomaly_map", action='store_true', help="Flag to produce the anomaly maps")

args = parser.parse_args()

setproctitle.setproctitle(f"exp_{args.dataset_name}_{args.category}_{args.model_name}_{args.backbone}")

assert args.dataset_name in DATASET_NAMES, f"Dataset {args.dataset_name} not supported. Supported datasets: {DATASET_NAMES}"
assert args.category in CATEGORIES, f"Dataset {args.dataset_name} not supported. Supported datasets: {CATEGORIES}"
assert args.model_name in MODEL_NAMES, f"Model {args.model_name} not supported. Supported models: {MODEL_NAMES}"
assert args.backbone in TORCH_BACKBONES or args.backbone in OTHERS_BACKBONES, f"Backbone {args.backbone} not supported. Supported backbones: {TORCH_BACKBONES + OTHERS_BACKBONES}"

device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

print('#'* 50)
print("------EXPERIMENT DETAILS------")
print(f"Dataset: {args.dataset_name}")
print(f"Category: {args.category}")
print(F"Model name: {args.model_name}")
print(f"Backbone: {args.backbone}")
print(f"AD Layers: {AD_LAYERS[args.backbone]}")
print(f"Device: {device}")
print('#'* 50)

#NOTE: Define the feature extractor (i.e. pre trained CNN) using the CustomFeatureExtractor class

feature_extractor = CustomFeatureExtractor(
    model_name = args.backbone,
    layers_idx = AD_LAYERS[args.backbone],
    device = device,
    frozen = True,
    quantized = False,
    calibration_dataloader=None
)

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

#NOTE: Define the model using the PatchCore class

if args.model_name == "patchcore":
    model = PatchCore(
        device = device,
        input_size = (224, 224),
        feature_extractor = feature_extractor,
        num_neighbors = 9,
        apply_quantization = False,
        k = 10000
    )

elif args.model_name == "padim":
    model = Padim(
        backbone_model_name = args.backbone,
        class_name = args.category,
        device = device,
        layers_idxs = [4,7,10],
        diag_cov = False
    )
elif args.model_name == "patchif":
    model = PatchIF(
        backbone_model_name = args.backbone,
        layers_idxs = AD_LAYERS[args.backbone],
        ad_model_type = args.ad_model_type,
        n_estimators = args.n_estimators,
        device = device
    )
else:
    print('#'* 50)
    print(f"Model {args.model_name} not yet implemented in this script")
    print('#'* 50)
    quit()

print('#'* 50)
print(f"Model chosen: {args.model_name}")
print('#'* 50)

#NOTE: PatchIF test:

x = torch.rand(1, 3, 224, 224).to(device)
model.to(device)
model.eval()
score_map,img_scores = model(x)

#NOTE: Define the model directory path where the model will be saved

model_dirpath = generate_path(
    basepath = os.getcwd(),
    folders = [
        "models_state_dict",
        args.dataset_name,
        args.category,
        model.name,
        args.backbone,
    ]
)

#NOTE: Define the trainer using the TrainerPatchCore class
# and use the `train` method to train the model and get the memory bank

if args.train_model:

    model.to(device)
    model.train()

    if args.model_name == "patchcore":

        trainer = TrainerPatchCore(
            patchcore_model = model,
            train_dataloader = train_loader,
            test_dataloder = test_loader,
            device = device,
            coreset_extractor = None,
            logger = None,
        )


    elif args.model_name == "padim":

        trainer = PadimTrainer(
            model = model,
            train_dataloader = train_loader,
            test_dataloader = test_loader,
            device = device,
            apply_diagonalization = False,
            logger = None,
        )

    elif args.model_name == "patchif":

        trainer = TrainerPatchIF(
            model = model,
            train_dataloader = train_loader,
            test_dataloader = test_loader,
            device = device,
            logger = None,
        )

    print('#'* 50)
    print(f"Training the {args.model_name} model")
    print('#'* 50)
    trainer.train()

    if args.save_model:
        if args.model_name == "patchcore":
            filename=f"{get_current_time()}_{args.dataset_name}_{args.category}_{model.name}_{args.backbone}"
            save_element(
                element = model,
                dirpath = model_dirpath,
                filename = filename,
                filetype = "pth",
                no_time = False
            )

try:
    model_path = get_most_recent_file(model_dirpath,file_pos=0)
    print('#'* 50)
    print(f"Loading the model state dict from: {model_path}")
    print('#'* 50)
except FileNotFoundError:
    print("Model directory not found. Please train the model first or check the path.")

#NOTE: Evaluate the model on the test set

if args.model_name == "patchcore":
    model = PatchCore(
        device = device,
        input_size = (224, 224),
        feature_extractor = feature_extractor,
        num_neighbors = 9,
        apply_quantization = False,
        k = 1000
    )
    model.load_model(model_path)
    model.to(device)
    model.eval()
elif args.model_name == "padim":
    model = Padim(
        backbone_model_name = args.backbone,
        class_name = args.category,
        device = device,
        layers_idxs = [4,7,10],
        diag_cov = False
    )

    model.load_state_dict(
        torch.load(model_path, weights_only=False, map_location=device), strict=False
    )
    model.to(device)
    model.eval()
else:
    print('#'* 50)
    print(f"Model {args.model_name} not yet implemented in this script")
    print('#'* 50)
    quit()


evaluator = Evaluator(test_loader, device)
metrics = evaluator.evaluate(model)

results = TrainerResult(**metrics)

print('#'* 50)
print("Evaluation performances:")
print(f"""
img_roc: {results.img_roc_auc}
pxl_roc: {results.pxl_roc_auc}
f1_img: {results.img_f1}
f1_pxl: {results.pxl_f1}
img_pr: {results.img_pr_auc}
pxl_pr: {results.pxl_pr_auc}
pxl_pro: {results.pxl_au_pro}
""")
print('#'* 50)

#NOTE: Produce the anomaly maps
if args.anomaly_map:

    visual_test_path = generate_path(
        basepath = os.getcwd(),
        folders = [
            "anomaly_maps",
            args.dataset_name,
            args.category,
            model.name,
            args.backbone,
        ]
    )
    for images, labels, masks, paths in tqdm(iter(test_loader)):
        anomaly_maps, pred_scores = model(images.to(device))

        anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

        for i in range(anomaly_maps.shape[0]):
            model.save_anomaly_map(
                dirpath = visual_test_path,
                anomaly_map = anomaly_maps[i].cpu().numpy(),
                pred_score = pred_scores[i],
                filepath = paths[i],
                x_type = labels[i],
                mask = masks[i]
            )
