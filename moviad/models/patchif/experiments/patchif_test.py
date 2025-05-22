"""
Python script to test the first things needed for the patchif project.
- Try to obtain the memory bank of patch features from the PatchCore model
"""

# general imports
import os
import ipdb
import torch
import gc
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

# moviad imports
#NOTE: AD model → CFA model
from moviad.models.cfa.cfa import CFA

#NOTE: AD model → PatchCore
from moviad.models.patchcore.patchcore import PatchCore

#NOTE: Trainer → TrainerPatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore

#NOTE: Datasets → MVTec and RealIad
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum

#NOTE: Feature Extractor → CustomFeatureExtractor
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

print('#'* 50)
print("All libraries loaded correctly")
print('#'* 50)

#TODO: Following the code contained in the `trainer_patchcore` function inside `main_patchcore.py`

#TODO: Define the feature extractor (i.e. pre trained CNN) using the CustomFeatureExtractor class

#TODO: Define and load the dataset using the MVTecDataset class

#TODO: Define the model using the PatchCore class

#TODO: Define the trainer using the TrainerPatchCore class

#TODO: Train the model using the trainer class → at this point after the model is trained
# I guess that I should be able to access the memory bank of patch features

