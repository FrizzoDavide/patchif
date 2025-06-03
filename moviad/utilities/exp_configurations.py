"""
Experiment configurations
"""

import numpy as np
import random
import torch

MODEL_NAMES = (
    "patchif",
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
    "mvtec",
    "iad",
    "visa",
    "miic",
)

#NOTE: Paths to the datasets in acquario3
DATASET_PATHS = {
    "mvtec": "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec",
    "realiad": "/mnt/disk1/yfbenkhalifa/datasets/realiad/realiad_256",
    "visa": "/mnt/disk1/yfbenkhalifa/datasets/visa"
}

AD_LAYERS = {
    "mobilenet_v2": ["features.4", "features.7", "features.10"],
    "wide_resnet50_2": ["layer1", "layer2", "layer3"],
    "phinet_1.2_0.5_6_downsampling": [2, 6, 7],
    "micronet-m1": [2, 4, 5],
    "mcunet-in3": [3, 6, 9],
    "resnet18": ["layer1", "layer2", "layer3"]
}

def set_exp_seed(
        seed: int = 0,
) -> None:

    """
    Set the seed for reproducibility for the different libraries used in the project.

    Args:
        seed: integer, the seed to set

    Returns:
        None → the function just sets the seed for the different libraries.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('#'* 50)
    print(f"Seed set to {seed}")
    print('#'* 50)

