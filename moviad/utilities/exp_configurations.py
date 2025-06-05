"""
Experiment configurations
"""

import numpy as np
import random
import torch
import socket

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

hostname = socket.gethostname()
if hostname == "acquario3":
    DATASET_PATHS = {
        "mvtec": "/mnt/disk1/manuel_barusco/CL_VAD/adcl_paper/data/mvtec",
        "realiad": "/mnt/disk1/yfbenkhalifa/datasets/realiad/realiad_256",
        "visa": "/mnt/disk1/yfbenkhalifa/datasets/visa"
    }
elif hostname == "aquarium2":
    DATASET_PATHS = {
        "mvtec": "/mnt/mydisk/manuel_barusco/datasets/mvtec",
        "visa": "/mnt/mydisk/manuel_barusco/datasets/visa"
    }
else:
    print(f"In {hostname} there are not vad datasets")

AD_LAYERS = {
    "mobilenet_v2": ["features.4", "features.7", "features.10"],
    "wide_resnet50_2": ["layer1", "layer2", "layer3"],
    "phinet_1.2_0.5_6_downsampling": [2, 6, 7],
    "micronet-m1": [2, 4, 5],
    "mcunet-in3": [3, 6, 9],
    "resnet18": ["layer1", "layer2", "layer3"]
}

EMBEDDING_SIZES = {
    "phinet_1.2_0.5_6_downsampling": {
        (4, 5, 6): (200, 50),
        (5, 6, 7): (400, 100),
        (6, 7, 8): (576, 144),
        (2, 6, 7): (376, 94),
    },
    "micronet-m1": {
        (1, 2, 3): (40, 10),
        (2, 3, 4): (64, 16),
        (3, 4, 5): (112, 28),
        (2, 4, 5): (112, 28),
    },
    "mcunet-in3": {
        (3, 6, 9): (80, 20),
        (6, 9, 12): (112, 28),
        (9, 12, 15): (184, 46),
        (2, 6, 14): (136, 34),
    },
    "mobilenet_v2": {
        ("features.4", "features.7", "features.10"): (160, 40),
        ("features.7", "features.10", "features.13"): (224, 56),
        ("features.10", "features.13", "features.16"): (320, 80),
        ("features.3", "features.8", "features.14"): (248, 62),
    },
    "wide_resnet50_2": {("layer1", "layer2", "layer3"): (1792, 550)},
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

    random.seed(int(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('#'* 50)
    print(f"Seed set to {seed}")
    print('#'* 50)

