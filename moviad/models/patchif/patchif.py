"""
Python script containig the implementation of the PatchIF model as a subclass of the
PatchCore model
"""

from __future__ import annotations
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
#from memory_profiler import profile
from torch import Tensor, nn

from moviad.models.patchcore.patchcore import PatchCore
from exiffi_core.model import ExtendedIsolationForest as EIF
from exiffi_core.model import IsolationForest as IF

from .product_quantizer import ProductQuantizer
from ...models.patchcore.anomaly_map import AnomalyMapGenerator
from ...utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from ...utilities.get_sizes import *

class PatchIF(PatchCore):
    """
    PatchIF Module
    """

    def __init__(
        self,
        ad_model_type: str = "eif",
        plus: bool = True,
        eta: float = 1.5,
        n_estimators: int = 100,
        max_samples: int = 256,
        max_depth: str = "auto",
        use_centroid_importance: bool = False,
        use_dist_split: bool = False,
        use_centroid_split: bool = False,
        use_centr_post_split: bool = False,
        use_centroid_woff: bool = False,
    ):

        if ad_model_type == "eif":
            self.ad_model = EIF(
                plus = plus,
                eta = eta,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_depth=max_depth,
                use_centroid_importance=use_centroid_importance,
                use_dist_split=use_dist_split,
                use_centroid_split=use_centroid_split,
                use_centr_post_split=use_centr_post_split,
                use_centroid_woff=use_centroid_woff
            )

        elif ad_model_type == "if":
            self.ad_model = IF(
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_depth=max_depth
            )
        else:
            raise ValueError(f"Unknown anomaly detection model type: {ad_model_type}. Supported types: 'eif', 'if'.")

    @property
    def name(self):
        """
        Returns the name of the model.
        """
        return "PatchIF"

    def forward(self,p:float=0.1):

        """
        In training mode get the memory bank of patch features (put together the patch embeddings of all the trainign samples)
        and fit the ExtendedIsolationForest model on it. However here I need all the training samples to get the memory bank,
        so I should first train a PatchCore model to obtain the memory bank?

        In inference mode, use the predict (or _predict) method of self.ad_model IF/EIF model on the patch embeddings of a test sample
        """

        pass

    def get_anomaly_score(self):

        """
        Overwrite the get_anomaly_score method to compute the anomaly score using the
        predict (or _predict) methods of self.ad_model (EIF or IF) on the patch embeddings
        """

        pass
