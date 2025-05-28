"""
Python script containig the implementation of the PatchIF model
"""

from __future__ import annotations
import os
import ipdb
from typing import Mapping, Union, Any, Dict, List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
#from memory_profiler import profile
from torch import Tensor, nn
from random import sample
from scipy.ndimage import gaussian_filter

from exiffi_core.model import ExtendedIsolationForest as EIF
from exiffi_core.model import IsolationForest as IF

from ...utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from ...utilities.get_sizes import *

from moviad.models.padim.padim import EMBEDDING_SIZES

class PatchIF(nn.Module):
    """
    PatchIF Module
    """

    def __init__(
        self,
        backbone_model_name: str = "mobilenet_v2",
        layers_idxs: list = ["fetures.4", "features.7", "features.10"],
        ad_model_type: str = "eif",
        plus: bool = True,
        eta: float = 1.5,
        n_estimators: int = 100,
        max_samples: int = 256,
        max_depth: str = "auto",
        device: torch.device = torch.device("cpu")
    ):
        super(PatchIF,self).__init__()

        self.backbone_model_name = backbone_model_name
        self.layers_idxs = layers_idxs
        self.ad_model_type = ad_model_type
        self.plus = plus
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.device = device

        # Load the feature extractor
        self.load_backbone()
        # dimensionality reduction: random projection
        random_dims = torch.tensor(sample(range(0, self.t_d), self.d))
        self.random_dimensions = torch.nn.Parameter(random_dims, requires_grad=False)

        # Load the anomaly detection model
        self.load_ad_model()

    @property
    def name(self):
        """
        Returns the name of the model.
        """
        return "PatchIF"

    def load_ad_model(self):

        if self.ad_model_type == "eif":
            self.ad_model = EIF(
                plus = self.plus,
                eta = self.eta,
                n_estimators = self.n_estimators,
                max_samples = self.max_samples,
                max_depth = self.max_depth,
            )

        elif self.ad_model_type == "if":
            self.ad_model = IF(
                n_estimators = self.n_estimators,
                max_samples = self.max_samples,
                max_depth = self.max_depth
            )
        else:
            raise ValueError(f"Unknown anomaly detection model type: {self.ad_model_type}. Supported types: 'eif', 'if'.")

    #NOTE: For the moment I am copying exactly the methods from Padim

    def load_backbone(self):
        """
        Load the backbone model

        Args:
            backbone_model_name: one of the following strings: 'wide_resnet50_2', 'mobilenet_v2'
        """
        backbone = CustomFeatureExtractor(
            model_name=self.backbone_model_name,
            layers_idx=self.layers_idxs,
            device=self.device,
            frozen=True,
        )
        self.backbone_model = backbone

        # define the backbone behavior
        def backbone_forward(x):
            self.outputs = backbone(x)

        self.backbone = backbone_forward

        # save the true and random projection dimensions
        self.t_d, self.d = EMBEDDING_SIZES[self.backbone_model_name][
            tuple(self.layers_idxs)
        ]

    @staticmethod
    def embedding_concat(
            x: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:

        """
        Concatenate the embeddings of two tensors x and y

        Args:
            x: Tensor of shape (B, C1, H1, W1)
            y: Tensor of shape (B, C2, H2, W2)

        Returns:
            z: Tensor of shape (B, C1 + C2, H2, W2)
            where H2 = H1 / s and W2 = W1 / s, with s = H1 / H2
        """
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def raw_feature_maps_to_embeddings(
        self,
        layer_outputs: Dict[str, List[torch.Tensor]]
    ):
        """
        Given a dict of lists of outputs of the layers, concatenate the feature maps and
        eventually reduce the dimensionality to return the embedding vectors.

        - embedding vector shape: (B, C, H, W)
        - B = number of samples in the train set
        - C = number of "channels", or number of feature maps --> may be reduced by dim. reduction
        - H, W = height and width of the feature maps
        """
        # concatenate the outputs of the different dataloader batches
        output_tensors: dict[str, torch.Tensor] = {
            layer: torch.cat(outputs, 0) for layer, outputs in layer_outputs.items()
        }
        # concatenate the feature maps to get the raw embedding vectors
        embedding_vectors: torch.Tensor = output_tensors[self.layers_idxs[0]]
        for layer in self.layers_idxs[1:]:
            embedding_vectors = PatchIF.embedding_concat(
                embedding_vectors, output_tensors[layer]
            )
        # dimensionality reduction: select the random dimensions to reduce the embedding vectors
        embedding_vectors = torch.index_select(
            embedding_vectors.to(self.device), 1, self.random_dimensions
        )
        return embedding_vectors

    def forward(self, x):

        """
        In training mode just return the patch embeddings, as done in Padim. Then in the Trainer class
        the embeddings obtained from all the input samples will be grouped together to obtain the memory bank
        on which `self.ad_model` will be trained.

        In inference mode, use the predict (or _predict) method of self.ad_model IF/EIF model on the patch embeddings of a test sample.
        Successfully the anomaly map and the image level anomaly score will be computed.
        """

        # 1. extract feature maps and get the raw layer outputs (conv. feature maps)
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.layers_idxs
        }
        # forward through the net to get the intermediate outputs with the hooks
        with torch.no_grad():
            _ = self.backbone(x)

        # get intermediate layer outputs
        for layer, output in zip(self.layers_idxs, self.outputs):  # new
            layer_outputs[layer].append(output.cpu().detach())  # new

        # initialize hook outputs
        self.outputs = []

        if self.training:
            return layer_outputs

        # Inference model → here we use the predict method of the ad_model
        # 2. use the feature maps to get the embeddings
        embedding_vectors = self.raw_feature_maps_to_embeddings(layer_outputs)

        #NOTE: embedding_vectors should have shape (B, C, H, W).
        # To make them usable for self.ad_model we need to reshape them as follows:
        # embedding_vectors = embedding_vectors.view(-1,embedding_vectors.size(1))

        #NOTE: In order to be able to produce the anomaly map I have to iterate over
        # all the patches (over all the H and W dimensions) and apply the predict method
        # on each one of them → so the predict method will be applied on a (32,40) tensor
        # and will give use the anomaly score → at the end we will have a tensor of shape
        # (B, H, W) with the anomaly scores for each patch

        # Copilot method:
        # anomaly_scores = []
        # for i in range(embedding_vectors.size(2)):
        #     patch_embedding = embedding_vectors[:, :, i, :].view(-1, embedding_vectors.size(1))
        #     anomaly_score = self.ad_model.predict(patch_embedding)
        #     anomaly_scores.append(anomaly_score)

        # My method
        anomaly_scores = np.zeros(shape=(embedding_vectors.size(0),embedding_vectors.size(2),embedding_vectors.size(3)))
        for i in range(embedding_vectors.size(2)):
            for j in range(embedding_vectors.size(3)):
                patch_embedding = embedding_vectors[:, :, i, j].view(-1, embedding_vectors.size(1))
                ipdb.set_trace()
                anomaly_score = self.ad_model.predict(patch_embedding.cpu().numpy())
                anomaly_scores[:, i, j] = anomaly_score

        ipdb.set_trace()

        # 4. upsample
        score_map = (
            F.interpolate(
                input=torch.tensor(anomaly_scores).unsqueeze(1),
                size=x.size(2),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        # 5. apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # 6. the image anomaly score is the maximum score in the score map
        img_scores = score_map.reshape(score_map.shape[0], -1).max(axis=1)

        # unsqueeze the score_map to have shape (B, 1, H, W)
        score_map = np.expand_dims(score_map, axis=1)

        return score_map, img_scores

    def get_anomaly_score(self):

        """
        Overwrite the get_anomaly_score method to compute the anomaly score using the
        predict (or _predict) methods of self.ad_model (EIF or IF) on the patch embeddings
        """

        pass
