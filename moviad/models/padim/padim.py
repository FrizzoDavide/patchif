import os
from random import sample
from typing import Mapping, Union, Any, Dict, List, Tuple

import numpy as np
#from profiler import profile
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis

import torch
from torch import nn
from torch.nn import functional as F

from ...utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

# Dict: "backbone_model_name" -> {(layer_idxs): (true_dimension, random_projection_dimension)}
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


def idx_to_layer_name(backbone_model_name, idx: Union[Tuple, List]):
    if backbone_model_name in ["wide_resnet50_2"]:
        return tuple(f"layer{i}" for i in idx)
    elif backbone_model_name == "mobilenet_v2":
        return tuple(f"features.{i}" for i in idx)
    else:
        return idx


class Padim(nn.Module):
    HYPERPARAMS = [
        "class_name",
        "backbone_model_name",
        "t_d",
        "d",
        "gauss_mean",
        "gauss_cov",
        "diag_cov",
        "layers_idxs",
    ]

    def __init__(
            self,
            backbone_model_name,
            class_name,
            device,
            layers_idxs: list,
            diag_cov=False,
    ):
        """
        Args:
            backbone_model_name: one of the following strings: 'wide_resnet50_2', 'mobilenet_v2'
            class_name: one of the following strings: 'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
            device: device where to run the model, e.g. 'cuda' or 'cpu'
            layers_idxs: list of integers or strings representing the indexes of the layers to use for feature extraction
            diag_cov: if True, keep only the diagonal elements of the covariance matrices
        """
        super(Padim, self).__init__()
        self.diagonal_gauss_cov = None
        self.class_name = class_name
        self.device = device
        self.diag_cov = diag_cov
        # feature extractor backbone model
        self.backbone_model_name = backbone_model_name
        self.layers_idxs = layers_idxs

        self.layers_idxs = idx_to_layer_name(
            backbone_model_name, layers_idxs
         )  # feature extraction layers

        self.load_backbone()
        # dimensionality reduction: random projection
        random_dims = torch.tensor(sample(range(0, self.t_d), self.d))
        self.random_dimensions = torch.nn.Parameter(random_dims, requires_grad=False)
        # training: learn the multivariate Gaussian distribution from the extracted features
        self.train_outputs = None  # list of mean and covariance matrix numpy arrays
        self.gauss_mean = None
        self.gauss_cov = None

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return "padim"

    @staticmethod
    def embedding_concat(x, y):
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
            self, layer_outputs: Dict[str, List[torch.Tensor]]
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
            embedding_vectors = Padim.embedding_concat(
                embedding_vectors, output_tensors[layer]
            )
        # dimensionality reduction: select the random dimensions to reduce the embedding vectors
        #NOTE: DIMENSIONALITY REDUCTION: This `torch.index_select` function selects the indexes contained in `self.random_dimensions`
        # along dimension 1 of `embedding_vectors` → `self.random_dimensions` contains indexes selected at random
        # so this is essentially the same thing as a random subsampling
        # This thing is done on the features, so it reduced the embedding dimension
        embedding_vectors = torch.index_select(
            embedding_vectors.to(self.device), 1, self.random_dimensions
        )
        return embedding_vectors

    def forward(self, x):
        # 1. extract feature maps and get the raw layer outputs (conv. feature maps)
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.layers_idxs
        }
        # forward through the net to get the intermediate outputs with the hooks
        with torch.no_grad():
            # _ = self.backbone(x)
            _ = self.backbone(x)
        # get intermediate layer outputs
        for layer, output in zip(self.layers_idxs, self.outputs):  # new
            layer_outputs[layer].append(output.cpu().detach())  # new
        # initialize hook outputs
        self.outputs = []

        if self.training:
            return layer_outputs

        # ---- EVAL INFERENCE ----
        # 2. use the feature maps to get the embeddings
        embedding_vectors = self.raw_feature_maps_to_embeddings(layer_outputs)
        # 3. compute the distance matrix
        if self.diag_cov:
            dist_list = self.compute_distances_diagonal(embedding_vectors)
        else:
            dist_list = self.compute_distances(embedding_vectors)
        # 4. upsample
        score_map = (
            F.interpolate(
                dist_list.unsqueeze(1),
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

        # need to unsqueeze to have (batch, 1, H, W), where 1 is the single channel
        # that represents the anomaly score for each pixel
        score_map = np.expand_dims(score_map, axis=1)

        return score_map, img_scores

    def fit_multivariate_diagonal_gaussian(self, embedding_vectors: torch.Tensor, update_params: bool, logger=None) -> (torch.Tensor, torch.Tensor):
        """
        Fit a multivariate Gaussian distribution to the set of given embedding vectors.

        Returns:
            List of mean and covariance matrix diagonal numpy arrays
        """
        B, C, H, W = embedding_vectors.size()

        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors.cpu(), dim=0).numpy()
        diagonal_cov = torch.zeros(C, H*W).numpy()
        I = np.identity(C)
        # for every "patch" in the feature map, compute the covariance across the batch
        for i in range(H * W):
            # TODO: use np.var instead of np.cov in diagonal covariance computation
            temp_cov = (
                    np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False)
                    + 0.01 * I
            )

            diagonal_cov[:, i] = np.diag(temp_cov)

        if update_params:
            self.gauss_mean, self.diagonal_gauss_cov = mean, diagonal_cov
        return mean, diagonal_cov

    def fit_multivariate_gaussian(self, embedding_vectors, update_params, logger=None):
        """
        Fit a multivariate Gaussian distribution to the set of given embedding vectors.

        Returns:
            List of mean and covariance matrix numpy arrays
        """
        B, C, H, W = embedding_vectors.size()

        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors.cpu(), dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        # for every "patch" in the feature map, compute the covariance across the batch
        for i in range(H * W):
            if self.diag_cov:
                temp_cov = (
                        np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False)
                        + 0.01 * I
                )
                temp_cov[~I.astype(bool)] = 0
                cov[:, :, i] = temp_cov
            else:
                cov[:, :, i] = (
                        np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False)
                        + 0.01 * I
                )
            if logger is not None:
                logger.log({
                    "cov": cov[:, :, i],
                    "mean": mean[:, i], }
                )
        if update_params:
            self.gauss_mean, self.gauss_cov = mean, cov
        return mean, cov

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

    def get_model_savepath(self, save_path):
        return os.path.join(
            save_path,
            f"checkpoints_{self.backbone_model_name}",
            f"train_{self.class_name}.pth.tar" % self.class_name,
        )

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # add all the hyperparameters to the state dict
        for p in self.HYPERPARAMS:
            state_dict[p] = getattr(self, p)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # load the hyperparameters
        for p in self.HYPERPARAMS:
            setattr(self, p, state_dict[p])
        # load the backbone models
        self.load_backbone()
        # remove the hyperparameters from the state dict
        state_dict = {k: v for k, v in state_dict.items() if k not in self.HYPERPARAMS}
        return super().load_state_dict(state_dict, strict=strict)


    def compute_distances(self, embedding_vectors: torch.Tensor):
        """
        Compute the Mahalanobis distances between the embedding vectors and the
        multivariate Gaussian distribution.
        """
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()
        dist_list = []
        assert (
                self.gauss_mean is not None and self.gauss_cov is not None
        ), "The model must be trained first."
        # compute each patch-embedding distance
        for i in range(H * W):
            mean = self.gauss_mean[:, i]
            cov_inv = np.linalg.inv(self.gauss_cov[:, :, i])
            dist = [
                mahalanobis(sample[:, i], mean, cov_inv) for sample in embedding_vectors
            ]
            dist_list.append(dist)
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        return torch.tensor(dist_list)


    def compute_distances_diagonal(self, embedding_vectors: torch.Tensor):
        """
        Compute the Mahalanobis distances between the embedding vectors and the
        multivariate Gaussian distribution.
        """
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()
        dist_list = []
        assert (
                self.gauss_mean is not None and self.diagonal_gauss_cov is not None
        ), "The model must be trained first."
        # compute each patch-embedding distance
        for i in range(H * W):
            mean = self.gauss_mean[:, i]
            diag_cov_i = self.diagonal_gauss_cov[:, i]
            dist = [
                malahanobis_distance_diagonal(sample[:, i], mean, diag_cov_i) for sample in embedding_vectors
            ]
            dist_list.append(dist)
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        return torch.tensor(dist_list)
