"""
Trainer PatchIF
"""

import os
from tqdm import tqdm
import torch

from moviad.models.patchif.patchif import PatchIF
from moviad.trainers.trainer import Trainer, TrainerResult

class TrainerPatchIF(Trainer):

    def __init__(
        self,
        model: PatchIF,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device,
        logger=None,
    ):
        """
        Args:
            device: one of the following strings: 'cpu', 'cuda', 'cuda:0', ...
        """
        super().__init__(
            model,
            train_dataloader,
            test_dataloader,
            device,
            logger
        )

    def train(self):
        print(f"Train PatchIF. Backbone: {self.model.backbone_model_name}")

        self.model.train()

        if self.logger is not None:
            self.logger.watch(self.model)

        # 1. get the feature maps from the backbone
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.model.layers_idxs
        }
        for x in tqdm(
            self.train_dataloader, "| feature extraction | train | %s |" 
        ):
            outputs = self.model(x.to(self.device))
            assert isinstance(outputs, dict)
            for layer, output in outputs.items():
                layer_outputs[layer].extend(output)

        # 2. use the feature maps to get the embeddings → this is the memory bank
        embedding_vectors = self.model.raw_feature_maps_to_embeddings(layer_outputs)

        # 3. Fit the self.ad_model on the memory bank

        # Reshape the embedding vectors to make them 2 dimensional?
        # embedding_vectors = torch.view(-1, embedding_vectors.size(1))
        self.model.ad_model.fit(embedding_vectors.cpu.numpy())

        # 4. Evaluate the model on the training set

        metrics = self.evaluator.evaluate(self.model)

        if self.logger is not None:
            self.logger.log(
                metrics
            )

        print("End training performances:")
        self.print_metrics(metrics)

        return TrainerResult(**metrics)

