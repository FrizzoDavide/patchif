"""
Trainer PatchIF
"""

import os
import ipdb
from tqdm import tqdm
import torch
from random import sample

from moviad.models.patchif.patchif import PatchIF
from moviad.trainers.trainer import Trainer, TrainerResult
from moviad.utilities.manage_files import save_element

#TODO: Start setting up paths to save the memory bank somewhere
pwd = os.getcwd()
patchif_results_path = os.path.join(pwd,"patchif_results")

class TrainerPatchIF(Trainer):

    def __init__(
        self,
        model: PatchIF,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        logger=None,
        save_memory_bank: bool = False,
        memory_bank_path: str = pwd,
        dataset_name: str = "mvtec",
        category: str = "pill",
        contaminate: bool = False,
        contaminate_ratio: float = 0.1,
    ):
        """
        Constructur TrainerPatchIF

        Args:
            model: PatchIF model to train
            train_dataloader: DataLoader for the training set
            test_dataloader: DataLoader for the test set
            device: one of the following strings: 'cpu', 'cuda', 'cuda:0', ...
            logger: logger to log the training process
            save_memory_bank: boolean to indicate if the memory bank should be saved
            memory_bank_path: path to save the memory bank
            dataset_name: name of the dataset
            category: category of the dataset
            contaminate: boolean to indicate if the dataset should be contaminated
            contaminate_ratio: ratio of contamination to apply to the dataset
        """
        super().__init__(
            model,
            train_dataloader,
            test_dataloader,
            device,
            logger
        )

        self.save_memory_bank = save_memory_bank
        self.memory_bank_path = memory_bank_path
        self.dataset_name = dataset_name
        self.category = category
        self.contaminate = contaminate
        self.contaminate_ratio = contaminate_ratio

    def train(self):

        print('#'* 50)
        print(f"Train PatchIF. Backbone: {self.model.backbone_model_name}")
        print('#'* 50)

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

        #NOTE: Comment this line → try an experiment with a memory bank shaped as:
        # (B, C, H, W)
        # Reshape the embedding vectors to make them 2 dimensional?
        # embedding_vectors = embedding_vectors.view(-1, embedding_vectors.size(1))

        #NOTE: Random subsampling the rows of the memory bank
        # Sample a random subset of the row indexes of the memory bank
        if self.model.subsample_ratio == 1.0:
            memory_bank = embedding_vectors
            print('#'* 50)
            print("Memory bank is not subsampled")
            print('#'* 50)
        else:
            print('#'* 50)
            print(f"Subsampling memory bank with ratio: {self.model.subsample_ratio}")
            print('#'* 50)
            subsample_size = int(embedding_vectors.shape[0]*(1 - self.model.subsample_ratio))
            random_rows = torch.tensor(sample(range(embedding_vectors.shape[0]),subsample_size))
            print("First elements of random rows:")
            print(random_rows[:10])
            memory_bank = torch.index_select(
                embedding_vectors, 0, random_rows.to(embedding_vectors.device)
            )

        # Save the memory bank to a file
        if self.save_memory_bank:
            if self.contaminate:
                filename = f"memory_bank_{self.model.backbone_model_name}_{self.dataset_name}_{self.category}_contamination_{self.contaminate_ratio}"
            else:
                filename = f"memory_bank_{self.model.backbone_model_name}_{self.dataset_name}_{self.category}"

            memory_bank_dict = {
                "memory_bank": memory_bank.cpu().numpy(),
            }

            save_element(
                element = memory_bank_dict,
                dirpath = self.memory_bank_path,
                filename = filename,
                filetype = "pickle",
                no_time = True
            )

            print('#'* 50)
            print(f"Memory bank saved to {self.memory_bank_path}/{filename}.pickle")
            print('#'* 50)
            ipdb.set_trace()

        print('#'* 50)
        print(f"Fitting {self.model.ad_model.name} model on the memory bank")
        print(f"Memory bank shape: {memory_bank.shape}")
        print('#'* 50)

        self.model.ad_model.fit(memory_bank.cpu().numpy())
        # Set the trees attribute of PatchIF to the trees attribute of the ad_model
        self.model.trees = self.model.ad_model.trees

        # 4. Evaluate the model on the training set

        metrics = self.evaluator.evaluate(self.model)

        if self.logger is not None:
            self.logger.log(
                metrics
            )

        print("End training performances:")
        self.print_metrics(metrics)

        return TrainerResult(**metrics)

