from abc import abstractmethod
from enum import Enum
from dataclasses import dataclass

from torch.utils.data.dataset import Dataset
from moviad.utilities.configurations import TaskType, Split


class IadDataset(Dataset):
    """
    Args:
        split (Split): The split of the dataset (train, val, test).
        dataset_path (str): The path to the dataset.
        contamination_ratio (float): The ratio of contamination in the dataset.
    """

    def __init__(
        self,
        split: Split,
        dataset_path: str,
        contamination_ratio: float,
    ):
        self.split = split
        self.dataset_path = dataset_path
        self.contamination_ratio = contamination_ratio

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def compute_contamination_ratio(self) -> float: ...

    @abstractmethod
    def load_dataset(self): ...

    @abstractmethod
    def contaminate(
        self, source: "IadDataset", ratio: float, seed: int = 42
    ) -> int: ...

    @abstractmethod
    def contains(self, entry) -> bool: ...

    @staticmethod
    def get_argpars_parameters(parser):
        parser.add_argument(
            "--normalize_dataset",
            action="store_true",
            help="If the dataset needs to be normalized to ImageNet mean and std",
        )
        parser.add_argument(
            "--batch_size", type=int, help="Batch size for the dataloader"
        )
        return parser
