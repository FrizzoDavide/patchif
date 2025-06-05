import math
from enum import Enum
from typing import Optional, Union

from numpy.ma.core import indices
from torchvision.transforms.functional import InterpolationMode

from pathlib import Path
import os
import glob
import ipdb

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from moviad.backbones.micronet.utils import compute_mask_contamination
from moviad.datasets.iad_dataset import IadDataset
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.utilities.configurations import TaskType, Split, LabelName

IMG_EXTENSIONS = (".png", ".PNG")

CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


class MvtecClassEnum(Enum):
    BOTTLE = "bottle"
    CABLE = "cable"
    CAPSULE = "capsule"
    CARPET = "carpet"
    GRID = "grid"
    HAZELNUT = "hazelnut"
    LEATHER = "leather"
    METAL_NUT = "metal_nut"
    PILL = "pill"
    SCREW = "screw"
    TILE = "tile"
    TOOTHBRUSH = "toothbrush"
    TRANSISTOR = "transistor"
    WOOD = "wood"
    ZIPPER = "zipper"


IMG_SIZE = (3, 900, 900)

"""Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
"""


class MVTecDataset(IadDataset):
    """MVTec dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/MVTec``.
        category (str): Sub-category of the dataset, e.g. 'bottle'
            Defaults to ``bottle``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``

    """

    def __init__(
            self,
            task: TaskType,
            root: str,
            category: str,
            split: Split,
            norm: bool = True,
            img_size=(224, 224),
            gt_mask_size: Optional[tuple] = None,
            preload_imgs: bool = True,
    ) -> None:
        super(MVTecDataset)

        gt_mask_size = img_size if gt_mask_size is None else gt_mask_size

        self.img_size = img_size
        self.gt_mask_size = gt_mask_size

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples: pd.DataFrame = None
        self.preload_imgs = preload_imgs

        if norm:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        else:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
            ]

        self.transform_image = transforms.Compose(t_list)

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    gt_mask_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

    def compute_contamination_ratio(self) -> float:

        """
        Compute the contamination ratio of the dataset.
        """

        if self.samples is None:
            raise ValueError("Dataset is not loaded")

        contaminated_samples = self.samples[self.samples["label_index"] == LabelName.ABNORMAL.value]
        if contaminated_samples.empty:
            return 0

        total_contamination_ratio = 0

        for _, row in contaminated_samples.iterrows():
            if not Path(row["mask_path"]).exists():
                raise ValueError("Mask file does not exist")

            mask = Image.open(row["mask_path"]).convert("L")
            mask = self.transform_mask(mask)
            total_contamination_ratio += compute_mask_contamination(mask)

        return total_contamination_ratio / len(contaminated_samples)

    def is_loaded(self) -> bool:
        return self.samples is not None

    def contains(self, item) -> bool:
        return self.samples['image_path'].eq(item['image_path']).any()

    def load_dataset(self):
        if self.is_loaded():
            print("Dataset already loaded")
            return

        root = Path(self.root_category)

        #NOTE: This creates a list of tuples where each tuple is composed by:
        # (root, split, label, image_path) where:
        # - root is the root directory of the dataset
        # - split is the split of the dataset (e.g. train, test, ground_truth)
        # - label is the label of the image:
            # - in `train` we have only `good` 
            # - in  `test` there are several labels: ['color', 'combined', 'contamination', 'crack', 'faulty_imprint', 'good', 'pill_type', 'scratch']
            # - in `ground_truth` we have the same labels as in `test` except for `good` (where the ground truth is not needed, it will be a all zero mask)
            # - image_path is the path to the image file → at this stage it is just the `file_name.png`
        samples_list = [
            (str(root),) + f.parts[-3:]
            for f in root.glob(r"**/*")
            if f.suffix in IMG_EXTENSIONS
        ]

        if not samples_list:
            msg = f"Found 0 images in {root}"
            raise RuntimeError(msg)

        # Create a DataFrame from the list of tuples
        samples = pd.DataFrame(
            samples_list, columns=["path", "split", "label", "image_path"]
        )

        # Modify image_path column by converting to absolute path
        samples["image_path"] = (
                samples.path
                + "/"
                + samples.split
                + "/"
                + samples.label
                + "/"
                + samples.image_path
        )

        # Create label index for normal (0) and anomalous (1) images.
        #NOTE: From what it's written here I understand that images with label `good`
        # are the normal images, while the others are the different typeof anomalous images
        samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
        samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
        samples.label_index = samples.label_index.astype(int)

        if self.split == Split.TEST:

            # separate masks from samples
            mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )
            samples = samples[samples.split != "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )

            # assign mask paths to anomalous test images
            samples["mask_path"] = ""
            samples.loc[
                (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
                "mask_path",
            ] = mask_samples.image_path.to_numpy()

            # assert that the right mask files are associated with the right test images
            abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
            if (
                    len(abnormal_samples)
                    and not abnormal_samples.apply(
                lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1
            ).all()
            ):
                msg = """Mismatch between anomalous images and ground truth masks. Make sure t
                he mask files in 'ground_truth' folder follow the same naming convention as the
                anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
                raise Exception(msg)

        self.samples = samples[samples.split == self.split].reset_index(drop=True)
        if self.preload_imgs:
            self.data = [
                self.transform_image(
                    Image.open(self.samples.iloc[index].image_path).convert("RGB")
                )
                for index in range(len(self.samples))
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def contaminate(
        self,
        source: 'IadDataset',
        ratio: float,
        seed: int = 42
    ) -> int:

        """
        Create a contaminated version of the dataset by adding anomalies so that the contamination ratio is equal to the specified ratio.

        Args:
            source (IadDataset): Source dataset from which anomalies are taken.
            ratio (float): Contamination ratio to be achieved in the destination dataset.
            seed (int): Random seed for reproducibility. Defaults to 42.

        Returns:
            contamination_set_size (int): Number of anomalous samples added to the input dataset
        """

        if type(source) != MVTecDataset:
            raise ValueError("Dataset should be of type MVTecDataset")
        if self.samples is None:
            raise ValueError("Destination dataset is not loaded")

        torch.manual_seed(seed)
        contamination_set_size = int(math.floor(len(self.samples) * ratio))
        contaminated_entries_indices = source.samples[source.samples["label_index"] == LabelName.ABNORMAL.value].index
        if len(contaminated_entries_indices) < contamination_set_size:
            raise DatasetTooSmallToContaminateException(
                f"Source dataset does not contain enough abnormal entries to contaminate the destination dataset. "
                f"Source dataset contains {len(contaminated_entries_indices)} abnormal entries, "
                f"while {contamination_set_size} are required."
            )

        #TODO: Maybe here we should use a stratified sampling in order to avoid biased sampling
        # towards a specific type of anomaly?
        np.random.seed(seed)
        contaminated_entries_indices = np.random.choice(
            contaminated_entries_indices,
            contamination_set_size,
            replace=False
        )

        for index in contaminated_entries_indices:
            entry_metadata = source.samples.iloc[index]
            if source.preload_imgs:
                entry = source.data[index]
                self.data.append(entry)
            else:
                entry = self.transform_image(
                    Image.open(self.samples.iloc[index].image_path).convert("RGB")
                )
                self.data.append(entry)
                source.data = [e for e in source.data if hash(e) != hash(entry)]

            self.samples = pd.concat([self.samples, pd.DataFrame([entry_metadata])], ignore_index=True)
            index_label = source.samples.index[index]

        source.samples = source.samples.drop(contaminated_entries_indices).reset_index(drop=True)
        source.data = [source.data[i] for i in range(len(source.data)) if i not in contaminated_entries_indices]
        return contamination_set_size

    def __getitem__(self, index: int):
        """
        Args:
            index (int) : index of the element to be returned

        Returns:
            image (Tensor) : tensor of shape (C,H,W) with values in [0,1]
            label (int) : label of the image
            anomaly_label (str) : type of anomaly
            mask (Tensor) : tensor of shape (1,H,W) with values in [0,1]
            path (str) : path of the input image
        """

        # open the image and get the tensor
        if self.preload_imgs:
            image = self.data[index]
        else:
            image = self.transform_image(
                Image.open(self.samples.iloc[index].image_path).convert("RGB")
            )

        if self.split == Split.TRAIN:
            return image
        else:
            # return also the label, the mask and the path
            label = self.samples.iloc[index].label_index
            anomaly_label = self.samples.iloc[index].label
            path = self.samples.iloc[index].image_path

            if label == LabelName.ABNORMAL:
                mask = Image.open(self.samples.iloc[index].mask_path).convert("L")
                mask = self.transform_mask(mask)
            else:
                mask = torch.zeros(1, *self.gt_mask_size)

            return image, label, anomaly_label, mask.int(), path

# Method to load the test and training data
def load_train_test_data(
    task: TaskType,
    root: str,
    category: str,
    train_split: Split = Split.TRAIN,
    test_split: Split = Split.TEST,
    norm: bool = True,
    img_size: tuple[int, int] = (224, 224),
    gt_mask_size: Optional[tuple[int, int]] = None,
    preload_imgs: bool = True,
    batch_size: int = 32,
    return_loaders: bool = False
) -> Union[
        tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.Dataset, torch.utils.data.DataLoader], # return_loaders = True
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset] # return_loaders = False
    ]:

    """
    Load the training and test data from MVTec AD and return the dataloaders

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        root (str): Path to the root of the dataset.
        category (str): Sub-category of the dataset, e.g. 'bottle'.
        train_split (Split): Split for the training data.
        test_split (Split): Split for the test data.
        norm (bool): Whether to normalize the images.
        img_size (tuple[int, int]): Size of the input images.
        gt_mask_size (Optional[tuple[int, int]]): Size of the ground truth masks.
        preload_imgs (bool): Whether to preload images into memory.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
            The training and test dataset in case `return_loaders` is False.
        tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.Dataset, torch.utils.data.DataLoader]:
            The training and test dataset and their respective dataloaders in case `return_loaders` is True.
    """

    print('#'* 50)
    print(f"Defining training dataset MVTecDataset category {category}")
    print('#'* 50)

    train_dataset = MVTecDataset(
        task = task,
        root = root,
        category = category,
        split = train_split,
        norm = norm,
        img_size = img_size,
        gt_mask_size = gt_mask_size,
        preload_imgs = preload_imgs
    )

    train_dataset.load_dataset()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True
    )

    test_dataset = MVTecDataset(
        task = task,
        root = root,
        category = category,
        split = test_split,
        norm = norm,
        img_size = img_size,
        gt_mask_size = gt_mask_size,
        preload_imgs = preload_imgs
    )

    print('#'* 50)
    print(f"Defining test dataset MVTecDataset category {category}")
    print('#'* 50)

    test_dataset.load_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True
    )

    if return_loaders:
        return train_dataset, train_loader, test_dataset, test_loader

    return train_dataset, test_dataset


# Method to load the training and test dataset and contaminate them

def load_contaminate_train_test_data(
    task: TaskType,
    root: str,
    category: str,
    train_split: Split,
    test_split: Split,
    norm: bool = True,
    img_size: tuple[int, int] = (224, 224),
    gt_mask_size: Optional[tuple[int, int]] = None,
    preload_imgs: bool = True,
    batch_size: int = 32,
    contamination_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.Dataset, torch.utils.data.DataLoader, int]:

    """
    Load the training and test data from MVTec AD and contaminate the training data
    with anomalies from the test data.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        root (str): Path to the root of the dataset.
        category (str): Sub-category of the dataset, e.g. 'bottle'.
        train_split (Split): Split for the training data.
        test_split (Split): Split for the test data.
        norm (bool): Whether to normalize the images.
        img_size (tuple[int, int]): Size of the input images.
        gt_mask_size (Optional[tuple[int, int]]): Size of the ground truth masks.
        preload_imgs (bool): Whether to preload images into memory.
        batch_size (int): Batch size for the dataloaders.
        contamination_ratio (float): Ratio of anomalies to add to the training set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.Dataset, torch.utils.data.DataLoader, int]:
            The training dataset, training dataloader, test dataset, test dataloader and the size of the contamination set.
    """

    train_dataset, test_dataset = load_train_test_data(
        task = task,
        root = root,
        category = category,
        train_split = train_split,
        test_split = test_split,
        norm = norm,
        img_size = img_size,
        gt_mask_size = gt_mask_size,
        preload_imgs = preload_imgs,
        batch_size = batch_size,
        return_loaders = False
    )

    contamination_set_size = train_dataset.contaminate(
        source = test_dataset,
        ratio = contamination_ratio,
        seed = seed
    )

    #NOTE: After the application of the contamination method, the train_dataset
    # and test_dataset objects are updated with the contaminated data and so we need to
    # call again the load_dataset method and re create the dataloaders

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True
    )

    return train_dataset, train_loader, test_dataset, test_loader, contamination_set_size
