# Description: Real-IAD dataset tests
# This file contains unit tests for the Real-IAD dataset, ensuring proper loading, serialization, and data indexing.
import unittest

import torch
from sympy import false
from torchvision.transforms import transforms

from moviad.datasets.builder import DatasetConfig
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadAnomalyClass
from moviad.utilities.configurations import TaskType, Split


IMAGE_SIZE = (224, 224)


class RealIadTrainDatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig("./config.yaml")
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        self.dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK.value,
                                      self.config.realiad_root_path,
                                      self.config.realiad_json_root_path,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TRAIN,
                                      image_size=IMAGE_SIZE,
                                      transform=self.transform)
        self.dataset.load_dataset()

    def test_dataset_is_not_none(self):
        self.assertIsNotNone(self.dataset)

    def test_dataset_should_return_dataset_length(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)
        self.assertIsNotNone(self.dataset.__len__())

    def test_dataset_should_serialize_json(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)

    def test_dataset_should_index_images_and_labels(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)
        self.assertIsNotNone(self.dataset.data)
        self.assertEqual(len(self.dataset.data), len(self.dataset.data.data))

    def test_dataset_should_get_item(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)
        image = self.dataset.__getitem__(0)
        self.assertIsNotNone(image)
        self.assertIs(type(image), torch.Tensor)
        self.assertEqual(image.shape, torch.Size([3, IMAGE_SIZE[0], IMAGE_SIZE[1]]))

    def test_dataset_should_get_item_with_mask(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)
        data, image, mask = self.dataset.__getitem__(0)
        self.assertIsNotNone(data)
        self.assertIsNotNone(image)
        self.assertEqual(image.dtype, torch.float32)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.dtype, torch.float32)


class RealIadTestDatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig("./config.yaml")
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        self.train_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK.value,
                                            self.config.realiad_root_path,
                                            self.config.realiad_json_root_path,
                                            task=TaskType.SEGMENTATION,
                                            split=Split.TRAIN,
                                            image_size=IMAGE_SIZE,
                                            gt_mask_size=IMAGE_SIZE,
                                            transform=self.transform)
        self.train_dataset.load_dataset()

        self.test_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK.value,
                                           self.config.realiad_root_path,
                                           self.config.realiad_json_root_path,
                                           task=TaskType.SEGMENTATION,
                                           split=Split.TEST,
                                           image_size=IMAGE_SIZE,
                                           gt_mask_size=IMAGE_SIZE,
                                           transform=self.transform)
        self.test_dataset.load_dataset()

    def test_dataset_is_not_none(self):
        self.assertIsNotNone(self.train_dataset)

    def test_dataset_should_return_dataset_length(self):
        self.assertIsNotNone(self.train_dataset.data)
        self.assertIsNotNone(self.train_dataset.data.meta)
        self.assertIsNotNone(self.train_dataset.data.data)
        self.assertIsNotNone(self.train_dataset.__len__())

    def test_dataset_should_serialize_json(self):
        self.assertIsNotNone(self.train_dataset.data)
        self.assertIsNotNone(self.train_dataset.data.meta)
        self.assertIsNotNone(self.train_dataset.data.data)

    def test_dataset_should_index_images_and_labels(self):
        self.assertIsNotNone(self.train_dataset.data)
        self.assertIsNotNone(self.train_dataset.data.meta)
        self.assertIsNotNone(self.train_dataset.data.data)
        self.assertIsNotNone(self.train_dataset.data)
        self.assertEqual(len(self.train_dataset.data), len(self.train_dataset.data.data))

    def test_dataset_should_get_item(self):
        self.assertIsNotNone(self.train_dataset.data)
        self.assertIsNotNone(self.train_dataset.data.meta)
        self.assertIsNotNone(self.train_dataset.data.data)
        image, label, mask, path = self.test_dataset.__getitem__(0)
        self.assertIsNotNone(image)
        self.assertIs(type(image), torch.Tensor)
        self.assertEqual(image.dtype, torch.float32)
        self.assertIsNotNone(label)
        self.assertIs(type(label), int)
        self.assertIn(label, [0, 1])  # 0: normal, 1: abnormal
        self.assertIsNotNone(mask)
        self.assertIs(type(mask), torch.Tensor)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertIsNotNone(path)
        self.assertIs(type(path), str)

    def test_training_dataset_should_not_contain_anoamlies(self):
        for item in self.train_dataset.data.data:
            self.assertEqual(item.anomaly_class, RealIadAnomalyClass.OK)

    def test_test_dataset_should_contain_anoamlies(self):
        contains_anomalies = False
        for item in self.test_dataset.data.data:
            if item.anomaly_class != RealIadAnomalyClass.OK:
                contains_anomalies = True
                break
        self.assertTrue(contains_anomalies)

    def test_dataset_is_contaminated(self):
        initial_train_size = self.train_dataset.__len__()
        initial_test_size = self.test_dataset.__len__()
        contamination_size = self.train_dataset.contaminate(self.test_dataset, 0.1)

        contaminated_entries = [entry for entry in self.train_dataset.data.images if
                                entry.anomaly_class != RealIadAnomalyClass.OK]

        self.assertGreater(len(contaminated_entries), 0)
        self.assertGreater(self.train_dataset.__len__(), initial_train_size)
        self.assertLess(self.test_dataset.__len__(), initial_test_size)
        self.assertEqual(contamination_size, abs(initial_train_size - self.train_dataset.__len__()))
        self.assertEqual(contamination_size, abs(initial_test_size - self.test_dataset.__len__()))
        self.assertEqual(contamination_size, len(contaminated_entries))

        contamination_ratio = self.train_dataset.compute_contamination_ratio()
        self.assertLess(contamination_ratio, 1.0)
        self.assertGreater(contamination_ratio, 0.0)


if __name__ == '__main__':
    unittest.main()
