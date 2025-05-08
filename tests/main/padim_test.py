import unittest
import unittest

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode

from moviad.datasets.builder import DatasetConfig
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.entrypoints.padim import PadimArgs
from moviad.models.padim.padim import Padim
from moviad.profiler.pytorch_profiler import Profiler
from moviad.trainers.trainer_padim import TrainerPadim
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator
from tests.datasets.realiaddataset_test import IMAGE_SIZE

profiler = Profiler()


class PadimTrainTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig("./config.json")
        self.args = PadimArgs()
        self.args.backbone = "mobilenet_v2"
        self.args.ad_layers = ["features.4", "features.7", "features.10"]
        self.args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.PILToTensor(),
                transforms.Resize(
                    IMAGE_SIZE,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def test_padim_with_diagonalization(self):
        profiler.start_profiling(True, "Padim with diagonalization, MVTec dataset")
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            "pill",
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            "pill",
            Split.TEST,
            img_size=IMAGE_SIZE,
            gt_mask_size=IMAGE_SIZE,
        )

        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            drop_last=True,
        )

        # evaluate the model
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True
        )

        padim = Padim(
            self.args.backbone,
            self.args.category,
            device=self.args.device,
            diag_cov=True,
            layers_idxs=self.args.ad_layers,
        )
        padim.to(self.args.device)
        trainer = TrainerPadim(
            model=padim,
            train_dataloader=train_dataloader,
            eval_dataloader=test_dataloader,
            device=self.args.device,
            apply_diagonalization=True,
        )

        evaluator = Evaluator(test_dataloader=test_dataloader, device=self.args.device)

        with profiler.profile_step():
            trainer.train()
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = (
                evaluator.evaluate(padim)
            )

        print("Evaluation performances:")
        print(
            f"""
                img_roc: {img_roc}
                pxl_roc: {pxl_roc}
                f1_img: {f1_img}
                f1_pxl: {f1_pxl}
                img_pr: {img_pr}
                pxl_pr: {pxl_pr}
                pxl_pro: {pxl_pro}
                """
        )
        profiler.end_profiling()

    def test_padim_without_diagonalization(self):
        profiler.start_profiling(True, "Padim without diagonalization, MVTec dataset")
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            "pill",
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            "pill",
            Split.TEST,
            img_size=IMAGE_SIZE,
            gt_mask_size=IMAGE_SIZE,
        )

        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            drop_last=True,
        )

        # evaluate the model
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True
        )

        padim = Padim(
            self.args.backbone,
            self.args.category,
            device=self.args.device,
            diag_cov=self.args.diagonal_convergence,
            layers_idxs=self.args.ad_layers,
        )
        padim.to(self.args.device)

        trainer = TrainerPadim(
            model=padim,
            train_dataloader=train_dataloader,
            eval_dataloader=test_dataloader,
            device=self.args.device,
            apply_diagonalization=False,
        )
        evaluator = Evaluator(test_dataloader=test_dataloader, device=self.args.device)
        with profiler.profile_step():
            trainer.train()
            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = (
                evaluator.evaluate(padim)
            )

        profiler.end_profiling()

        print("Evaluation performances:")
        print(
            f"""
                img_roc: {img_roc}
                pxl_roc: {pxl_roc}
                f1_img: {f1_img}
                f1_pxl: {f1_pxl}
                img_pr: {img_pr}
                pxl_pr: {pxl_pr}
                pxl_pro: {pxl_pro}
                """
        )


if __name__ == "__main__":
    unittest.main()
