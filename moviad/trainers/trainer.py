from __future__ import annotations
import torch
from typing import Any, Callable

from moviad.utilities.evaluation.evaluator import Evaluator


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader | None,
        device: torch.device,
        logger: Any,
        save_path: str | None = None,
        saving_criteria: Callable | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.logger = logger
        self.save_path = save_path
        self.saving_criteria = saving_criteria
        self.evaluator = Evaluator(self.eval_dataloader, self.device)

    @staticmethod
    def update_best_metrics(best_metrics, metrics):
        for m in metrics.keys():
            best_metrics[m] = max(best_metrics[m], metrics[m])
            
        return best_metrics

    @staticmethod
    def print_metrics(metrics):
        print("\n".join([f"{k}: {v}" for k, v in metrics.items()]))

    def train(self, epochs: int, evaluation_epoch_interval: int):
        pass


class TrainerResult:
    # TODO: REMOVE HARD-CODED METRICS
    img_roc_auc: float
    pxl_roc_auc: float
    img_f1: float
    pxl_f1: float
    img_pr_auc: float
    pxl_pr_auc: float
    pxl_au_pro: float

    def __init__(
        self,
        img_roc_auc,
        pxl_roc_auc,
        img_f1,
        pxl_f1,
        img_pr_auc,
        pxl_pr_auc,
        pxl_au_pro,
    ):
        self.img_roc_auc = img_roc_auc
        self.pxl_roc_auc = pxl_roc_auc
        self.img_f1 = img_f1
        self.pxl_f1 = pxl_f1
        self.img_pr_auc = img_pr_auc
        self.pxl_pr_auc = pxl_pr_auc
        self.pxl_au_pro = pxl_au_pro
