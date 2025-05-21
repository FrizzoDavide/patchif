from __future__ import annotations

from typing import Callable
from tqdm import tqdm
import torch
import numpy as np

from .metrics import MetricLvl, Metric


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def append(prev, new, dtype=None, to_numpy=True):
    new = new.cpu().numpy() if to_numpy else new
    new = new.astype(dtype) if dtype else new
    return np.concatenate((prev, new), axis=0)


class Evaluator:
    """
    This class will evaluate the trained model on the test set
    and it will produce the evaluation metrics needed

    Args:
        test_dataloader (Dataloader): test dataloader
        device (torch.device): device where to run the model
    """

    def __init__(self, dataloader, metrics: list[Metric], device):
        """
        Args:
            dataloader (Dataloader): dataloader on which to compute the metrics
            device (torch.device): device where to run the model
        """
        self.dataloader = dataloader
        self.metrics = metrics
        self.device = device

    def evaluate(self, model, postprocess: Callable = min_max_norm):
        """
        Args:
            model: a model object on which you can call model.predict(batched_images)
                and returns a tuple of anomaly_maps and anomaly_scores
            output_path (str): path where to store the output masks
        """
        model.eval()

        # Initialize results as numpy arrays
        init = lambda *t: (np.empty((0,), dtype=t_) for t_ in t)
        gt_mask, gt_label, pred_anom_map, pred_anom_score = init(int, *(float,) * 3)

        for image, label, mask, path in tqdm(self.dataloader, desc="Eval"):
            with torch.no_grad():  # get anomaly map and score
                anom_maps, anom_scores = model(image.to(self.device))

            # Append ground truth, anomaly scores, and predicted masks
            gt_mask = append(gt_mask, mask, dtype=int)
            gt_label = append(gt_label, label)
            pred_anom_map = append(pred_anom_map, anom_maps)
            pred_anom_score = append(pred_anom_score, anom_scores)

        pred_anom_map = postprocess(pred_anom_map)

        # TODO: Implement using generic metrics
        """
        report = []
        for metric in metrics:
            if metric.level == MetricLvl.IMAGE:
                pred = anom_score
                gt = labels
            elif metric.level == MetricLvl.PIXEL:
                pred = anom_maps
                gt = masks
            report.append({metric.name:metric.compute(gt, pred)})
        """


'''
TODO: remove this
def append_results(
    output_path: Union[str, os.PathLike],
    category: str,
    seed: Optional[int],
    img_roc_auc: float,
    per_pixel_rocauc: float,
    f1_img: float,
    f1_pxl: float,
    pr_auc_img: float,
    pr_auc_pxl: float,
    au_pro_pxl: float,
    ad_model: str,
    feature_layers: str,
    backbone: str,
    weights: Optional[str],
    bootstrap_layer: Optional[int],
    epochs: Optional[int],
    input_img_size: Optional[tuple[int, int]],
    output_img_size: Optional[tuple[int, int]],
):
    """
    Save the results of the evaluation in a file
    """
    df = pd.DataFrame(
        {
            "category": [category],
            "seed": [seed],
            "img_roc_auc": [img_roc_auc],
            "per_pixel_rocauc": [per_pixel_rocauc],
            "f1_img": [f1_img],
            "f1_pxl": [f1_pxl],
            "pr_auc_img": [pr_auc_img],
            "pr_auc_pxl": [pr_auc_pxl],
            "au_pro_pxl": [au_pro_pxl],
            "ad_model": [ad_model],
            "feature_layers": [feature_layers],
            "backbone": [backbone],
            "weights": [weights],
            "eval_datetime": [pd.Timestamp.now()],
            "bootstrap_layer": [bootstrap_layer],
            "epochs": [epochs],
        }
    )
    if os.path.isfile(output_path):
        old_df = pd.read_csv(output_path)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(output_path, index=False)
'''
