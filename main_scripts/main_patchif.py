"""
Python script to launch experiments on the PatchIF dataset.
"""

import os
import random
from datetime import datetime
import ipdb
import argparse
import setproctitle

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from moviad.models.patchif.patchif import PatchIF
from moviad.trainers.trainer_patchif import TrainerPatchIF
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset, CATEGORIES
from moviad.utilities.evaluator import Evaluator, append_results
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.exp_configurations import DATASET_PATHS, AD_LAYERS
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor, TORCH_BACKBONES, OTHERS_BACKBONES
from moviad.utilities.manage_files import save_element, generate_path, get_most_recent_file

BATCH_SIZE = 8
IMAGE_INPUT_SIZE = (224, 224)
OUTPUT_SIZE = (224, 224)

def main(args):

    assert [category in CATEGORIES for category in args.categories], f"Some categories are not valid. Available categories: {CATEGORIES}"

    device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print('#'* 50)
    print("--- PatchIF Experiment ---")
    print(f"Backbone model: {args.backbone_model_name}")
    print(f"AD layers: {AD_LAYERS[args.backbone_model_name]}")
    print(f"AD model: {args.ad_model_name}")
    print(f"Number of estimators: {args.n_estimators}")
    print(f"device: {device}")
    print(f"Contamination ratio: {args.contamination_ratio}")
    print(f"Contamination ratio type: {type(args.contamination_ratio)}")
    print('#'* 50)

    # Set paths

    model_dirpath = generate_path(
        basepath = os.getcwd(),
        folders = [
            "models_state_dict",
            args.dataset_name,
        ]
    )

    results_dirpath = generate_path(
        basepath = os.getcwd(),
        folders = [
            "patchif_results",
            args.dataset_name,
        ]
    )

    if args.exp_name == "":
        if args.contaminate:
            exp_name = f"PatchIF_{args.ad_model_name}_{args.backbone_model_name}_n_estimators_{args.n_estimators}"
        else:
            exp_name = f"PatchIF_{args.ad_model_name}_{args.backbone_model_name}_n_estimators_{args.n_estimators}_contamination_{args.contamination_ratio}"
    else:
        exp_name = args.exp_name

    print('#'* 50)
    print(f"Starting experiment: {exp_name}")
    print('#'* 50)

    for seed in args.seeds:

        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print('#'* 50)
        print(f"Experiment for seed: {seed}")
        print('#'* 50)

        for category in args.categories:

            print('#'* 50)
            print(f"Experiment for category: {category}")
            print('#'* 50)


            exp_name = f"{exp_name}_{category}_seed_{seed}"

            setproctitle.setproctitle(exp_name)

            if args.train:

                print("---- PatchIF Training ----")

                print('#'* 50)
                print(f"Defining training dataset MVTecDataset category {category}")
                print('#'* 50)

                train_dataset = MVTecDataset(
                    task = TaskType.SEGMENTATION,
                    root = DATASET_PATHS[args.dataset_name],
                    category = category,
                    split = Split.TRAIN,
                    norm = True,
                    img_size = IMAGE_INPUT_SIZE,
                    gt_mask_size = None,
                    preload_imgs = True
                )

                train_dataset.load_dataset()
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    pin_memory=True
                )

                print('#'* 50)
                print(f"Defining test dataset MVTecDataset category {category}")
                print('#'* 50)

                test_dataset = MVTecDataset(
                    task = TaskType.SEGMENTATION,
                    root = DATASET_PATHS[args.dataset_name],
                    category = category,
                    split = Split.TEST,
                    norm = True,
                    img_size = IMAGE_INPUT_SIZE,
                    gt_mask_size = None,
                    preload_imgs = True
                )

                test_dataset.load_dataset()
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size = BATCH_SIZE,
                    shuffle = True,
                    pin_memory = True
                )

                if args.contaminate:

                    print('#'* 50)
                    print(f"Contaminating the train dataset with contamination ratio {args.contamination_ratio}")
                    print('#'* 50)

                    _ = train_dataset.contaminate(
                        source = test_dataset,
                        ratio = args.contamination_ratio,
                        seed = seed
                    )

                print('#'* 50)
                print(f"-- TRAIN DATASET INFORMATION --")
                print(f"Length: {len(train_dataset)}")
                print(f"Contamination ratio: {train_dataset.compute_contamination_ratio()}")
                print('#'* 50)

                print('#'* 50)
                print(f"-- TEST DATASET INFORMATION --")
                print(f"Length: {len(test_dataset)}")
                print(f"Contamination ratio: {test_dataset.compute_contamination_ratio()}")
                print('#'* 50)

                print('#'* 50)
                print("Defining PatchIF model")
                print('#'* 50)

                model = PatchIF(
                    backbone_model_name = args.backbone_model_name,
                    layers_idxs = AD_LAYERS[args.backbone_model_name],
                    ad_model_type = args.ad_model_name,
                    n_estimators = args.n_estimators,
                    plus = args.plus,
                    device = device
                )

                print('#'* 50)
                print(f"Model name: {model.name}")
                print('#'* 50)

                print('#'* 50 )
                print("Defining PatchIF trainer")
                print('#'* 50 )

                trainer = TrainerPatchIF(
                    model = model,
                    train_dataloader = train_loader,
                    test_dataloader = test_loader,
                    device = device,
                    logger = None,
                )

                # Set paths
                model_dirpath_seed = generate_path(
                    basepath = model_dirpath,
                    folders = [
                        category,
                        model.name,
                        args.backbone,
                        exp_name,
                        f"seed_{seed}"
                    ]
                )

                print('#'* 50)
                print(f"Training the {args.model_name} model")
                print('#'* 50)
                trainer.train()

                if args.save_model:

                    #TODO: Placeholder → here we have to see how to save the model state dict in a pth file or similar with the ctypes trees objects
                    save_element(
                        element = model.state_dict(),
                        dirpath = model_dirpath_seed,
                        filename = f"{model.name}_{args.backbone_model_name}_{args.category}_state_dict_seed_{seed}.pth",
                        filetype = "pth"
                    )

            if args.test:

                print("---- PatchIF test ----")

                if not args.train:

                    model = PatchIF(
                        backbone_model_name = args.backbone_model_name,
                        layers_idxs = AD_LAYERS[args.backbone_model_name],
                        ad_model_type = args.ad_model_name,
                        n_estimators = args.n_estimators,
                        plus = args.plus,
                        device = device
                    )

                    print('#'* 50)
                    print("Loading the model from the state dict")
                    print('#'* 50)

                    #TODO: Now model_dirpath_seed is defined both in args.train
                    # and in args.test, for the future find a way to define it just one time
                    model_dirpath_seed = generate_path(
                        basepath = model_dirpath,
                        folders = [
                            category,
                            model.name,
                            args.backbone,
                            exp_name,
                            f"seed_{seed}"
                        ]
                    )

                    results_dirpath_seed = generate_path(
                        basepath = results_dirpath,
                        folders = [
                            category,
                            model.name,
                            args.backbone,
                            exp_name,
                            f"seed_{seed}"
                        ]
                    )

                    model_path = get_most_recent_file(model_dirpath_seed, file_pos = args.file_pos)

                    #TODO: Placeholder → here we have to see how to save the model state dict in a pth file or similar with the ctypes trees objects
                    model.load_state_dict(
                        torch.load(model_path, map_location=device, weights_only=False), strict=False
                    )
                    model.to(device)
                    print(f"Loaded model from model_path: {model_path}")

                model.eval()

                test_dataset = MVTecDataset(
                    task = TaskType.SEGMENTATION,
                    root = DATASET_PATHS[args.dataset_name],
                    category = category,
                    split = Split.TEST,
                    norm = True,
                    img_size = IMAGE_INPUT_SIZE,
                    gt_mask_size = None,
                    preload_imgs = True
                )

                test_dataset.load_dataset()
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size = BATCH_SIZE,
                    shuffle = True,
                    pin_memory = True
                )

                print('#'* 50)
                print("Evaluating the model on the test set")
                print('#'* 50)

                evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
                scores = evaluator.evaluate(padim)

                metrics_path = os.path.join(results_dirpath_seed, f"{model.name}_{args.backbone_model_name}_{category}_metrics_seed_{seed}.csv")

                print('#'* 50)
                print("Saving metrics to file")
                print('#'* 50)

                append_results(
                    output_path = metrics_path,
                    category = category,
                    seed = seed,
                    *scores,
                    ad_model = "padim",  # ad_model
                    feature_layers = AD_LAYERS[args.backbone_model_name],
                    backbone = args.backbone_model_name,
                    weights = "IMAGENET1K_V2",  # NOTE: hardcoded, should be changed
                    bootstrap_layer = None,
                    epochs = -1,  # epochs (not used)
                    input_img_size = IMAGE_INPUT_SIZE,
                    output_img_size = OUTPUT_SIZE,
                )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PatchIF Experiment")
    # experiment parameters
    parser.add_argument("--train", action="store_true",help="Run training")
    parser.add_argument("--test", action="store_true",help="Run testing")
    parser.add_argument("--debug", action="store_true",help="Run debug")
    parser.add_argument("--save_figures", action="store_true",help="Save figures")
    parser.add_argument("--save_model", action="store_true",help="Save model")
    parser.add_argument("--save_logs", action="store_true",help="Save logs")
    parser.add_argument("--exp_name", type=str , default="", help="Experiment name")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--results_dirpath", type=str, default=None)
    parser.add_argument("--device_num", type=int, default=0, help="cuda device number")
    # model parameters
    parser.add_argument("--backbone_model_name",type=str,help="Available backbones: resnet18, wide_resnet50_2, mobilenet_v2, mcunet-in3")
    parser.add_argument("--ad_model_name", type=str, default="eif", help="Type of AD model, eif or if")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators for the IF/EIF model, for patchif")
    parser.add_argument("--plus", action='store_true', help="Flag to use the EIF or EIF+ AD model")
    # dataset parameters
    parser.add_argument("--categories", type=str, nargs="+", default=CATEGORIES)
    parser.add_argument("--contaminate", action="store_true", help="Contaminate the dataset with anomalies")
    parser.add_argument("--contamination_ratio", type=float, default=0.1, help="Contamination ratio for the dataset")

    args = parser.parse_args()


    log_filename = "padim.log"
    s = "DEBUG " if args.debug else ""

    try:
        main(args)

        if args.save_logs:
            # create a log file if it does not exist
            if not os.path.exists(log_filename):
                with open(log_filename, "w") as f:
                    f.write("")
            # write the args as a string to the log file
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_filename, "a") as f:
                f.write(s + "finished " + "\t" + now_str + "\t" + str(args) + "\n")

    except Exception as e:
        if args.save_logs:
            # write the args as a string to the log file
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_filename, "a") as f:
                f.write(s + "** FAILED **" + "\t" + now_str + "\t" + str(args) + "\n")
        raise e

