"""
Python script to launch experiments on the PatchIF dataset.
"""

import os
import random
from datetime import datetime
import copy
import ipdb
import argparse
import setproctitle
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from moviad.models.patchif.patchif import PatchIF
from moviad.trainers.trainer_patchif import TrainerPatchIF
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset, CATEGORIES, load_train_test_data, load_contaminate_train_test_data
from moviad.utilities.evaluator import Evaluator, append_results
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.exp_configurations import DATASET_PATHS, AD_LAYERS, set_exp_seed
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor, TORCH_BACKBONES, OTHERS_BACKBONES
from moviad.utilities.manage_files import get_current_time, save_element, generate_path, get_most_recent_file, open_element

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
    print(f"Max nodes: {args.max_nodes}")
    print(f"device: {device}")
    print(f"Contamination ratio: {args.contamination_ratio}")
    print(f"Contamination ratio type: {type(args.contamination_ratio)}")
    print('#'* 50)

    # Set paths

    patchif_results_path = generate_path(
        basepath = os.getcwd(),
        folders = ["patchif_results"]
    )

    model_dirpath = generate_path(
        basepath = patchif_results_path,
        folders = [
            "models_state_dict",
            args.dataset_name,
        ]
    )

    results_dirpath = generate_path(
        basepath = patchif_results_path,
        folders = [
            "metrics",
            args.dataset_name,
        ]
    )

    anomaly_map_dirpath = generate_path(
        basepath = patchif_results_path,
        folders = [
            "anomaly_maps",
            args.dataset_name,
        ]
    )

    if args.exp_name == "":
        if args.contaminate:
            exp_name = f"PatchIF_{args.ad_model_name}_{args.backbone_model_name}_n_estimators_{args.n_estimators}_contamination_{args.contamination_ratio}"
        else:
            exp_name = f"PatchIF_{args.ad_model_name}_{args.backbone_model_name}_n_estimators_{args.n_estimators}"
    else:
        exp_name = args.exp_name

    exp_time = get_current_time()
    exp_dir_name = f"{exp_time}_{exp_name}"

    if args.inference:
        seeds = np.array([args.best_seed])
    else:
        seeds = np.arange(0,args.n_runs)

    for seed in seeds:

        set_exp_seed(seed = seed)

        print('#'* 50)
        print(f"Experiment for seed: {seed}") if not args.inference else print(f"Experiment for seed: {args.best_seed}")
        print('#'* 50)

        for category in args.categories:

            print('#'* 50)
            print(f"Experiment for category: {category}")
            print('#'* 50)

            run_name = f"{exp_name}_{category}_seed_{seed}" if not args.inference else f"{exp_name}_{category}_seed_{seed}_inference"
            setproctitle.setproctitle(run_name)

            if args.train:

                print("---- PatchIF Training ----")

                print('#'* 50)
                print(f"Starting experiment: {exp_name}")
                print('#'* 50)

                print('#'* 50)
                print("Loading training and test datasets")
                print('#'* 50)

                if args.contaminate:

                    train_dataset, train_loader, test_dataset, test_loader, contamination_set_size = load_contaminate_train_test_data(
                        task = TaskType.SEGMENTATION,
                        root = DATASET_PATHS[args.dataset_name],
                        category = category,
                        train_split = Split.TRAIN,
                        test_split = Split.TEST,
                        norm = True,
                        img_size = IMAGE_INPUT_SIZE,
                        gt_mask_size = None,
                        preload_imgs = True,
                        batch_size = BATCH_SIZE,
                        contamination_ratio = args.contamination_ratio,
                        seed = seed,
                    )

                else:

                    contamination_set_size = 0
                    train_dataset, train_loader, test_dataset, test_loader = load_train_test_data(
                        task = TaskType.SEGMENTATION,
                        root = DATASET_PATHS[args.dataset_name],
                        category = category,
                        train_split = Split.TRAIN,
                        test_split = Split.TEST,
                        norm = True,
                        img_size = IMAGE_INPUT_SIZE,
                        gt_mask_size = None,
                        preload_imgs = True,
                        batch_size = BATCH_SIZE,
                        return_loaders = True,
                    )

                print('#'* 50)
                print(f"-- TRAIN DATASET INFORMATION --")
                print(f"Length: {len(train_dataset)}")
                print(f"Number of anomalies: {contamination_set_size}")
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
                    subsample_ratio = args.subsample_ratio,
                    ad_model_type = args.ad_model_name,
                    n_estimators = args.n_estimators,
                    max_nodes = args.max_nodes,
                    plus = args.plus,
                    device = device
                )

                print('#'* 50)
                print(f"Model name: {model.name}")
                print('#'* 50)

                print('#'* 50 )
                print("Defining PatchIF trainer")
                print('#'* 50 )

                if args.save_memory_bank:

                    memory_bank_dirpath = generate_path(
                        basepath = patchif_results_path,
                        folders = [
                            "memory_bank",
                            args.dataset_name,
                            category,
                            args.backbone_model_name,
                        ]
                    )

                    trainer = TrainerPatchIF(
                        model = model,
                        train_dataloader = train_loader,
                        test_dataloader = test_loader,
                        device = device,
                        logger = None,
                        save_memory_bank = args.save_memory_bank,
                        memory_bank_path = memory_bank_dirpath,
                        dataset_name = args.dataset_name,
                        category = category,
                    )

                else:

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
                        args.backbone_model_name,
                        exp_dir_name,
                        f"seed_{seed}"
                    ]
                )

                print('#'* 50)
                print(f"Training the {model.name} model")
                print('#'* 50)
                trainer.train()

                if args.save_model:

                    #TODO: Find a way to copy the `model` object into a new `model_to_save` object on which to call `state_dict()`.
                    # In this way if we are also in `train` mode we can use the `model` object also for the test phase without the need of
                    # loading it from the `pickle` file → `copy.deepcopy` does not work here because it tries to serialize the `model` object
                    # and the problem of the C pointer come back again
                    model_to_save = model
                    model_dict = model_to_save.state_dict()

                    save_element(
                        element = model_dict,
                        dirpath = model_dirpath_seed,
                        filename = f"{model.name}_{args.backbone_model_name}_{category}_state_dict_seed_{seed}",
                        filetype = "pickle"
                    )

                    print('#'* 50)
                    print(f"Model state dict successfully saved in: {model_dirpath_seed}")
                    print('#'* 50)

            if args.test:

                print("---- PatchIF test ----")

                # if not args.train:

                model = PatchIF(
                    backbone_model_name = args.backbone_model_name,
                    layers_idxs = AD_LAYERS[args.backbone_model_name],
                    subsample_ratio = args.subsample_ratio,
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
                        args.backbone_model_name,
                        exp_name if not args.train else exp_dir_name,
                        f"seed_{seed}"
                    ]
                )

                model_path = get_most_recent_file(model_dirpath_seed, file_pos = args.file_pos)
                model_state_dict = open_element(model_path,filetype="pickle")
                model.load_state_dict(state_dict=model_state_dict)
                model.to(device)
                print(f"Loaded model from model_path: {model_path}")

                model.eval()

                print('#'* 50)
                print("Loading the test dataset")
                print('#'* 50)

                if args.contaminate:

                    train_dataset, train_loader, test_dataset, test_loader, contamination_set_size = load_contaminate_train_test_data(
                        task = TaskType.SEGMENTATION,
                        root = DATASET_PATHS[args.dataset_name],
                        category = category,
                        train_split = Split.TRAIN,
                        test_split = Split.TEST,
                        norm = True,
                        img_size = IMAGE_INPUT_SIZE,
                        gt_mask_size = None,
                        preload_imgs = True,
                        batch_size = BATCH_SIZE,
                        contamination_ratio = args.contamination_ratio,
                        seed = seed,
                    )

                else:

                    contamination_set_size = 0
                    train_dataset, train_loader, test_dataset, test_loader = load_train_test_data(
                        task = TaskType.SEGMENTATION,
                        root = DATASET_PATHS[args.dataset_name],
                        category = category,
                        train_split = Split.TRAIN,
                        test_split = Split.TEST,
                        norm = True,
                        img_size = IMAGE_INPUT_SIZE,
                        gt_mask_size = None,
                        preload_imgs = True,
                        batch_size = BATCH_SIZE,
                        return_loaders = True,
                    )

                print('#'* 50)
                print(f"-- TEST DATASET INFORMATION --")
                print(f"Length: {len(test_dataset)}")
                print(f"Contamination ratio: {test_dataset.compute_contamination_ratio()}")
                print('#'* 50)

                if args.save_metrics:

                    results_dirpath_seed = generate_path(
                        basepath = results_dirpath,
                        folders = [
                            category,
                            model.name,
                            args.backbone_model_name,
                            exp_name if not args.train else exp_dir_name,
                            f"seed_{seed}"
                        ]
                    )

                    print('#'* 50)
                    print("Evaluating the model on the test set")
                    print('#'* 50)

                    evaluator = Evaluator(test_dataloader=test_loader, device=device)
                    scores = evaluator.evaluate(model)

                    metrics_filename = f"{model.name}_{args.backbone_model_name}_{category}_metrics_seed_{seed}.csv"
                    metrics_path = os.path.join(results_dirpath_seed, metrics_filename)

                    print('#'* 50)
                    print(f"Saving metrics to path: {metrics_path}")
                    print('#'* 50)

                    append_results(
                        output_path = metrics_path,
                        category = category,
                        seed = seed,
                        scores_dict = scores,
                        ad_model = model.name,
                        feature_layers = AD_LAYERS[args.backbone_model_name],
                        backbone = args.backbone_model_name,
                        input_img_size = IMAGE_INPUT_SIZE,
                        output_img_size = OUTPUT_SIZE,
                    )

                if args.save_figures:

                    anomaly_map_dirpath_seed = generate_path(
                        basepath = anomaly_map_dirpath,
                        folders = [
                            category,
                            model.name,
                            args.backbone_model_name,
                            exp_name if not args.train else exp_dir_name,
                            f"seed_{seed}"
                        ]
                    )

                    print('#'* 50)
                    print(f"Producing anomaly maps for category {category}")
                    print('#'* 50)

                    for images, labels, anomaly_labels, masks, paths in tqdm(iter(test_loader)):

                        anomaly_maps, pred_scores = model(images.to(device))
                        anomaly_maps = torch.permute(torch.tensor(anomaly_maps), (0, 2, 3, 1))

                        for i in range(anomaly_maps.shape[0]):

                            print("#" * 50)
                            print(f"Creating anomaly map for image with path: {paths[i]}")
                            print(f"Label: {labels[i]}")
                            print(f"Anomaly label: {anomaly_labels[i]}")
                            print(f"Predicted score: {pred_scores[i]}")
                            print("#" * 50)

                            model.save_anomaly_map(
                                dirpath = anomaly_map_dirpath_seed,
                                anomaly_map = anomaly_maps[i].cpu().numpy(),
                                pred_score = pred_scores[i],
                                filepath = paths[i],
                                label = labels[i],
                                anomaly_label = anomaly_labels[i],
                                mask = masks[i],
                            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PatchIF Experiment")
    # experiment parameters
    parser.add_argument("--train", action="store_true",help="Run training")
    parser.add_argument("--test", action="store_true",help="Run testing")
    parser.add_argument("--inference", action="store_true",help="Run inference")
    parser.add_argument("--debug", action="store_true",help="Run debug")
    parser.add_argument("--save_figures", action="store_true",help="Save figures")
    parser.add_argument("--save_memory_bank", action="store_true",help="Save the memory bank")
    parser.add_argument("--save_metrics", action="store_true",help="Save metrics")
    parser.add_argument("--save_model", action="store_true",help="Save model")
    parser.add_argument("--save_logs", action="store_true",help="Save logs")
    parser.add_argument("--exp_name", type=str , default="", help="Experiment name")
    parser.add_argument("--dataset_name", type=str , default="mvtec", help="Dataset name, available: mvtec")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs for the experiment")
    parser.add_argument("--best_seed", type=int, default=0, help="Seed to use for the inference experiments")
    parser.add_argument("--device_num", type=int, default=0, help="cuda device number")
    parser.add_argument("--file_pos", type=int, default=0, help="file position in a folder, to load the last saved model for testing")
    # model parameters
    parser.add_argument("--backbone_model_name",type=str,help="Available backbones: resnet18, wide_resnet50_2, mobilenet_v2, mcunet-in3")
    parser.add_argument("--ad_model_name", type=str, default="eif", help="Type of AD model, eif or if")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators for the IF/EIF model, for patchif")
    parser.add_argument("--max_nodes", type=int, default=10000, help="Maximum number of nodes per tree in the IF/EIF model, for patchif")
    parser.add_argument("--plus", action='store_true', help="Flag to use the EIF or EIF+ AD model")
    parser.add_argument("--subsample_ratio", type=float, default=1.0, help="Subsample ratio for the memory bank")
    # dataset parameters
    parser.add_argument("--categories", type=str, nargs="+", default=CATEGORIES)
    parser.add_argument("--contaminate", action="store_true", help="Contaminate the dataset with anomalies")
    parser.add_argument("--contamination_ratio", type=float, default=0.1, help="Contamination ratio for the dataset")

    args = parser.parse_args()


    log_filename = "patchif.log"
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

