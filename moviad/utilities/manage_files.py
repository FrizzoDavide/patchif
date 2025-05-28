"""
Some utility functions to manage files
"""

import os
import pickle
import json
import time
import pandas as pd
import numpy as np
from typing import Union, List

import torch
import torch.nn as nn

def get_current_time() -> str:
    """
    This function returns the current time in the format 'dd-mm-YYYY_HH-MM-SS'.
    It is used to produce the name of the files saved

    Returns:
        current_time: string representing the current time

    """

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    return current_time


def save_element(
    element: Union[dict, pd.DataFrame, nn.Module],
    dirpath: str,
    filename: str = "",
    filetype: str = "pickle",
    no_time: bool = False,
) -> None:
    """
    This function saves the model to the specified path

    Args:
        element: element to be saved, can be a dictionary, pandas DataFrame or nn.Module
        dirpath: path of the directory where to save the model
        filename: name of the file to save the model, default=''
        filetype: type of the file to save the model, default='pickle'
        no_time: boolean to indicate whether to include the current time in the filename, default=False

    Returns:
        The method saves the model and does not return any value
    """

    if no_time:
        path = os.path.join(dirpath, filename)
    else:
        current_time = get_current_time()
        filename = current_time + "_" + filename
        path = os.path.join(dirpath, filename)

    #NOTE: This should work both for pickle and json
    if filetype=="pickle":
        with open(path + f".{filetype}", "wb") as f:
            pickle.dump(element, f)
    elif filetype == "json":
        # Check if the element is a pandas DataFrame
        if isinstance(element, pd.DataFrame):
            # Use the DataFrame's to_json method to get a JSON string
            json_string = element.to_json(orient="records", indent=4) # You can adjust orient and indent as needed
            with open(path + f".{filetype}", "w") as f:
                f.write(json_string)
        else:
            # If it's not a DataFrame, assume it's a dictionary or other serializable type
            with open(path + f".{filetype}", "w") as f:
                json.dump(element, f, indent=4) # Added indent for readability
    elif filetype == "csv.gz":
        # Check if the element is a pandas DataFrame
        if isinstance(element, pd.DataFrame):
            # Use the DataFrame's to_csv method to save as gzipped CSV
            element.to_csv(path + f".{filetype}", index=False, compression='gzip')
        else:
            raise ValueError("Element must be a pandas DataFrame to save as gzipped CSV")
    elif filetype == "pth":
        # Save the model state_dict
        if not isinstance(element, nn.Module):
            torch.save(element,path + f".{filetype}")
        else:
            torch.save(element.state_dict(), path + f".{filetype}")


def get_most_recent_file(dirpath: str, file_pos: int = 0) -> str:
    """
    This function returns the most recent file in a directory

    Args:
        dirpath: path of the directory
        file_pos: position of the file in the list of files in the directory sorted in order of creation time, default=0
        (i.e. the most recent file is at position 0)

    Returns:
        The most recent file in the directory
    """

    assert os.path.isdir(dirpath), "The provided path  is not a directory"

    files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {dirpath}")
    paths = [os.path.join(dirpath, basename) for basename in files]
    sorted_paths = sorted(paths, key=os.path.getmtime)[::-1]
    return sorted_paths[file_pos]


def get_most_recent_dir(dirpath: str, file_pos: int = 0) -> str:
    """
    This function returns the most recent subdirectory inside a directory

    Args:
        dirpath: path of the directory
        file_pos: position of the directory in the list of files in the directory sorted in order of creation time, default=0
        (i.e. the most recent file is at position 0)

    Returns:
        The most recent subdirectory in the directory
    """

    assert os.path.isdir(dirpath), "The provided path  is not a directory"

    dirs = [f for f in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, f))]
    paths = [os.path.join(dirpath, basename) for basename in dirs]
    sorted_paths = sorted(paths, key=os.path.getmtime)[::-1]
    return sorted_paths[file_pos]

def open_element(
    file_path: str,
    filetype: str = "pickle"
) -> Union[np.ndarray, list, pd.DataFrame, dict]:
    """
    Function to open an element from a file (i.e. `npz` or `pickle` file) in the specified directory path.

    Args:
        file_path: Path to the file
        filetype: Type of the file (i.e. `npz` or `pickle`)

    Returns:
        Element stored in the file
    """

    assert filetype in ["pickle", "json", "pth"], "filetype must be one of pickle,json or pth"
    if filetype == "pickle":
        with open(file_path, "rb") as fl:
            element = pickle.load(fl)
    elif filetype == "json":
        with open(file_path, "r") as fl:
            element = json.load(fl)
    elif filetype == "pth":
        element = torch.load(file_path)
    else:
        raise ValueError("Invalid filetype. Please choose either 'pickle' 'json' or 'pth'")
    return element

def generate_path(basepath:str = os.getcwd(),
                  folders:List[str] = []) -> str:
    """
    Generate a path starting from a basepath and a list of folders to join to the basepath.

    Args:
        basepath: The basepath from which to start to generate the path, by default os.getcwd()
        folders: A list of strings containing the ordered list of subfolders to join to the basepath, by default []
    Returns:
        path: The path generated by joining the basepath and the folders
    """

    # Verify weather basepath is a valid path in the system
    assert os.path.exists(basepath), f"Basepath {basepath} does not exist"

    path=basepath+"/"

    # Join the basepath with the folders
    for folder in folders:
        path=os.path.join(path, folder) + "/"
        # Verify weather the path exists or not
        if not os.path.exists(path):
            os.makedirs(path)

    return path[:-1]
