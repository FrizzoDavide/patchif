"""
Python script to analyse the memory bank for `PatchIF`
"""

import os
import ipdb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moviad.utilities.manage_files import open_element, generate_path, get_most_recent_file

pwd = os.getcwd()
patchif_results_path = os.path.join(pwd,"patchif_results")

memory_bank_dirpath = generate_path(
    basepath = patchif_results_path,
    folders = [
        "memory_bank",
        "mvtec",
        "pill",
        "mobilenet_v2",
    ]
)

memory_bank_filepath = get_most_recent_file(memory_bank_dirpath,file_pos=0)
memory_bank_dict = open_element(memory_bank_filepath,filetype="pickle")
memory_bank = memory_bank_dict["memory_bank"]

print('#'* 50)
print(f"Memory bank successfully loaded from {memory_bank_filepath}")
print(f"Memory bank shape: {memory_bank.shape}")
print('#'* 50)
