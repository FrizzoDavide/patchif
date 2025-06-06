"""
Python script to analyse the memory bank for `PatchIF`
"""

import os
import ipdb
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from moviad.utilities.manage_files import open_element, generate_path, get_most_recent_file
import umap

pwd = os.getcwd()
patchif_results_path = os.path.join(pwd,"patchif_results")

parser = argparse.ArgumentParser(description="Analyse the memory bank for PatchIF")

parser.add_argument("--dataset_name",type=str,default="mvtec",help="Dataset name")
parser.add_argument("--category",type=str,default="pill",help="Category of the dataset")
parser.add_argument("--backbone",type=str,default="mobilenet_v2",help="Pre trained CNN backbone")
parser.add_argument("--file_pos",type=int,default=0,help="Position of the memory bank in the memory bank folder")
parser.add_argument("--decomposition",type=str,default="pca",help="Dimensionality reduction technique to use")
parser.add_argument("--n_components",type=int,default=2,help="Number of components for PCA and UMAP")
parser.add_argument("--n_neighbors",type=int,default=15,help="Number of neighbors for UMAP")
parser.add_argument("--min_dist",type=float,default=0.1,help="Minimum distance for UMAP")

args = parser.parse_args()

memory_bank_dirpath = generate_path(
    basepath = patchif_results_path,
    folders = [
        "memory_bank",
        args.dataset_name,
        args.category,
        args.backbone,
    ]
)

memory_bank_filepath = get_most_recent_file(memory_bank_dirpath,file_pos=1)
memory_bank_dict = open_element(memory_bank_filepath,filetype="pickle")
memory_bank = memory_bank_dict["memory_bank"]

#NOTE: Apply PCA to reduce dimensionality and visualize the memory bank

# Reshape the memory bank to 2D to make PCA work
new_samples = memory_bank.shape[0]*memory_bank.shape[2]*memory_bank.shape[3]
memory_bank = memory_bank.reshape(new_samples, memory_bank.shape[1])

# Normalize the memory bank before applying PCA
scaler = StandardScaler()
memory_bank_scaled = scaler.fit_transform(memory_bank)

print('#'* 50)
print(f"New shape of memory bank: {memory_bank.shape}")
print('#'* 50)

if args.decomposition == "pca":
    pca = PCA(n_components=args.n_components)
    memory_bank_decomposed = pca.fit_transform(memory_bank_scaled)

    print('#'* 50)
    print("decomposed Results with 3 components")
    print(f"Shape of transformed memory bank: {memory_bank_decomposed.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print('#'* 50)


elif args.decomposition == "umap":

    umap = umap.UMAP(
        n_components = args.n_components,
        n_neighbors = args.n_neighbors,
        min_dist = args.min_dist,
        random_state = 42
    )

    memory_bank_decomposed = umap.fit_transform(memory_bank_scaled)


#NOTE: 3D plot of the PCA or UMAP results

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    memory_bank_decomposed[:, 0],
    memory_bank_decomposed[:, 1],
    memory_bank_decomposed[:, 2],
    c="skyblue",
    marker='o',
    s=5,
    alpha=0.6
)

ax.set_xlabel(f'{args.decomposition} Component 1')
ax.set_ylabel(f'{args.decomposition} Component 2')
ax.set_zlabel(f'{args.decomposition} Component 3')
ax.set_title(f'3D {args.decomposition} of Memory Bank')

plot_path_dirpath = generate_path(
    basepath=patchif_results_path,
    folders=["memory_bank_plots",args.decomposition]
)

if args.decomposition == "pca":
    filename = f"memory_bank_{args.decomposition}_3d_scaled"
elif args.decomposition == "umap":
    filename = f"Amemory_bank_{args.decomposition}_3d_scaled_neighbors_{args.n_neighbors}_min_dist_{args.min_dist}"
else:
    filename=None
    raise ValueError(f"Unknown decomposition technique: {args.decomposition}")

if "contamination" in os.path.basename(memory_bank_filepath):
    filename = f"{filename}_contaminated.png"
else:
    filename = f"{filename}.png"

plot_path = os.path.join(plot_path_dirpath, filename)
plt.savefig(plot_path, dpi=300)
print('#'* 50)
print(f"Plot saved to {plot_path}")
print('#'* 50)
