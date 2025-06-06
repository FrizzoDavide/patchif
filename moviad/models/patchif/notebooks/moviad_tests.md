---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: moviad
    language: python
    name: moviad
---

# Memory Bank Analysis for `PatchIF`

In this notebooks I will do some quick tests and stuff for the `moviad` library and in particular for the `patchif` project.

In particular, after the first not optimal experiment results on the new `PatchIF` model I want to study the memory bank extracted from the pre trained `CNN`s in order to see how it is structured and weather it is necessary to adapt the `IF/EIF` models to work with it.


```python
import os
import ipdb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moviad.utilities.manage_files import open_element

root_dir = "../../../../main_scripts"
patchif_results_path = os.path.join(root_dir, "patchif_results")
print(f"root_dir: {root_dir}")
print(f"patchif_results_path: {patchif_results_path}")
```

Let's load the `pickle` file where I saved the memory bank produced by the `mobilenet_v2` backbone on `MVTec AD` dataset on the `pill` category.


```python

```

