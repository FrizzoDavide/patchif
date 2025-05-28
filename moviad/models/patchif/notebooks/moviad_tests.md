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

# `moviad` tests

In this notebooks I will do some quick tests and stuff for the `moviad` library and in particular for the `patchif` project.


```python
import os
from pathlib import Path
import argparse
import ipdb
import torch
import gc
from torchvision.transforms import transforms
from tqdm import tqdm

from moviad.models.patchcore.patchcore import PatchCore
```

