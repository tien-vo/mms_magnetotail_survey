import os

import h5py as h5
import numpy as np

import lib
from lib.utils import read_num_intervals

h5f = h5.File(lib.data_file, "w")
for root, dirs, files in os.walk(dir := lib.h5_dir):
    if len(dirs) > 0:
        continue

    path = os.path.relpath(root, dir)
    for f in files:
        h5f[f"{path}/{os.path.splitext(f)[0]}"] = h5.ExternalLink(
            f"{root}/{f}", "/"
        )


for root, dirs, files in os.walk(dir := lib.postprocess_dir):
    if len(dirs) > 0:
        continue

    path = os.path.relpath(root, dir)
    for f in files:
        h5f[f"/postprocess/{path}/{os.path.splitext(f)[0]}"] = h5.ExternalLink(
            f"{root}/{f}", "/"
        )

h5f["analysis"] = h5.ExternalLink(lib.data_dir / "analysis.h5", "/")
