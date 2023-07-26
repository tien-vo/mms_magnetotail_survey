from lib.utils import read_num_intervals
import numpy as np
import h5py as h5
import lib
import os

h5f = h5.File(lib.data_file, "w")
for root, dirs, files in os.walk(lib.h5_dir):
    if len(dirs) > 0:
        continue

    path = os.path.relpath(root, lib.h5_dir)
    for f in files:
        h5f[f"{path}/{os.path.splitext(f)[0]}"] = h5.ExternalLink(
            f"{root}/{f}", "/")


#for interval in range(N_intervals):
#    h5f[f"postprocess/interval_{interval}"] = h5.ExternalLink(
#        lib.postprocess_dir / f"interval_{interval}.h5", "/")
#
#h5f["analysis"] = h5.ExternalLink(lib.data_dir / "analysis.h5", "/")
