import numpy as np
import h5py as h5
import lib
import os


trange = np.loadtxt("./resources/intervals.csv", delimiter=",").astype("datetime64[s]").astype("datetime64[ns]")
probes = range(1, 5)
instruments = ["fgm", "edp", "ion-fpi-moms", "elc-fpi-moms", "ion-feeps", "elc-feeps"]

with h5.File(lib.data_file, "w") as h5f:
    h5f.create_dataset("/trange", data=trange.astype("f8"))
    for probe in probes:
        for instrument in instruments:
            for interval in range(N_intervals := trange.shape[0]):
                fname = f"{lib.data_dir}/h5/mms{probe}/{instrument}/interval_{interval}.h5"
                h5f[f"mms{probe}/{instrument}/interval_{interval}"] = h5.ExternalLink(fname, "/")

    for interval in range(N_intervals):
        fname = f"{lib.postprocess_dir}/interval_{interval}.h5"
        h5f[f"postprocess/interval_{interval}"] = h5.ExternalLink(fname, "/")

    h5f["analysis"] = h5.ExternalLink(f"{lib.data_dir}/analysis.h5", "/")
