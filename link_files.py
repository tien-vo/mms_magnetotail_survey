import numpy as np
import h5py as h5
import lib
import os


trange = np.loadtxt("./resources/intervals.csv", delimiter=",").astype("datetime64[s]").astype("datetime64[ns]")
for probe in range(1, 5):
    os.makedirs(f"{lib.data_dir}/h5/mms{probe}", exist_ok=True)

with h5.File(lib.data_file, "w") as h5f_data:
    for probe in range(1, 5):
        for interval in range(trange.shape[0]):
            fname = f"{lib.data_dir}/h5/mms{probe}/interval_{interval}.h5"
            with h5.File(fname, "w") as h5f:
                h5f.create_dataset("/trange", data=trange[interval, :].astype("f8"))

            h5f_data[f"mms{probe}/interval_{interval}"] = h5.ExternalLink(fname, "/")
