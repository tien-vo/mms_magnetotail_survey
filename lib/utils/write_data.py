__all__ = ["write_dataset", "TempDir"]

import astropy.units as u
import numpy as np
import h5py as h5
import subprocess
import contextlib
import random
import string
import lib
import os


def write_dataset(probe, interval, instrument, data, where=None):
    fname = f"{lib.h5_dir}/mms{probe}/{instrument}/interval_{interval}.h5"
    with h5.File(fname, "a") as h5f:
        if isinstance(data, dict):
            for key, value in data.items():
                write_dataset(probe, interval, instrument, value, where=key)
        elif isinstance(data, np.ndarray):
            if where in h5f:
                del h5f[where]
            if isinstance(data, u.Quantity):
                h5d = h5f.create_dataset(where, data=data.value)
                h5d.attrs["unit"] = str(data.unit)
            else:
                h5f.create_dataset(where, data=data)


@contextlib.contextmanager
def TempDir():

    # Create temporary directory with random string ID
    tmp_dir = f"{lib.tmp_dir}/{''.join(random.choices(string.ascii_uppercase, k=10))}"
    os.makedirs(tmp_dir, exist_ok=True)

    yield tmp_dir

    # Clean up with subprocess
    os.system(f"rm -rf {tmp_dir}")
    #subprocess.run(["rm", "-rf", tmp_dir], check=True, shell=True)
    #blank = f"{lib.tmp_dir}/{''.join(random.choices(string.ascii_uppercase, k=10))}"
    #os.makedirs(blank, exist_ok=True)
    #subprocess.run(["rm", "-rf", blank], check=True, shell=True)
