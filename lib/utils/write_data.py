__all__ = ["write_dataset", "TempDir"]

import astropy.units as u
import h5py as h5
import subprocess
import contextlib
import random
import string
import lib
import os


def write_dataset(probe, interval, where, data):
    fname = f"{lib.data_dir}/h5/mms{probe}/interval_{interval}.h5"
    with h5.File(fname, "a") as h5f:
        for key, value in data.items():
            if (write_where := f"{where}/{key}") in h5f:
                del h5f[write_where]

            if isinstance(value, u.Quantity):
                h5d = h5f.create_dataset(write_where, data=value.value)
                h5d.attrs["unit"] = str(value.unit)
            else:
                h5f.create_dataset(write_where, data=value)


@contextlib.contextmanager
def TempDir():

    # Create temporary directory with random string ID
    tmp_dir = f"{lib.tmp_dir}/{''.join(random.choices(string.ascii_uppercase, k=10))}"
    os.makedirs(tmp_dir, exist_ok=True)

    yield tmp_dir

    # Clean up with subprocess
    blank = f"{lib.tmp_dir}/{''.join(random.choices(string.ascii_uppercase, k=10))}"
    os.makedirs(blank, exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", f"{blank}/", f"{tmp_dir}/"], check=True)
    subprocess.run(["rm", "-rf", tmp_dir], check=True)
    subprocess.run(["rm", "-rf", blank], check=True)
