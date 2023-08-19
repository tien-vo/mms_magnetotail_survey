r"""Load MMS-MEC data into a dictionary"""

__all__ = ["mec"]

import tempfile

import numpy as np
import tvolib as tv
from pyspedas.mms import mms_config, mms_load_mec
from pytplot import del_data, get

import lib
from lib.utils import read_trange


def mec(probe, interval, drate="srvy"):
    trange = read_trange(interval, dtype=str)
    pfx = f"mms{probe}_mec"

    # Download MEC files
    with tempfile.TemporaryDirectory(dir=lib.tmp_dir) as tmp_dir:
        tempfile.tempdir = tmp_dir
        mms_config.CONFIG["local_data_dir"] = tmp_dir
        kw = dict(
            trange=trange,
            probe=probe,
            data_rate=drate,
            time_clip=True,
            varnames=[f"{pfx}_{var}" for var in ["dipole_tilt", "kp", "dst"]],
        )
        for _ in range(3):
            try:
                mms_load_mec(latest_version=True, **kw)
                break
            except OSError:
                mms_load_mec(major_version=True, **kw)
                break

    # Unpack data
    t, dipole_tilt = get(f"{pfx}_dipole_tilt", dt=True, units=True)
    _, kp = get(f"{pfx}_kp", dt=True, units=True)
    _, dst = get(f"{pfx}_dst", dt=True, units=True)

    del_data()
    return dict(t=t.astype("f8"), dipole_tilt=dipole_tilt, kp=kp, dst=dst)


if __name__ == "__main__":
    data = mec(1, 0)
