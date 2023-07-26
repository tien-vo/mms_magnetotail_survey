r"""Load MMS-FGM data into a dictionary"""

__all__ = ["fgm"]

import tvolib as tv
import numpy as np
import h5py as h5
import tempfile
import lib
from pyspedas.mms import mms_load_fgm, mms_config
from pytplot import get, del_data
from lib.utils import read_trange


def fgm(probe, interval, drate="srvy"):

    trange = read_trange(interval, dtype=str)
    pfx = f"mms{probe}_fgm"
    sfx = f"{drate}_l2"

    # Download FGM files
    with tempfile.TemporaryDirectory(dir=lib.tmp_dir) as tmp_dir:
        tempfile.tempdir = tmp_dir
        mms_config.CONFIG["local_data_dir"] = tmp_dir
        kw = dict(
            trange=trange,
            probe=probe,
            data_rate=drate,
            time_clip=True,
            get_fgm_ephemeris=True,
            varnames=[f"{pfx}_{var}_{sfx}" for var in ["b_gsm", "r_gsm"]],
        )
        for _ in range(3):
            try:
                mms_load_fgm(latest_version=True, **kw)
                break
            except OSError:
                mms_load_fgm(major_version=True, **kw)
                break

    # Unpack data
    t, B_gsm = get(f"{pfx}_b_gsm_{sfx}", dt=True, units=True)
    t_eph, R_gsm = get(f"{pfx}_r_gsm_{sfx}", dt=True, units=True)
    R_gsm = tv.numeric.interpol(R_gsm, t_eph, t)

    del_data()
    return dict(t=t.astype("f8"), B_gsm=B_gsm[:, :3], R_gsm=R_gsm[:, :3])


if __name__ == "__main__":
    data = fgm(1, 0)
