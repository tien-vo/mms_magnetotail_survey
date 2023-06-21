__all__ = ["load_fgm"]

from pyspedas.mms import mms_load_fgm, mms_config
from pytplot import get_data, del_data
from lib.utils import read_trange
import astropy.constants as c
import astropy.units as u
import tvolib as tv
import numpy as np
import h5py as h5
import tempfile
import lib


def load_fgm(probe, interval, drate="srvy"):

    trange = read_trange(interval, dtype=str)
    prefix = f"mms{probe}_fgm"
    suffix = f"{drate}_l2"

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
            varnames=[f"{prefix}_b_gsm_{suffix}", f"{prefix}_r_gsm_{suffix}"],
        )
        for _ in range(3):
            try:
                mms_load_fgm(latest_version=True, **kw)
                break
            except OSError:
                mms_load_fgm(major_version=True, **kw)
                break

    # Unpack data
    t, B_gsm = tv.utils.get_data(f"{prefix}_b_gsm_{suffix}")
    _t, R_gsm = tv.utils.get_data(f"{prefix}_r_gsm_{suffix}")
    R_gsm = tv.numeric.interpol(R_gsm, _t, t)

    del_data()
    return dict(t=t.astype("f8"), B_gsm=B_gsm[:, :3], R_gsm=R_gsm[:, :3])


if __name__ == "__main__":
    data = load_fgm(1, 0)
