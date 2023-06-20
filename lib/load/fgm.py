__all__ = ["load_fgm"]

from pyspedas.mms import mms_load_fgm, mms_config
from lib.utils import read_trange, TempDir
from pytplot import get_data, del_data
import astropy.constants as c
import astropy.units as u
import tvolib as tv
import numpy as np
import h5py as h5
import lib


def load_fgm(probe, interval, drate="srvy", wipe=True):

    trange = read_trange(interval, dtype=str)
    prefix = f"mms{probe}_fgm"
    suffix = f"{drate}_l2"

    # Load FGM data
    with TempDir() as tmp_dir:
        mms_config.CONFIG["local_data_dir"] = tmp_dir
        kw = dict(
            trange=trange,
            probe=probe,
            data_rate=drate,
            time_clip=True,
            get_fgm_ephemeris=True,
            varnames=[f"{prefix}_b_gsm_{suffix}", f"{prefix}_r_gsm_{suffix}"],
        )
        try:
            mms_load_fgm(latest_version=True, **kw)
        except OSError:
            mms_load_fgm(major_version=True, **kw)

    # Unpack data
    t, B_gsm = tv.utils.get_data(f"{prefix}_b_gsm_{suffix}")
    _t, R_gsm = tv.utils.get_data(f"{prefix}_r_gsm_{suffix}")
    R_gsm = tv.numeric.interpol(R_gsm, _t, t)

    if wipe:
        del_data()

    return dict(t=t.astype("f8"), B_gsm=B_gsm[:, :3], R_gsm=R_gsm[:, :3])


if __name__ == "__main__":
    data = load_fgm(1, 0, wipe=False)
