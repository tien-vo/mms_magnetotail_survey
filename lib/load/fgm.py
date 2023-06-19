__all__ = ["load_fgm"]

from pyspedas.mms import mms_load_fgm, mms_config
from lib.utils import get_trange, TempDir
from pytplot import get_data, del_data
import astropy.constants as c
import astropy.units as u
import tvolib as tv
import numpy as np
import h5py as h5
import lib


def load_fgm(probe, interval, drate="srvy", wipe=True):

    trange = get_trange(probe, interval)
    prefix = f"mms{probe}_fgm"
    suffix = f"{drate}_l2"

    # Load FGM data
    with TempDir() as tmp_dir:
        mms_config.CONFIG["local_data_dir"] = tmp_dir
        mms_load_fgm(
            trange=trange,
            probe=probe,
            data_rate=drate,
            time_clip=True,
            latest_version=True,
            get_fgm_ephemeris=True,
            varnames=[f"{prefix}_b_gsm_{suffix}", f"{prefix}_r_gsm_{suffix}"],
        )

    # Unpack data
    t, B = tv.utils.get_data(f"{prefix}_b_gsm_{suffix}")
    _t, R = tv.utils.get_data(f"{prefix}_r_gsm_{suffix}")
    R = tv.numeric.interpol(R, _t, t)

    if wipe:
        del_data()

    return dict(t=t.astype("f8"), B=B[:, :3], R=R[:, :3])


if __name__ == "__main__":
    data = load_fgm(1, 0, wipe=False)
