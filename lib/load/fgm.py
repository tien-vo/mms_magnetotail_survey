from pyspedas.mms import mms_load_fgm, mms_config
from lib.utils import get_trange, interpol
from tempfile import TemporaryDirectory
from pytplot import get_data, del_data
import astropy.constants as c
import astropy.units as u
import numpy as np
import h5py as h5
import lib
import os

__all__ = ["load_fgm"]


def load_fgm(probe, interval, drate="srvy", wipe=True):

    trange = get_trange(probe, interval)
    prefix = f"mms{probe}_fgm"
    suffix = f"{drate}_l2"

    # Load FGM data
    with TemporaryDirectory(dir=lib.tmp_dir) as tmp_dir:
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
    B_data = get_data(f"{prefix}_b_gsm_{suffix}", xarray=True)
    R_data = get_data(f"{prefix}_r_gsm_{suffix}", xarray=True)

    t = B_data.time.values.astype("datetime64[ns]")
    B = B_data.values * u.Unit(B_dat.CDF["VATT"]["UNITS"])
    R = interpol(R_data.values, R_data.time.values.astype("f8"), t.astype("f8")) * u.Unit(R_data.CDF["VATT"]["UNITS"])

    if wipe:
        del_data()

    return dict(t=t.astype("S29"), B=B[:, :3], R=R[:, :3])


if __name__ == "__main__":
    data = load_fgm(1, 0, wipe=False)
