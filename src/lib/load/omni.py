r"""Load OMNI data into a dictionary"""

__all__ = ["omni"]

import tempfile

import numpy as np
import tvolib as tv
from pyspedas.omni import data as omni_data
from pyspedas.omni.config import CONFIG
from pytplot import del_data, get

import lib
from lib.utils import read_trange


def omni(interval):
    trange = read_trange(interval, dtype=str)

    # Download MEC files
    with tempfile.TemporaryDirectory(dir=lib.tmp_dir) as tmp_dir:
        tempfile.tempdir = tmp_dir
        CONFIG["local_data_dir"] = tmp_dir
        omni_data(trange=trange, level="hro2", time_clip=True)

    # Unpack data
    t, Bx_gse = get("BX_GSE", dt=True, units=True)
    _, By_gse = get("BY_GSE", dt=True, units=True)
    _, Bz_gse = get("BZ_GSE", dt=True, units=True)
    _, By_gsm = get("BY_GSM", dt=True, units=True)
    _, Bz_gsm = get("BZ_GSM", dt=True, units=True)
    _, Vp = get("flow_speed", dt=True, units=True)
    _, Vx = get("Vx", dt=True, units=True)
    _, Vy = get("Vy", dt=True, units=True)
    _, Vz = get("Vz", dt=True, units=True)
    _, Np = get("proton_density", dt=True, units=True)
    _, Pp = get("Pressure", dt=True, units=True)
    _, SYM_H = get("SYM_H", dt=True, units=True)
    _, SYM_D = get("SYM_D", dt=True, units=True)
    _, ASY_H = get("ASY_H", dt=True, units=True)
    _, ASY_D = get("ASY_D", dt=True, units=True)

    del_data()
    return dict(
        t=t.astype("f8"), Bx_gse=Bx_gse, By_gse=By_gse, Bz_gse=Bz_gse, By_gsm=By_gsm, Bz_gsm=Bz_gsm,
        Vp=Vp, Vx=Vx, Vy=Vy, Vz=Vz, Np=Np, Pp=Pp, SYM_H=SYM_H, SYM_D=SYM_D, ASY_H=ASY_H, ASY_D=ASY_D,
    )


if __name__ == "__main__":
    data = omni(0)
