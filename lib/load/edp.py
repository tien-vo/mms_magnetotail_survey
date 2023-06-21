__all__ = ["load_edp"]

from pyspedas.mms import mms_load_edp, mms_load_mec, mms_config
from pyspedas.mms.cotrans.mms_qcotrans import mms_qcotrans
from tempfile import TemporaryDirectory
from pytplot import get_data, del_data
from lib.utils import read_trange
from pyspedas import tinterpol
import astropy.constants as c
import astropy.units as u
import tvolib as tv
import numpy as np
import h5py as h5
import lib


def load_edp(probe, interval, drate="fast"):

    trange = read_trange(interval, dtype=str)
    prefix = f"mms{probe}_edp"
    suffix = f"{drate}_l2"

    # Download EDP files
    with TemporaryDirectory(dir=lib.tmp_dir) as tmp_dir:
        mms_config.CONFIG["local_data_dir"] = tmp_dir
        edp_vars = ["dce_gse", "dce_err", "bitmask"]
        mec_coords = ["gse", "gsm", "dsl"]
        edp_kw = dict(
            trange=trange,
            probe=probe,
            data_rate=drate,
            time_clip=True,
            get_support_data=True,
            varnames=[f"{prefix}_{var}_{suffix}" for var in edp_vars],
        )
        mec_kw = dict(
            trange=trange,
            probe=probe,
            data_rate="srvy" if drate == "fast" else drate,
            time_clip=True,
            varnames=[f"mms{probe}_mec_quat_eci_to_{coord}" for coord in mec_coords],
        )
        for _ in range(3):
            try:
                mms_load_edp(latest_version=True, **edp_kw)
                mms_load_mec(latest_version=True, **mec_kw)
            except OSError:
                mms_load_edp(major_version=True, **edp_kw)
                mms_load_mec(major_version=True, **mec_kw)

    # Rotate GSE to GSM
    for coord in mec_coords:
        tinterpol(f"mms{probe}_mec_quat_eci_to_{coord}", f"{prefix}_dce_gse_{suffix}", suffix="")

    mms_qcotrans(f"{prefix}_dce_gse_{suffix}", f"{prefix}_dce_gsm_{suffix}", "gse", "gsm")
    mms_qcotrans(f"{prefix}_dce_err_{suffix}", f"{prefix}_dce_gsm_err_{suffix}", "dsl", "gsm")

    # Unpack data
    t, E_gsm = tv.utils.get_data(f"{prefix}_dce_gsm_{suffix}")
    _, E_gsm_err = tv.utils.get_data(f"{prefix}_dce_gsm_err_{suffix}")
    _, bitmask = tv.utils.get_data(f"{prefix}_bitmask_{suffix}")

    del_data()
    return dict(t=t.astype("f8"), E_gsm=E_gsm, E_gsm_err=E_gsm_err, bitmask=bitmask)


if __name__ == "__main__":
    data = load_edp(1, 0)
