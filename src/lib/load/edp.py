r"""Load MMS-EDP data into a dictionary"""

__all__ = ["edp"]

import tempfile

import numpy as np
import tvolib as tv
from pyspedas import tinterpol
from pyspedas.mms import mms_config, mms_load_edp, mms_load_mec
from pyspedas.mms.cotrans.mms_qcotrans import mms_qcotrans
from pytplot import del_data, get

import lib
from lib.utils import read_trange


def edp(probe, interval, drate="fast"):
    trange = read_trange(interval, dtype=str)
    pfx = f"mms{probe}_edp"
    sfx = f"{drate}_l2"

    # Download EDP files
    with tempfile.TemporaryDirectory(dir=lib.tmp_dir) as tmp_dir:
        tempfile.tempdir = tmp_dir
        mms_config.CONFIG["local_data_dir"] = tmp_dir
        edp_vars = ["dce_gse", "dce_err", "bitmask"]
        mec_coords = ["gse", "gsm", "dsl"]
        edp_kw = dict(
            trange=trange,
            probe=probe,
            data_rate=drate,
            time_clip=True,
            get_support_data=True,
            varnames=[f"{pfx}_{var}_{sfx}" for var in edp_vars],
        )
        mec_kw = dict(
            trange=trange,
            probe=probe,
            data_rate="srvy" if drate == "fast" else drate,
            time_clip=True,
            varnames=[
                f"mms{probe}_mec_quat_eci_to_{coord}" for coord in mec_coords
            ],
        )
        for _ in range(3):
            try:
                mms_load_edp(latest_version=True, **edp_kw)
                mms_load_mec(latest_version=True, **mec_kw)
                break
            except OSError:
                mms_load_edp(major_version=True, **edp_kw)
                mms_load_mec(major_version=True, **mec_kw)
                break

    # Rotate GSE to GSM
    for coord in mec_coords:
        tinterpol(
            f"mms{probe}_mec_quat_eci_to_{coord}",
            f"{pfx}_dce_gse_{sfx}",
            suffix="",
        )

    mms_qcotrans(f"{pfx}_dce_gse_{sfx}", f"{pfx}_dce_gsm_{sfx}", "gse", "gsm")
    mms_qcotrans(
        f"{pfx}_dce_err_{sfx}", f"{pfx}_dce_gsm_err_{sfx}", "dsl", "gsm"
    )

    # Unpack data
    t, E_gsm = get(f"{pfx}_dce_gsm_{sfx}", dt=True, units=True)
    _, E_gsm_err = get(f"{pfx}_dce_gsm_err_{sfx}", dt=True, units=True)
    _, bitmask = get(f"{pfx}_bitmask_{sfx}", dt=True, units=True)

    del_data()
    return dict(
        t=t.astype("f8"), E_gsm=E_gsm, E_gsm_err=E_gsm_err, bitmask=bitmask
    )


if __name__ == "__main__":
    data = edp(1, 0)
