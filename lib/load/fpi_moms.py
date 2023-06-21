__all__ = ["load_fpi_moms"]

from pyspedas.mms import mms_load_fpi, mms_load_mec, mms_config
from pyspedas.mms.cotrans.mms_qcotrans import mms_qcotrans
from pytplot import get_data, store_data, del_data
from lib.utils import read_trange
from pyspedas import tinterpol
import astropy.constants as c
import astropy.units as u
import tvolib as tv
import numpy as np
import h5py as h5
import tempfile
import lib


def load_fpi_moms(probe, interval, drate="fast", species="elc", E_cutoff=60 * u.eV):

    trange = read_trange(interval, dtype=str)
    dtype = "dis" if species == "ion" else "des"
    charge = c.si.e if species == "ion" else -c.si.e
    mass = c.si.m_p if species == "ion" else c.si.m_e
    prefix = f"mms{probe}_{dtype}"
    suffix = f"{drate}"

    # Download FPI moment files
    with tempfile.TemporaryDirectory(dir=lib.tmp_dir) as tmp_dir:
        tempfile.tempdir = tmp_dir
        mms_config.CONFIG["local_data_dir"] = tmp_dir
        mec_coords = ["gse", "gsm", "dbcs"]
        fpi_kw = dict(
            trange=trange,
            probe=probe,
            data_rate=drate,
            datatype=[f"{dtype}-moms", f"{dtype}-partmoms"],
            notplot=True,  # pyspedas not plotting all of the data, awaiting updates
            time_clip=True,
            get_support_data=True,
            center_measurement=True,
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
                data = mms_load_fpi(latest_version=True, **fpi_kw)
                mms_load_mec(latest_version=True, **mec_kw)
                break
            except OSError:
                data = mms_load_fpi(major_version=True, **fpi_kw)
                mms_load_mec(major_version=True, **mec_kw)
                break

    # Unpack time and perform sanity check of timing arrays
    t = np.array(data[f"{prefix}_energyspectr_omni_{suffix}"]["x"], dtype="datetime64[ns]")
    for var in [
        "numberdensity_part",
        "bulkv_part_gse",
        "bulkv_spintone_gse",
        "prestensor_part_gse",
        "bhat_dbcs",
        "scpmean",
    ]:
        assert (np.array(data[f"{prefix}_{var}_{suffix}"]["x"], dtype="datetime64[ns]") == t).all()

    # Unpack other variables
    f_omni = data[f"{prefix}_energyspectr_omni_{suffix}"]["y"] * u.Unit("cm-2 s-1 sr-1")
    f_omni_energy = (data[f"{prefix}_energyspectr_omni_{suffix}"]["v"] * u.eV).to(u.keV)
    Vsc = (charge * data[f"{prefix}_scpmean_{suffix}"]["y"] * u.V).to(u.keV)
    idx = np.nanargmin(np.abs(f_omni_energy - E_cutoff), axis=1)
    b_dbcs = data[f"{prefix}_bhat_dbcs_{suffix}"]["y"]
    N = np.take_along_axis(
        data[f"{prefix}_numberdensity_part_{suffix}"]["y"],
        idx[:, np.newaxis],
        axis=1,
    ).squeeze() * u.Unit("cm-3")
    V_gse = np.take_along_axis(
        data[f"{prefix}_bulkv_part_gse_{suffix}"]["y"],
        idx[:, np.newaxis, np.newaxis],
        axis=1,
    ).squeeze() * u.Unit("km / s")
    V_gse -= data[f"{prefix}_bulkv_spintone_gse_{suffix}"]["y"] * u.Unit("km / s")
    P_tensor_gse = (np.take_along_axis(
        data[f"{prefix}_prestensor_part_gse_{suffix}"]["y"],
        idx[:, np.newaxis, np.newaxis, np.newaxis],
        axis=1,
    ).squeeze() * u.nPa).to(u.Unit("keV cm-3"))

    # Rotate V_gse to GSM
    store_data(f"{prefix}_bulkv_gse_{suffix}", data=dict(x=t.astype("datetime64[s]").astype("f8"), y=V_gse.value))
    store_data(f"{prefix}_bhat_dbcs_{suffix}", data=dict(x=t.astype("datetime64[s]").astype("f8"), y=b_dbcs))
    for coord in mec_coords:
        tinterpol(f"mms{probe}_mec_quat_eci_to_{coord}", f"{prefix}_bulkv_gse_{suffix}", suffix="")

    mms_qcotrans(f"{prefix}_bulkv_gse_{suffix}", f"{prefix}_bulkv_gsm_{suffix}", "gse", "gsm")
    mms_qcotrans(f"{prefix}_bhat_dbcs_{suffix}", f"{prefix}_bhat_gse_{suffix}", "dbcs", "gse")
    V_gsm = get_data(f"{prefix}_bulkv_gsm_{suffix}").y * u.Unit("km/s")
    b_gse = get_data(f"{prefix}_bhat_gse_{suffix}").y

    # Account for background in ion moments
    if species == "ion":
        N_bg = data[f"{prefix}_numberdensity_bg_{suffix}"]["y"] * u.Unit("cm-3")
        P_bg = (data[f"{prefix}_pres_bg_{suffix}"]["y"] * u.nPa).to(u.Unit("keV cm-3"))
        bg_data = dict(N_bg=N_bg, P_bg=P_bg)
    else:
        bg_data = dict()

    del_data()
    return dict(
        t=t.astype("f8"),
        f_omni=f_omni,
        f_omni_energy=f_omni_energy,
        N=N,
        V_gsm=V_gsm,
        V_gse=V_gse,
        P_tensor_gse=P_tensor_gse,
        b_gse=b_gse,
        Vsc=Vsc,
        idx=idx,
    )


if __name__ == "__main__":
    data = load_fpi_moms(1, 418, species="ion")
