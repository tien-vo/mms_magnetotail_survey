import numpy as np
import xarray as xr

from mms_survey.load.fgm import LoadFluxGateMagnetometer
from mms_survey.utils.cotrans import quaternion_rotate
from mms_survey.utils.io import raw_store


def test_quaternion_rotate(probe="mms1", drate="srvy", B_tol=0.01):
    d = LoadFluxGateMagnetometer(
        start_date="2017-07-26",
        end_date="2017-07-26",
        probe=probe,
        data_rate=drate,
        skip_processed_data=True,
    )
    d.download()

    fgm_group = f"/{drate}/fgm/{probe}/20170726"
    mec_group = f"/{drate}/mec/{probe}/20170726"
    ds_fgm = xr.open_zarr(raw_store, group=fgm_group, consolidated=False)
    ds_mec = xr.open_zarr(raw_store, group=mec_group, consolidated=False)

    kw = dict(time=ds_fgm.time, kwargs=dict(fill_value=np.nan))
    Q_eci_to_gse = ds_mec.Q_eci_to_gse.interp(**kw)
    Q_eci_to_gsm = ds_mec.Q_eci_to_gsm.interp(**kw)

    B_eci = quaternion_rotate(ds_fgm.B_gse, Q_eci_to_gse, inverse=True)
    B_gsm = quaternion_rotate(B_eci, Q_eci_to_gsm)

    dB = np.linalg.norm(ds_fgm.B_gsm - B_gsm, axis=1)
    dB[np.isnan(dB)] = 0.0
    assert (dB <= B_tol).all()
