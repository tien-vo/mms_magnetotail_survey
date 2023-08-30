import zarr
import xarray as xr

from mms_survey.utils.io import store, compressor
from mms_survey.utils.cotrans import quaternion_rotate

ds_fgm = xr.open_zarr(
    store,
    group="/srvy/fgm/mms1/20170726",
    consolidated=False,
)
ds_mec = xr.open_zarr(
    store,
    group="/srvy/mec/mms1/20170726",
    consolidated=False,
)

Q_eci_to_gse = ds_mec.Q_eci_to_gse.interp(
    time=ds_fgm.time,
    kwargs=dict(fill_value=np.nan),
)
#B_eci = quaternion_rotate(ds_fgm.R_eci, ds.Q_eci_to_gsm)
