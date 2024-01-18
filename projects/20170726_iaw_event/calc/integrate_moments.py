import numpy as np
import xarray as xr

from mms_survey.utils.io import data_dir

species = "ion"
probe = "mms2"
data_rate = "brst"
data_level = "l2"
store = (
    data_dir /
    "20170726_iaw_event" /
    probe /
    f"fpi_{species}_distribution" /
    data_rate /
    data_level /
    "*"
)

ds = xr.open_mfdataset(
    str(store),
    engine="zarr",
    combine="nested",
    concat_dim="time",
    consolidated=False,
    combine_attrs="drop",
).compute()
