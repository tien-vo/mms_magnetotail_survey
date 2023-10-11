import zarr
import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar

from mms_survey.utils.io import raw_dir, raw_store


def preprocess_fgm(ds: xr.Dataset):
    return ds.drop_dims("eph_time").drop_duplicates("time").sortby("time")


def preprocess_eph(ds: xr.Dataset):
    return ds.drop_dims("time").rename({
        "eph_time": "time"
    }).drop_duplicates("time").sortby("time")


with ProgressBar():
    ds = xr.open_mfdataset(
        str(raw_dir / "srvy" / "fgm" / "mms1" / "*"),
        engine="zarr",
        combine="nested",
        concat_dim="time",
        preprocess=preprocess_eph,
        parallel=True,
        consolidated=False,
    )

#f = zarr.open(raw_store)
#g = f["/srvy/fgm/mms1"]
#groups = sorted([group for group, _ in g.groups()])
#
#kw = dict(store=raw_store, consolidated=False)
##for i in range(len(groups) - 1):
#for i in range(2):
#    group = groups[i]
#    next_group = groups[i + 1]
#
#    ds = xr.open_zarr(group=g[group].name, **kw)
#    ds_next = xr.open_zarr(group=g[next_group].name, **kw)
#    end_date = ds.eph_time.values[-1]
#    start_next = ds_next.eph_time.values[0]
#
#    #end_date = np.datetime64(g[group].attrs["end_date"])
#    #start_next = np.datetime64(g[next_group].attrs["start_date"])
#    if start_next < end_date:
#        print(i, start_next, end_date)
