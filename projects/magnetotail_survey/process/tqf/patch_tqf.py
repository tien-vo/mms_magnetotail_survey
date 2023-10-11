import zarr
import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar

from mms_survey.utils.io import raw_dir, raw_store


def preprocess(ds: xr.Dataset):
    return ds.drop_duplicates("time").sortby("time")


#with ProgressBar():
#    ds = xr.open_mfdataset(
#        str(raw_dir / "ancillary" / "tqf" / "*"),
#        engine="zarr",
#        coords=["time",],
#        compat="override",
#        preprocess=preprocess,
#        parallel=True,
#        consolidated=False,
#    )

f = zarr.open(raw_store)
g = f["/ancillary/tqf"]
groups = sorted([group for group, _ in g.groups()])

for i in range(len(groups) - 1):
    group = groups[i]
    next_group = groups[i + 1]
    end_date = np.datetime64(g[group].attrs["end_date"])
    start_next = np.datetime64(g[next_group].attrs["start_date"])
    if start_next < end_date:
        print(i, start_next, end_date)


