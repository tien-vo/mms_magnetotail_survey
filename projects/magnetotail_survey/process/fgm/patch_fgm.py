import zarr
import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar

from mms_survey.utils.io import data_dir, default_compressor


def preprocess(ds):
    return ds.isel(time=slice(0, -18))


with ProgressBar():
    ds = xr.open_mfdataset(
        str(data_dir / "test" / "srvy" / "fgm_field" / "mms1" / "*"),
        engine="zarr",
        combine="nested",
        concat_dim="time",
        parallel=True,
        consolidated=False,
        combine_attrs="drop",
    )

    print("Rechunking")
    ds = ds.chunk({"time": 250_000})

    print("Writing")
    ds.to_zarr(
        mode="w",
        store=zarr.NestedDirectoryStore(data_dir / "test"),
        group="/srvy/fgm_field/mms1_combined",
        consolidated=False,
    )
