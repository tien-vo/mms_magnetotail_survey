__all__ = [
    "data_dir",
    "store",
    "compressor",
    "fix_epoch_metadata",
    "dataset_is_ok",
]

from pathlib import Path

import numpy as np
import pandas.api.types as pd
import xarray as xr
import zarr
from numcodecs import Blosc

work_dir = Path(__file__).resolve().parent / ".." / ".." / ".."
data_dir = work_dir / "data"
store = zarr.DirectoryStore(data_dir)
compressor = Blosc(cname="zstd", clevel=9)


def fix_epoch_metadata(
    dataset: xr.Dataset, vars: list = ["Epoch"]
) -> xr.Dataset:
    for var in vars:
        for key_to_remove in ["units", "UNITS"]:
            if key_to_remove in dataset[var].attrs:
                del dataset[var].attrs[key_to_remove]

        for key, value in dataset[var].attrs.items():
            if pd.is_datetime64_dtype(value):
                dataset[var].attrs[key] = value.astype(str)

    return dataset


def dataset_is_ok(group: str) -> bool:
    database = zarr.open(store)
    try:
        return database[group].attrs["ok"]
    except KeyError:
        return False
