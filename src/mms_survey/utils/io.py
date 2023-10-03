__all__ = [
    "compressor",
    "data_dir",
    "plot_dir",
    "raw_dir",
    "process_dir",
    "raw_store",
    "process_store",
    "fix_epoch_metadata",
    "dataset_is_processed",
]

from pathlib import Path

import numpy as np
import pandas.api.types as pd
import xarray as xr
import zarr
from numcodecs import Blosc

compressor = Blosc(cname="zstd", clevel=9)
work_dir = Path(__file__).resolve().parent / ".." / ".." / ".."
data_dir = work_dir / "data"
plot_dir = work_dir / "plots"
raw_dir = data_dir / "raw"
process_dir = data_dir / "processed"
raw_store = zarr.DirectoryStore(raw_dir)
process_store = zarr.NestedDirectoryStore(process_dir)


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


def dataset_is_processed(group: str) -> bool:
    database = zarr.open(store)
    try:
        return database[group].attrs["processed"]
    except KeyError:
        return False
