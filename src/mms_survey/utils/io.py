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
import zarr
from numcodecs import Blosc

work_dir = Path(__file__).resolve().parent / ".." / ".." / ".."
data_dir = work_dir / "data"
for dir in [data_dir]:
    dir.mkdir(parents=True, exist_ok=True)

store = zarr.NestedDirectoryStore(data_dir / "database.zarr")
compressor = Blosc(cname="zstd", clevel=7)


def fix_epoch_metadata(dataset, vars=["Epoch"]):
    for var in vars:
        for key_to_remove in ["units", "UNITS"]:
            if key_to_remove in dataset[var].attrs:
                del dataset[var].attrs[key_to_remove]

        for key, value in dataset[var].attrs.items():
            if pd.is_datetime64_dtype(value):
                dataset[var].attrs[key] = value.astype(str)

    return dataset


def dataset_is_ok(group: str):
    database = zarr.open(store)
    try:
        return database[group].attrs["ok"]
    except KeyError:
        return False
