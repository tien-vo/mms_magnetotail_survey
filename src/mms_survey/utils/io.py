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


def dataset_is_processed(group: str) -> bool:
    database = zarr.open(raw_store)
    try:
        return database[group].attrs["processed"]
    except KeyError:
        return False
