from pathlib import Path

import numpy as np
import pandas.api.types as pd
import xarray as xr
import zarr
from numcodecs import Blosc

work_dir = Path(__file__).resolve().parent / ".." / ".." / ".."
data_dir = work_dir / "data"

default_compressor = Blosc(cname="zstd", clevel=9)
default_store = zarr.DirectoryStore(data_dir)
