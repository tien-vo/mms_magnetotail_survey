__all__ = ["data_dir", "zarr_store"]

import zarr
from pathlib import Path

work_dir = Path(__file__).resolve().parent / ".." / ".." / ".."
data_dir = work_dir / "data"
for dir in [data_dir]:
    dir.mkdir(parents=True, exist_ok=True)

zarr_store = zarr.NestedDirectoryStore(data_dir / "database.zarr")
