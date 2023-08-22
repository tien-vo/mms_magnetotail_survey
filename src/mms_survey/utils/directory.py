__all__ = ["data_dir", "tmp_dir", "zarr_store"]

import zarr
from pathlib import Path

work_dir = Path(__file__).resolve().parent / ".." / ".." / ".."
data_dir = work_dir / "data"
tmp_dir = data_dir / "tmp"
for dir in [data_dir, tmp_dir]:
    dir.mkdir(parents=True, exist_ok=True)

zarr_store = zarr.DirectoryStore(data_dir / "database.zarr")
