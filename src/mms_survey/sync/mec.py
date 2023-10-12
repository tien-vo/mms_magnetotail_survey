import logging
import os

import zarr
from numcodecs.abc import Codec
from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import default_store, default_compressor

from .base import BaseSync
from .utils import process_epoch_metadata, clean_metadata


class SyncMagneticEphemerisCoordinates(BaseSync):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "epht89q",
        data_level: str = "l2",
        update: bool = False,
        store: zarr._storage.store.Store = default_store,
        compressor: Codec = default_compressor,
    ):
        super().__init__(
            instrument="mec",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_type=data_type,
            data_level=data_level,
            product=None,
            query_type="science",
            update=update,
            store=store,
            compressor=compressor,
        )
        self.compression_factor = 0.147

    def get_metadata(self, file: str) -> dict:
        (
            probe,
            instrument,
            data_rate,
            data_level,
            data_type,
            time,
            version,
        ) = os.path.splitext(file)[0].split("_")
        return {
            "file": file,
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_level": data_level,
            "data_type": data_type,
            "group": (
                f"/{probe}/{instrument}/{data_rate}/{data_level}/"
                f"{data_type}/{time}"
            ),
        }

    def process_file(self, file: str, metadata: str):
        # Load file and fix metadata
        ds = cdf_to_xarray(file, to_datetime=True, fillval_to_nan=True)
        ds = process_epoch_metadata(ds, epoch_vars=["Epoch"])
        ds = ds.reset_coords()
        ds = ds.rename_dims(dict(dim0="quaternion", dim2="space"))
        ds = ds.assign_coords(
            {
                "space": ["x", "y", "z"],
                "quaternion": ["x", "y", "z", "w"],
            }
        )

        # Rename variables and remove unwanted variables
        pfx = f"{metadata['probe']}_{metadata['instrument']}"
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                f"{pfx}_dipole_tilt": "dipole_tilt",
                f"{pfx}_kp": "kp",
                f"{pfx}_dst": "dst",
                f"{pfx}_quat_eci_to_dbcs": "Q_eci_to_dbcs",
                f"{pfx}_quat_eci_to_dsl": "Q_eci_to_dsl",
                f"{pfx}_quat_eci_to_gse": "Q_eci_to_gse",
                f"{pfx}_quat_eci_to_gsm": "Q_eci_to_gsm",
            }
        )
        ds = clean_metadata(ds[list(vars.values())])
        ds["dipole_tilt"].attrs["standard_name"] = "Dipole tilt"
        ds["kp"].attrs["standard_name"] = "Kp"
        ds["dst"].attrs["standard_name"] = "Dst"

        # Save
        ds = ds.drop_duplicates("time").sortby("time")
        ds.attrs["start_date"] = str(ds.time.values[0])
        ds.attrs["end_date"] = str(ds.time.values[-1])
        encoding = {x: {"compressor": self.compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=self.store,
            group=metadata["group"],
            encoding=encoding,
            consolidated=False,
        )


if __name__ == "__main__":
    d = SyncMagneticEphemerisCoordinates()
    d.download()
