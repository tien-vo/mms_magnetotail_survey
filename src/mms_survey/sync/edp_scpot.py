import logging
import os

import zarr
from numcodecs.abc import Codec
from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import default_store, default_compressor

from .base import BaseSync
from .utils import process_epoch_metadata, clean_metadata


class SyncElectricDoubleProbesSCPOT(BaseSync):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_level: str = "l2",
        update: bool = False,
        store: zarr._storage.store.Store = default_store,
        compressor: Codec = default_compressor,
    ):
        super().__init__(
            instrument="edp",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate="fast" if data_rate == "srvy" else data_rate,
            data_type="scpot",
            data_level=data_level,
            product=None,
            query_type="science",
            update=update,
            store=store,
            compressor=compressor,
        )
        self.compression_factor = 0.09

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
            "data_type": data_type,
            "data_level": data_level,
            "version": version,
            "group": (
                f"/{probe}/{instrument}_scpot/{data_rate}/{data_level}/{time}"
            ),
        }

    def process_file(self, file: str, metadata: dict):
        pfx = f"{metadata['probe']}_{metadata['instrument']}"
        sfx = f"{metadata['data_rate']}_{metadata['data_level']}"

        # Load file and fix epoch metadata
        ds = cdf_to_xarray(file, to_datetime=True, fillval_to_nan=True)
        ds = process_epoch_metadata(ds, epoch_vars=[f"{pfx}_epoch_{sfx}"])
        ds = ds.reset_coords()

        # Rename variables and remove unwanted variables
        ds = ds.drop_dims("dim0").rename(
            vars := {
                f"{pfx}_epoch_{sfx}": "time",
                f"{pfx}_scpot_{sfx}": "Phi",
            }
        )
        ds = clean_metadata(ds[list(vars.values())])
        ds["Phi"].attrs["standard_name"] = "Vsc"

        # Save
        ds = ds.drop_duplicates("time").sortby("time")
        ds = ds.chunk(chunks={"time": 250_000})
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
    d = SyncElectricDoubleProbesSCPOT()
    d.download()