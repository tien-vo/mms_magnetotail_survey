import logging
import os

import zarr
from numcodecs.abc import Codec
from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import default_store, default_compressor

from ..base import BaseSync
from ..utils import process_epoch_metadata, clean_metadata


class SyncElectricDoubleProbesPotential(BaseSync):
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

    def get_file_metadata(self, file_name: str) -> dict:
        (
            probe,
            instrument,
            data_rate,
            data_level,
            data_type,
            time,
            version,
        ) = os.path.splitext(file_name)[0].split("_")
        return {
            "file_name": file_name,
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_type": data_type,
            "data_level": data_level,
            "version": version,
            "group": (
                f"/{probe}/{instrument}_potential/"
                f"{data_rate}/{data_level}/{time}"
            ),
        }

    def process_file(self, file_name: str, file_metadata: dict):
        pfx = "{probe}_{instrument}".format(**file_metadata)
        sfx = "{data_rate}_{data_level}".format(**file_metadata)

        # Load file and fix epoch metadata
        ds = cdf_to_xarray(file_name, to_datetime=True, fillval_to_nan=True)
        ds = process_epoch_metadata(ds, epoch_vars=[f"{pfx}_epoch_{sfx}"])
        ds = ds.reset_coords()

        # Rename variables and remove unwanted variables
        ds = ds.drop_dims("dim0").rename(
            vars := {
                f"{pfx}_epoch_{sfx}": "time",
                f"{pfx}_scpot_{sfx}": "V_sc",
            }
        )
        ds = clean_metadata(ds[list(vars.values())])
        ds["V_sc"].attrs["standard_name"] = "Vsc"

        print(ds)

        # Save
        ds = ds.drop_duplicates("time").sortby("time")
        ds = ds.chunk(chunks={"time": 250_000})
        ds.attrs["start_date"] = str(ds.time.values[0])
        ds.attrs["end_date"] = str(ds.time.values[-1])
        encoding = {x: {"compressor": self.compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=self.store,
            group=file_metadata["group"],
            encoding=encoding,
            consolidated=False,
        )