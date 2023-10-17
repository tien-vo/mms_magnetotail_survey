import logging
import os

import zarr
from numcodecs.abc import Codec
from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import default_store, default_compressor

from ..base import BaseSync
from ..utils import process_epoch_metadata, clean_metadata


class SyncFluxGateMagnetometer(BaseSync):
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
            instrument="fgm",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_type=None,
            data_level=data_level,
            product=None,
            query_type="science",
            update=update,
            store=store,
            compressor=compressor,
        )
        self.compression_factor = 0.259

    def get_file_metadata(self, file_name: str) -> dict:
        (
            probe,
            instrument,
            data_rate,
            data_level,
            time,
            version,
        ) = os.path.splitext(file_name)[0].split("_")
        return {
            "file_name": file_name,
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_level": data_level,
            "version": version,
            "group": (
                f"/{probe}/{instrument}_fields/"
                f"{data_rate}/{data_level}/{time}"
            ),
            "eph_group": (
                f"/{probe}/{instrument}_ephemeris/"
                f"{data_rate}/{data_level}/{time}"
            ),
        }

    def process_file(self, file_name: str, file_metadata: dict):
        pfx = "{probe}_{instrument}".format(**file_metadata)
        sfx = "{data_rate}_{data_level}".format(**file_metadata)

        # Load file and fix epoch metadata
        ds = cdf_to_xarray(file_name, to_datetime=True, fillval_to_nan=True)
        ds = process_epoch_metadata(ds, epoch_vars=["Epoch", "Epoch_state"])
        ds = ds.reset_coords()

        # Rename dimensions and drop magnitude
        ds = ds.rename_dims(dict(dim0="space"))
        ds = ds.assign_coords({"space": ["x", "y", "z", "mag"]})
        ds = ds.drop_sel(space="mag")

        # Rename and remove unwanted data variables
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                "Epoch_state": "eph_time",
                f"{pfx}_b_gse_{sfx}": "B_gse",
                f"{pfx}_b_gsm_{sfx}": "B_gsm",
                f"{pfx}_r_gse_{sfx}": "R_gse",
                f"{pfx}_r_gsm_{sfx}": "R_gsm",
                f"{pfx}_flag_{sfx}": "flag",
            }
        )
        ds = clean_metadata(ds[list(vars.values())])
        ds["B_gse"].attrs["standard_name"] = "B GSE"
        ds["B_gsm"].attrs["standard_name"] = "B GSM"
        ds["R_gse"].attrs["standard_name"] = "R GSE"
        ds["R_gsm"].attrs["standard_name"] = "R GSM"
        ds["flag"].attrs["standard_name"] = "Flag"

        # Save FGM field measurements
        ds_field = ds.drop_dims("eph_time")
        ds_field = ds_field.drop_duplicates("time").sortby("time")
        ds_field = ds_field.chunk(chunks={"time": 250_000, "space": 3})
        ds_field.attrs["start_date"] = str(ds_field.time.values[0])
        ds_field.attrs["end_date"] = str(ds_field.time.values[-1])
        encoding = {x: {"compressor": self.compressor} for x in ds_field}
        ds_field.to_zarr(
            mode="w",
            store=self.store,
            group=file_metadata["group"],
            encoding=encoding,
            consolidated=False,
        )

        # Save FGM ephemeris measurements
        ds_eph = ds.drop_dims("time").rename({"eph_time": "time"})
        ds_eph = ds_eph.drop_duplicates("time").sortby("time")
        ds_eph = ds_eph.chunk(chunks={"time": 250_000, "space": 3})
        ds_eph.attrs["start_date"] = str(ds_eph.time.values[0])
        ds_eph.attrs["end_date"] = str(ds_eph.time.values[-1])
        encoding = {x: {"compressor": self.compressor} for x in ds_eph}
        ds_eph.to_zarr(
           mode="w",
           store=self.store,
           group=file_metadata["eph_group"],
           encoding=encoding,
           consolidated=False,
        )
