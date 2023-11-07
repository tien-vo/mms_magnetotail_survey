import logging
from os.path import splitext

from cdflib.xarray import cdf_to_xarray

from ..base import BaseSynchronizer
from ..utils import clean_metadata, process_epoch_metadata


class SyncElectricDoubleProbesScpot(BaseSynchronizer):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_level: str = "l2",
        **kwargs,
    ):
        super().__init__(
            instrument="edp",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate="fast" if data_rate == "srvy" else data_rate,
            data_type="scpot",
            data_level=data_level,
            query_type="science",
            **kwargs,
        )
        self._compression_factor = 0.09

    def get_file_metadata(self, cdf_file_name: str) -> dict:
        (
            probe,
            instrument,
            data_rate,
            data_level,
            data_type,
            time_string,
            version,
        ) = splitext(cdf_file_name)[0].split("_")
        assert (
            instrument == "edp"
        ), "Incorrect input for EDP SCPOT synchronizer!"
        assert (
            data_type == "scpot"
        ), "Incorrect data type for EDP SCPOT synchronizer!"
        return {
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_level": data_level,
            "version": version,
            "group": (
                f"/{probe}/{instrument}_potential/"
                f"{data_rate}/{data_level}/{time_string}"
            ),
        }

    def process_file(self, temp_file_name: str, file_metadata: dict):
        pfx = "{probe}_{instrument}".format(**file_metadata)
        sfx = "{data_rate}_{data_level}".format(**file_metadata)

        # Load file and fix epoch metadata
        ds = cdf_to_xarray(
            temp_file_name, to_datetime=True, fillval_to_nan=True
        )
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
