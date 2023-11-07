import logging
from os.path import splitext

from cdflib.xarray import cdf_to_xarray

from ..base import BaseSynchronizer
from ..utils import clean_metadata, process_epoch_metadata


class SyncMagneticEphemerisCoordinates(BaseSynchronizer):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "epht89q",
        data_level: str = "l2",
        **kwargs,
    ):
        super().__init__(
            instrument="mec",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_type=data_type,
            data_level=data_level,
            query_type="science",
            **kwargs,
        )
        self._compression_factor = 0.147

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
        assert instrument == "mec", "Incorrect input for MEC synchronizer!"
        return {
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_level": data_level,
            "data_type": data_type,
            "version": version,
            "group": (
                f"/{probe}/{instrument}/{data_rate}/{data_level}/"
                f"{data_type}/{time_string}"
            ),
        }

    def process_file(self, temp_file_name: str, file_metadata: str):
        pfx = "{probe}_{instrument}".format(**file_metadata)

        # Load file and fix metadata
        ds = cdf_to_xarray(
            temp_file_name, to_datetime=True, fillval_to_nan=True
        )
        ds = process_epoch_metadata(ds, epoch_vars=["Epoch"])
        ds = ds.reset_coords()
        ds = ds.rename_dims(dict(dim0="quaternion", dim2="rank_1_space"))
        ds = ds.assign_coords(
            {
                "rank_1_space": ["x", "y", "z"],
                "quaternion": ["x", "y", "z", "w"],
            }
        )

        # CDF metadata for version is incorrect here, so we fix it
        ds.attrs["Data_version"] = file_metadata["version"]

        # Rename variables and remove unwanted variables
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
            group=file_metadata["group"],
            encoding=encoding,
            consolidated=False,
        )
