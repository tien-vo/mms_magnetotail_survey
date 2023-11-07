import logging
from os.path import splitext

import numpy as np
from cdflib.xarray import cdf_to_xarray

from ..base import BaseSynchronizer
from ..utils import clean_metadata, process_epoch_metadata

_legible_dtype = ["ion", "elc"]


class SyncFastPlasmaInvestigationDistribution(BaseSynchronizer):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "ion",
        data_level: str = "l2",
        center_timestamps: bool = True,
        **kwargs,
    ):
        super().__init__(
            instrument="fpi",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate="fast" if data_rate == "srvy" else data_rate,
            data_type=data_type,
            data_level=data_level,
            query_type="science",
            **kwargs,
        )
        self.center_timestamps = center_timestamps
        self._compression_factor = 1.0

    @property
    def data_type(self) -> str | None:
        if self._data_type is None:
            return None
        else:
            _data_type = ",".join(self._data_type)
            _data_type = _data_type.replace("ion", "dis-dist")
            _data_type = _data_type.replace("elc", "des-dist")
            return _data_type

    @data_type.setter
    def data_type(self, data_type: str | list[str] | None):
        assert data_type is None or isinstance(
            data_type, (str, list)
        ), "Incorrect type for data type input"
        if isinstance(data_type, str):
            data_type = [data_type]

        for dtype in data_type:
            assert dtype in _legible_dtype, f"{dtype} is not a legible type"

        self._data_type = data_type

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
        assert instrument == "fpi", "Incorrect input for FPI synchronizer!"
        assert (
            data_type[4:] == "dist"
        ), "Incorrect data type for FPI distribution synchronizer!"
        species = "ion" if (data_type := data_type[:3]) == "dis" else "elc"
        return {
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_type": data_type,
            "data_level": data_level,
            "version": version,
            "species": species,
            "group": (
                f"/{probe}/{instrument}_{species}_distribution/"
                f"{data_rate}/{data_level}/{time_string}"
            ),
        }

    def process_file(self, temp_file_name: str, file_metadata: dict):
        pfx = "{probe}_{data_type}".format(**file_metadata)
        sfx = "{data_rate}".format(**file_metadata)

        # Load file and fix epoch metadata
        ds = cdf_to_xarray(
            temp_file_name, to_datetime=True, fillval_to_nan=True
        )
        ds = process_epoch_metadata(ds, epoch_vars=["Epoch"])
        if self.center_timestamps:
            dt = np.int64(
                0.5e9 * (ds.Epoch_plus_var - ds.Epoch_minus_var).values
            )
            attrs = ds.Epoch.attrs
            ds = ds.assign_coords(Epoch=ds.Epoch + np.timedelta64(dt, "ns"))
            ds.Epoch.attrs.update({**attrs, "CATDESC": "Centered timestamps"})

        # Rename dimensions
        ds = ds.swap_dims(
            {
                f"{pfx}_energy_{sfx}_dim": "energy_channel",
                f"{pfx}_theta_{sfx}": "zenith_sector",
            }
        )
        if file_metadata["data_rate"] == "brst":
            ds = ds.swap_dims({f"{pfx}_phi_{sfx}_dim": "azimuthal_sector"})
        else:
            ds = ds.swap_dims({f"{pfx}_phi_{sfx}": "azimuthal_sector"})
        ds = ds.assign_coords(
            energy_channel=("energy_channel", np.arange(32, dtype="i1")),
            zenith_sector=("zenith_sector", np.arange(16, dtype="i1")),
            azimuthal_sector=("azimuthal_sector", np.arange(32, dtype="i1")),
        ).reset_coords()

        # Rename and remove unwanted variables
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                f"{pfx}_energy_{sfx}": "W",
                f"{pfx}_theta_{sfx}": "theta",
                f"{pfx}_phi_{sfx}": "phi",
                f"{pfx}_dist_{sfx}": "f3d",
                f"{pfx}_disterr_{sfx}": "f3d_err",
            }
        ).set_coords(["W", "theta", "phi"])
        ds = clean_metadata(ds[list(vars.values())])

        # Force monotonic
        ds = ds.drop_duplicates("time").sortby("time")

        # Save
        ds.attrs["start_date"] = str(ds.time.values[0])
        ds.attrs["end_date"] = str(ds.time.values[-1])
        ds = ds.chunk(
            chunks={
                "time": 1000,
                "energy_channel": 1,
                "zenith_sector": 16,
                "azimuthal_sector": 32,
            }
        )
        encoding = {x: {"compressor": self.compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=self.store,
            group=file_metadata["group"],
            encoding=encoding,
            consolidated=False,
        )
