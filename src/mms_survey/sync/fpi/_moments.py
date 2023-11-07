import logging
from os.path import splitext

import numpy as np
import xarray as xr
from cdflib.xarray import cdf_to_xarray

from ..base import BaseSynchronizer
from ..utils import clean_metadata, process_epoch_metadata

_legible_dtype = ["ion", "elc"]


class SyncFastPlasmaInvestigationMoments(BaseSynchronizer):
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
            _data_type = _data_type.replace("ion", "dis-moms")
            _data_type = _data_type.replace("elc", "des-moms")
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
        assert (
            instrument == "fpi"
        ), "Incorrect input for FPI moments synchronizer!"
        assert (
            data_type[4:] == "moms"
        ), "Incorrect data type for FPI moments synchronizer!"
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
                f"/{probe}/{instrument}_{species}_moments/"
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
                "dim0": "rank_1_space",
                f"{pfx}_energy_{sfx}_dim": "energy_channel",
            }
        )
        ds = ds.assign_coords(
            rank_1_space=("rank_1_space", ["x", "y", "z"]),
            rank_2_space=(
                "rank_2_space",
                ["xx", "yy", "zz", "xy", "xz", "yz"],
            ),
            energy_channel=("energy_channel", np.arange(32, dtype="i1")),
        ).reset_coords()

        # Reorganize rank-2 tensors
        P_tensor_dbcs = ds[f"{pfx}_prestensor_dbcs_{sfx}"]
        P_tensor_gse = ds[f"{pfx}_prestensor_gse_{sfx}"]
        ds = ds.assign(
            P_tensor_dbcs=xr.DataArray(
                np.array(
                    [
                        P_tensor_dbcs.values[:, 0, 0],
                        P_tensor_dbcs.values[:, 1, 1],
                        P_tensor_dbcs.values[:, 2, 2],
                        P_tensor_dbcs.values[:, 0, 1],
                        P_tensor_dbcs.values[:, 0, 2],
                        P_tensor_dbcs.values[:, 1, 2],
                    ]
                ).T,
                dims=("time", "rank_2_space"),
                attrs=P_tensor_dbcs.attrs,
            ),
            P_tensor_gse=xr.DataArray(
                np.array(
                    [
                        P_tensor_gse.values[:, 0, 0],
                        P_tensor_gse.values[:, 1, 1],
                        P_tensor_gse.values[:, 2, 2],
                        P_tensor_gse.values[:, 0, 1],
                        P_tensor_gse.values[:, 0, 2],
                        P_tensor_gse.values[:, 1, 2],
                    ]
                ).T,
                dims=("time", "rank_2_space"),
                attrs=P_tensor_gse.attrs,
            ),
        )

        # Rename and remove unwanted variables
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                f"{pfx}_errorflags_{sfx}": "flag",
                f"{pfx}_energyspectr_omni_{sfx}": "f1d_W",
                f"{pfx}_numberdensity_{sfx}": "N",
                f"{pfx}_bulkv_dbcs_{sfx}": "V_dbcs",
                f"{pfx}_bulkv_gse_{sfx}": "V_gse",
                f"{pfx}_bulkv_spintone_dbcs_{sfx}": "V_spintone_dbcs",
                f"{pfx}_bulkv_spintone_gse_{sfx}": "V_spintone_gse",
                f"{pfx}_heatq_dbcs_{sfx}": "Q_dbcs",
                f"{pfx}_heatq_gse_{sfx}": "Q_gse",
                f"{pfx}_temppara_{sfx}": "T_para",
                f"{pfx}_tempperp_{sfx}": "T_perp",
                f"{pfx}_energy_{sfx}": "W",
            }
        ).set_coords("W")
        ds = clean_metadata(
            ds[list(vars.values()) + ["P_tensor_dbcs", "P_tensor_gse"]]
        )

        # Force monotonic
        ds = ds.drop_duplicates("time").sortby("time")

        # Save
        ds.attrs["start_date"] = str(ds.time.values[0])
        ds.attrs["end_date"] = str(ds.time.values[-1])
        ds = ds.chunk(chunks={
            "time": 1000,
            "rank_1_space": 3,
            "rank_2_space": 6,
            "energy_channel": 32,
        })
        encoding = {x: {"compressor": self.compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=self.store,
            group=file_metadata["group"],
            encoding=encoding,
            consolidated=False,
        )
