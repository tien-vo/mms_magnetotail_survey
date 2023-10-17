import logging
import os

import zarr
import numpy as np
from numcodecs.abc import Codec
from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import default_store, default_compressor

from ..base import BaseSync
from ..utils import process_epoch_metadata, clean_metadata

_legible_dtype = ["ion", "elc"]


class SyncFastPlasmaInvestigationMoments(BaseSync):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "ion",
        data_level: str = "l2",
        center_timestamps: bool = True,
        update: bool = False,
        store: zarr._storage.store.Store = default_store,
        compressor: Codec = default_compressor,
    ):
        super().__init__(
            instrument="fpi",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate="fast" if data_rate == "srvy" else data_rate,
            data_type=data_type,
            data_level=data_level,
            product=None,
            query_type="science",
            update=update,
            store=store,
            compressor=compressor,
        )
        self.center_timestamps = center_timestamps
        self.compression_factor = 1.0

    @property
    def data_type(self) -> None | str:
        if self._data_type is None:
            return None
        else:
            _data_type = ",".join(self._data_type)
            _data_type = _data_type.replace("ion", "dis-moms")
            _data_type = _data_type.replace("elc", "des-moms")
            return _data_type

    @data_type.setter
    def data_type(self, data_type: None | str | list):
        assert data_type is None or isinstance(
            data_type, (str, list)
        ), "Incorrect type for data type input"
        if isinstance(data_type, str):
            data_type = [data_type]

        for dtype in data_type:
            assert dtype in _legible_dtype, f"{dtype} is not a legible type"

        self._data_type = data_type

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
        data_type = data_type[:3]
        species = "ion" if data_type == "dis" else "elc"
        return {
            "file_name": file_name,
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_type": data_type[:3],
            "data_level": data_level,
            "version": version,
            "species": species,
            "group": (
                f"/{probe}/{instrument}_{species}_moments/"
                f"{data_rate}/{data_level}/{time}"
            ),
        }

    def process_file(self, file_name: str, file_metadata: dict):
        pfx = "{probe}_{data_type}".format(**file_metadata)
        sfx = "{data_rate}".format(**file_metadata)

        # Load file and fix epoch metadata
        ds = cdf_to_xarray(file_name, to_datetime=True, fillval_to_nan=True)
        ds = process_epoch_metadata(ds, epoch_vars=["Epoch"])
        if self.center_timestamps:
            dt = np.int64(
                0.5e9 * (ds.Epoch_plus_var - ds.Epoch_minus_var).values
            )
            attrs = ds.Epoch.attrs
            ds = ds.assign_coords(Epoch=ds.Epoch + np.timedelta64(dt, "ns"))
            ds.Epoch.attrs.update({**attrs, "CATDESC": "Centered timestamps"})

        # Rename dimensions
        ds = ds.rename(Epoch="time").rename_dims(dim0="space")
        ds = ds.swap_dims({f"{pfx}_energy_{sfx}_dim": "energy"})
        ds = ds.assign_coords(
            space=("space", ["x", "y", "z"]),
            energy=("energy", np.arange(32, dtype="i1")),
        ).reset_coords()

        # Rename and remove unwanted variables
        ds = ds.rename(
            vars := {
                f"{pfx}_errorflags_{sfx}": "flag",
                f"{pfx}_energyspectr_omni_{sfx}": "f_omni",
                f"{pfx}_numberdensity_{sfx}": "N",
                f"{pfx}_bulkv_dbcs_{sfx}": "V_dbcs",
                f"{pfx}_bulkv_gse_{sfx}": "V_gse",
                f"{pfx}_bulkv_spintone_dbcs_{sfx}": "V_spintone_dbcs",
                f"{pfx}_bulkv_spintone_gse_{sfx}": "V_spintone_gse",
                f"{pfx}_prestensor_dbcs_{sfx}": "P_tensor_dbcs",
                f"{pfx}_prestensor_gse_{sfx}": "P_tensor_gse",
                f"{pfx}_heatq_dbcs_{sfx}": "Q_dbcs",
                f"{pfx}_heatq_gse_{sfx}": "Q_gse",
                f"{pfx}_temppara_{sfx}": "T_para",
                f"{pfx}_tempperp_{sfx}": "T_perp",
                f"{pfx}_energy_{sfx}": "W",
            }
        )
        ds = clean_metadata(ds[list(vars.values())])

        # Force monotonic
        ds = ds.drop_duplicates("time").sortby("time")

        # Save
        ds.attrs["start_date"] = str(ds.time.values[0])
        ds.attrs["end_date"] = str(ds.time.values[-1])
        ds = ds.chunk(chunks={"time": 1000, "space": 3, "energy": 32})
        encoding = {x: {"compressor": self.compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=self.store,
            group=file_metadata["group"],
            encoding=encoding,
            consolidated=False,
        )
