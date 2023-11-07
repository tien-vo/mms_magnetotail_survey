import logging
from os.path import splitext

from cdflib.xarray import cdf_to_xarray

from ..base import BaseSynchronizer
from ..utils import clean_metadata, process_epoch_metadata


class SyncElectricDoubleProbesDce(BaseSynchronizer):
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
            data_type="dce",
            data_level=data_level,
            query_type="science",
            **kwargs,
        )
        self._compression_factor = 0.288

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
        assert instrument == "edp", "Incorrect input for EDP DCE synchronizer!"
        assert (
            data_type == "dce"
        ), "Incorrect data type for EDP DCE synchronizer!"
        return {
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_level": data_level,
            "version": version,
            "group": (
                f"/{probe}/{instrument}_dce/{data_rate}/"
                f"{data_level}/{time_string}"
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
        ds = ds.rename(
            vars := {
                f"{pfx}_epoch_{sfx}": "time",
                f"{pfx}_dce_gse_{sfx}": "E_gse",
                f"{pfx}_dce_par_epar_{sfx}": "E_para",
                f"{pfx}_dce_dsl_{sfx}": "E_dsl",
                f"{pfx}_bitmask_{sfx}": "bitmask",
            }
        )
        attrs = ds.E_para.attrs
        E_para = ds.E_para.values
        ds = ds.assign(
            E_para=("time", E_para[:, -1]),
            E_para_err=("time", E_para[:, 0]),
        )
        ds.E_para.attrs.update(**attrs)
        ds = ds.rename_dims(dict(dim0="rank_1_space")).drop_dims("dim1")
        ds = ds.assign_coords({"rank_1_space": ["x", "y", "z"]})
        ds = clean_metadata(ds[list(vars.values()) + ["E_para_err"]])
        ds["E_gse"].attrs["standard_name"] = "E GSE"
        ds["E_dsl"].attrs["standard_name"] = "E DSL"
        ds["E_para"].attrs["standard_name"] = "E para"
        ds["E_para_err"].attrs["standard_name"] = "E para error"
        ds["bitmask"].attrs["standard_name"] = "Bitmask"

        # Save
        ds = ds.drop_duplicates("time").sortby("time")
        ds = ds.chunk(chunks={"time": 250_000, "rank_1_space": 3})
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
