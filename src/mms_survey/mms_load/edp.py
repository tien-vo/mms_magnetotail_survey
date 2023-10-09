__all__ = ["LoadElectricDoubleProbes"]

import os
import logging

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import compressor, raw_store

from .base import BaseLoader
from .utils import process_epoch_metadata


class LoadElectricDoubleProbes(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "dce",
        data_level: str = "l2",
        skip_processed_data: bool = False,
    ):
        super().__init__(
            instrument="edp",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate="fast" if data_rate == "srvy" else data_rate,
            data_type=data_type,
            data_level=data_level,
            product=None,
            query_type="science",
            skip_processed_data=skip_processed_data,
        )
        if data_type == "dce":
            self.compression_factor = 0.288
        elif data_type == "scpot":
            self.compression_factor = 0.09
        else:
            raise NotImplementedError()

    def get_metadata(self, file: str) -> dict:
        name = os.path.splitext(file)[0]
        (
            probe,
            instrument,
            data_rate,
            data_level,
            data_type,
            tid,
            _,
        ) = name.split("_")
        return {
            "file": file,
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_type": data_type,
            "data_level": data_level,
            "group": f"/{data_rate}/{instrument}/{data_type}/{probe}/{tid}",
        }

    def process_file(self, file: str, metadata: dict):
        if metadata["instrument"] != self.instrument:
            logging.warning(f"{file} is not in EDP dataset!")
            return

        # Load file and fix epoch metadata
        ds = cdf_to_xarray(file, to_datetime=True, fillval_to_nan=True)
        pfx = f"{metadata['probe']}_{metadata['instrument']}"
        sfx = f"{metadata['data_rate']}_{metadata['data_level']}"
        ds = process_epoch_metadata(ds, epoch_vars=[f"{pfx}_epoch_{sfx}"])
        ds = ds.reset_coords()

        # Rename variables and remove unwanted variables
        if metadata["data_type"] == "dce":
            ds = ds.drop_dims("dim1").rename(
                vars := {
                    f"{pfx}_epoch_{sfx}": "time",
                    f"{pfx}_dce_gse_{sfx}": "E_gse",
                    f"{pfx}_dce_dsl_{sfx}": "E_dsl",
                    f"{pfx}_dce_err_{sfx}": "E_err",
                    f"{pfx}_bitmask_{sfx}": "bitmask",
                }
            )
            ds = ds.rename_dims(dict(dim0="space"))
            ds = ds.assign_coords({"space": ["x", "y", "z"]})
        elif metadata["data_type"] == "scpot":
            ds = ds.drop_dims("dim0").rename(
                vars := {
                    f"{pfx}_epoch_{sfx}": "time",
                    f"{pfx}_scpot_{sfx}": "Phi",
                }
            )
        else:
            raise NotImplementedError()
        ds = ds[list(vars.values())]
        ds.attrs["processed"] = True

        # Save
        ds = ds.drop_duplicates("time")
        ds = ds.sortby("time")
        ds = ds.chunk(chunks={"time": 250_000})
        encoding = {x: {"compressor": compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=raw_store,
            group=metadata["group"],
            encoding=encoding,
            consolidated=False,
        )


if __name__ == "__main__":
    d = LoadElectricDoubleProbes(data_type="dce")
    d.download()
