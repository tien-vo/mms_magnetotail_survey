__all__ = ["LoadFluxGateMagnetometer"]

import os
import logging

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import compressor, raw_store

from .base import BaseLoader
from .utils import process_epoch_metadata


class LoadFluxGateMagnetometer(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_level: str = "l2",
        skip_processed_data: bool = False,
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
            skip_processed_data=skip_processed_data,
        )

    def get_metadata(self, file: str) -> dict:
        name = os.path.splitext(file)[0]
        probe, instrument, data_rate, data_level, tid, _ = name.split("_")
        return {
            "file": file,
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_level": data_level,
            "group": f"/{data_rate}/{instrument}/{probe}/{tid}",
        }

    def process_file(self, file: str, metadata: dict):
        if metadata["instrument"] != self.instrument:
            logging.warning(f"{file} is not in FGM dataset!")
            return

        # Load file and fix metadata
        ds = cdf_to_xarray(file, to_datetime=True, fillval_to_nan=True)
        ds = fix_epoch_metadata(ds, vars=["Epoch", "Epoch_state"])
        ds = ds.reset_coords()
        ds = ds.rename_dims(dict(dim0="space"))
        ds = ds.assign_coords({"space": ["x", "y", "z", "mag"]})

        # Rename variables and remove unwanted variables
        pfx = f"{metadata['probe']}_{metadata['instrument']}"
        sfx = f"{metadata['data_rate']}_{metadata['data_level']}"
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
        ds = ds[list(vars.values())].pint.quantify()

        # Remove some unnecessary data attributes
        keys_to_remove = [
            "UNITS",
            "DEPEND_0",
            "DISPLAY_TYPE",
            "FIELDNAM",
            "FORMAT",
            "LABL_PTR_1",
            "LABL_AXIS",
            "long_name",
            "REPRESENTATION_1",
        ]
        for var in ds:
            for key in keys_to_remove:
                if key in ds[var].attrs:
                    del ds[var].attrs[key]

        ds["B_gse"].attrs["standard_name"] = "B GSE"
        ds["B_gsm"].attrs["standard_name"] = "B GSM"
        ds["R_gse"].attrs["standard_name"] = "R GSE"
        ds["R_gsm"].attrs["standard_name"] = "R GSM"
        ds["flag"].attrs["standard_name"] = "Flag"
        ds.attrs["processed"] = True

        # Save
        ds.attrs["start_date"] = str(ds.time.values[0])
        ds.attrs["end_date"] = str(ds.time.values[-1])
        encoding = {x: {"compressor": compressor} for x in ds}
        ds = ds.chunk(chunks={"time": 250_000})
        ds.pint.dequantify().to_zarr(
            mode="w",
            store=raw_store,
            group=metadata["group"],
            encoding=encoding,
            consolidated=False,
        )


if __name__ == "__main__":
    d = LoadFluxGateMagnetometer()
    d.download()
