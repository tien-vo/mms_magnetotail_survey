__all__ = ["LoadMagneticEphemerisCoordinates"]

import os
import logging

import zarr
from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import compressor, fix_epoch_metadata, store

from .base import BaseLoader


class LoadMagneticEphemerisCoordinates(BaseLoader):
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
            instrument="mec",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_type="epht89q",
            data_level=data_level,
            product=None,
            query_type="science",
            skip_processed_data=skip_processed_data,
        )

    def get_metadata(self, file: str) -> dict:
        name = os.path.splitext(file)[0]
        probe, instrument, data_rate, _, _, tid, _ = name.split("_")
        return {
            "file": file,
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "group": f"/{data_rate}/{instrument}/{probe}/{tid}",
        }

    def process_file(self, file: str, metadata: str):
        if metadata["instrument"] != self.instrument:
            logging.warning(f"{file} is not in MEC dataset!")
            return

        # Load file and fix metadata
        ds = cdf_to_xarray(file, to_datetime=True, fillval_to_nan=True)
        ds = fix_epoch_metadata(ds, vars=["Epoch"]).reset_coords()
        ds = ds.rename_dims(dict(dim0="quaternion", dim2="space"))
        ds = ds.assign_coords(
            {
                "space": ["x", "y", "z"],
                "quaternion": ["x", "y", "z", "w"],
            }
        )

        # Rename variables and remove unwanted variables
        pfx = f"{metadata['probe']}_{metadata['instrument']}"
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                f"{pfx}_dipole_tilt": "dipole_tilt",
                f"{pfx}_kp": "kp",
                f"{pfx}_dst": "dst",
                f"{pfx}_r_eci": "R_eci",
                f"{pfx}_v_eci": "V_eci",
                f"{pfx}_quat_eci_to_bcs": "Q_eci_to_bcs",
                f"{pfx}_quat_eci_to_dbcs": "Q_eci_to_dbcs",
                f"{pfx}_quat_eci_to_dsl": "Q_eci_to_dsl",
                f"{pfx}_quat_eci_to_gse": "Q_eci_to_gse",
                f"{pfx}_quat_eci_to_gsm": "Q_eci_to_gsm",
            }
        )
        ds = ds[list(vars.values())].pint.quantify()
        ds.attrs["processed"] = True

        # Save
        encoding = {x: {"compressor": compressor} for x in ds}
        ds.pint.dequantify().to_zarr(
            mode="w",
            store=store,
            group=metadata["group"],
            encoding=encoding,
            consolidated=False,
        )


if __name__ == "__main__":
    d = LoadMagneticEphemerisCoordinates()
    d.download()
