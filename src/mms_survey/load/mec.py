__all__ = ["LoadMEC"]

import os
import warnings

import zarr
from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import (
    compressor,
    dataset_is_ok,
    fix_epoch_metadata,
    store,
)

from .base import BaseLoader
from .download import download

warnings.filterwarnings("ignore")


class LoadMEC(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_level: str = "l2",
        skip_ok_dataset: bool = False,
    ):
        super().__init__(
            instrument="mec",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_type="epht89q",
            data_level=data_level,
            query_type="science",
            skip_ok_dataset=skip_ok_dataset,
        )

    def process(self, file: str):
        # Extract some metadata from file name
        probe, instrument, data_rate, _, data_type, tid, _ = file.split("_")
        group = f"/{instrument}/{data_rate}/{probe}/{tid}"
        pfx = f"{probe}_{instrument}"
        if instrument != "mec":
            return f"File {file} is not in MEC dataset!"
        if self.skip_ok_dataset and dataset_is_ok(group):
            return f"{file} already processed. Skipping..."

        # Download file into temporary file
        url = f"{self.server}/download/{self.query_type}?file={file}"
        temp_file = download(url)
        if temp_file is None:
            return f"Issue encountered! {file} was not processed!"

        # Fix dataset dimension & metadata
        ds = fix_epoch_metadata(
            cdf_to_xarray(temp_file, to_datetime=True, fillval_to_nan=True),
            vars=["Epoch"],
        ).reset_coords()
        ds = ds.rename_dims(dict(dim0="quaternion", dim2="space"))
        ds = ds.assign_coords(
            {
                "space": ["x", "y", "z"],
                "quaternion": ["x", "y", "z", "w"],
            }
        )

        # Rename variables and remove unwanted variables
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                f"{pfx}_dipole_tilt": "dipole_tilt",
                f"{pfx}_kp": "kp",
                f"{pfx}_dst": "dst",
                f"{pfx}_r_eci": "R_eci",
                f"{pfx}_v_eci": "V_eci",
                f"{pfx}_quat_eci_to_bcs": "Q_bcs",
                f"{pfx}_quat_eci_to_dbcs": "Q_dbcs",
                f"{pfx}_quat_eci_to_dsl": "Q_dsl",
                f"{pfx}_quat_eci_to_gse": "Q_gse",
                f"{pfx}_quat_eci_to_gsm": "Q_gsm",
            }
        )
        ds = ds[list(vars.values())]

        # Save
        encoding = {x: {"compressor": compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=store,
            group=group,
            encoding=encoding,
            consolidated=False,
        )

        os.unlink(temp_file)
        zarr_file = zarr.open(store)
        zarr_file[group].attrs["ok"] = True
        return f"Processed {file}"


if __name__ == "__main__":
    d = LoadMEC()
    d.download()
