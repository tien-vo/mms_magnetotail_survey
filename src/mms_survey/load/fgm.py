__all__ = ["LoadFGM"]

import os
import warnings

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.directory import zarr_store, zarr_compressor

from .base import BaseLoader
from .download import download
from .utils import fix_epoch_metadata

warnings.filterwarnings("ignore")


class LoadFGM(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_level: str = "l2",
    ):
        super().__init__(
            instrument="fgm",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_level=data_level,
            query_type="science",
        )

    def process(self, file: str):
        # Extract some metadata from file name
        probe, instrument, data_rate, level, tid, _ = file.split("_")
        pfx = f"{probe}_{instrument}"
        sfx = f"{data_rate}_{level}"
        if instrument != "fgm":
            return f"File {file} is not in FGM dataset!"

        # Download file into temporary file
        url = f"{self.server}/download/{self.query_type}?file={file}"
        temp_file = download(url)
        if temp_file is None:
            return f"Issue encountered! {file} was not processed!"

        # Fix dataset dimension & metadata
        ds = fix_epoch_metadata(
            cdf_to_xarray(temp_file, to_datetime=True, fillval_to_nan=True),
            vars=["Epoch", "Epoch_state"],
        ).reset_coords()
        ds = ds.rename_dims(dict(dim0="space"))
        ds = ds.assign_coords({"space": ["x", "y", "z", "mag"]})

        # Rename variables and remove unwanted variables
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                "Epoch_state": "ephemeris_time",
                f"{pfx}_b_gse_{sfx}": "B_gse",
                f"{pfx}_b_gsm_{sfx}": "B_gsm",
                f"{pfx}_r_gse_{sfx}": "R_gse",
                f"{pfx}_r_gsm_{sfx}": "R_gsm",
                f"{pfx}_flag_{sfx}": "flag",
            }
        )
        ds = ds[list(vars.values())]

        # Save
        encoding = {x: {"compressor": zarr_compressor} for x in ds}
        ds.to_zarr(
            store=zarr_store,
            group=f"/{probe}/{instrument}/{data_rate}/{tid}",
            mode="w",
            encoding=encoding,
            consolidated=False,
        )

        os.unlink(temp_file)
        return f"Processed {file}"


if __name__ == "__main__":
    d = LoadFGM()
    d.download()
