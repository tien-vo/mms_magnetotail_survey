__all__ = ["LoadFPI"]

import os
import warnings

import numpy as np
from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.directory import zarr_compressor, zarr_store

from .base import BaseLoader
from .download import download
from .utils import fix_epoch_metadata

warnings.filterwarnings("ignore")


class LoadFPI(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "dis-dist",
        data_level: str = "l2",
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
        )

    def process(self, file: str):
        # Extract some metadata from file name
        (probe, instrument, data_rate, _, data_type, tid, _) = file.split("_")
        data_type = data_type.split("-")[0]
        species = "ion" if data_type == "dis" else "elc"
        pfx = f"{probe}_{data_type}"
        sfx = f"{data_rate}"
        if instrument != "fpi":
            return f"File {file} is not in FPI dataset!"

        # Download file into temporary file
        url = f"{self.server}/download/{self.query_type}?file={file}"
        temp_file = download(url)
        if temp_file is None:
            return f"Issue encountered! {file} was not processed!"

        # Fix dataset dimension & metadata
        ds = (
            fix_epoch_metadata(
                cdf_to_xarray(
                    temp_file, to_datetime=True, fillval_to_nan=True
                ),
                vars=["Epoch"],
            )
            .rename_dims({f"{pfx}_energy_{sfx}_dim": "energy"})
            .reset_coords()
        )

        # Center time stamps
        dt = np.int64(0.5e9 * (ds.Epoch_plus_var - ds.Epoch_minus_var).values)
        attrs = ds.Epoch.attrs
        ds = ds.assign_coords(Epoch=ds.Epoch + np.timedelta64(dt, "ns"))
        ds.Epoch.attrs.update({**attrs, "CATDESC": "Centered timestamp"})

        # Rename variables and remove unwanted variables
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                f"{pfx}_dist_{sfx}": "f3d",
                f"{pfx}_disterr_{sfx}": "f3d_err",
                f"{pfx}_phi_{sfx}": "phi",
                f"{pfx}_theta_{sfx}": "theta",
                f"{pfx}_energy_{sfx}": "energy",
                f"{pfx}_errorflags_{sfx}": "flag",
                f"{pfx}_compressionloss_{sfx}": "compression_loss",
            }
        )
        ds = ds[list(vars.values())].set_coords("energy")

        # Save
        encoding = {x: {"compressor": zarr_compressor} for x in ds}
        ds.to_zarr(
            store=zarr_store,
            group=f"/{probe}/fpi_{species}_dist/{data_rate}/{tid}",
            mode="w",
            encoding=encoding,
            consolidated=False,
        )

        os.unlink(temp_file)
        return f"Processed {file}"


if __name__ == "__main__":
    d = LoadFPI(data_type="dis-dist")
    d.download()
