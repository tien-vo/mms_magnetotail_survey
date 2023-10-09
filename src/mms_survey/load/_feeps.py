__all__ = ["LoadFEEPS"]

import os
import warnings

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.directory import zarr_compressor, zarr_store

from .base import BaseLoader
from .download import download
from .utils import fix_epoch_metadata

warnings.filterwarnings("ignore")

eyes = dict(
    ion=[6, 7, 8],
    elc=[1, 2, 3, 4, 5, 9, 10, 11, 12],
)


class LoadFEEPS(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "ion",
        data_level: str = "l2",
    ):
        super().__init__(
            instrument="feeps",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_type=data_type,
            data_level=data_level,
            query_type="science",
        )

    def process(self, file: str):
        # Extract some metadata from file name
        probe, instrument, data_rate, level, data_type, tid, _ = file.split(
            "_"
        )
        species = "ion" if data_type == "ion" else "elc"
        pfx = f"{probe}_epd_{instrument}_{data_rate}_{level}_{data_type}"
        if instrument != "feeps":
            return f"File {file} is not in FEEPS dataset!"

        # Download file into temporary file
        url = f"{self.server}/download/{self.query_type}?file={file}"
        temp_file = download(url)
        if temp_file is None:
            return f"Issue encountered! {file} was not processed!"

        # Fix dataset dimension & metadata
        ds = fix_epoch_metadata(
            cdf_to_xarray(temp_file, to_datetime=True, fillval_to_nan=True),
            vars=["epoch"],
        ).reset_coords()

        # Rename variables and remove unwanted variables
        ds = ds.rename(
            vars := {
                "epoch": "time",
                **{
                    f"{pfx}_{head}_{var}_sensorid_{eye}": f"{var}_{head[0]}{eye}"
                    for head in ["top", "bottom"]
                    for eye in eyes[species]
                    for var in [
                        "quality_indicator",
                        "energy_centroid",
                        "count_rate",
                        "intensity",
                        "sector_mask",
                        "sun_contamination",
                        "percent_error",
                    ]
                },
                **{
                    f"{pfx}_{var}": var
                    for var in [
                        "spin",
                        "spinsectnum",
                        "integration_sectors",
                        "spin_duration",
                        "pitch_angle",
                    ]
                },
            }
        ).drop_dims(["dim0", "dim3", "dim4"])
        ds = ds.rename_dims(dict(dim1="mask", dim2="space"))
        ds = ds.assign_coords({"space": ["x", "y", "z"]})
        ds = ds[list(vars.values())]

        # Save
        encoding = {x: {"compressor": zarr_compressor} for x in ds}
        ds.to_zarr(
            store=zarr_store,
            group=f"/{probe}/feeps_{species}_dist/{data_rate}/{tid}",
            mode="w",
            encoding=encoding,
            consolidated=False,
        )

        os.unlink(temp_file)
        return f"Processed {file}"


if __name__ == "__main__":
    d = LoadFEEPS(data_type="electron")
    d.download()
