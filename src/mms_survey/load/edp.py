__all__ = ["LoadElectricDoubleProbes"]

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
from mms_survey.utils.download import download

from .base import BaseLoader

warnings.filterwarnings("ignore")


class LoadElectricDoubleProbes(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "dce",
        data_level: str = "l2",
        skip_ok_dataset: bool = False,
    ):
        super().__init__(
            instrument="edp",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate="fast" if data_rate == "srvy" else data_rate,
            data_type=data_type,
            data_level=data_level,
            query_type="science",
            skip_ok_dataset=skip_ok_dataset,
        )

    def process(self, file: str):
        # Extract some metadata from file name
        (probe, instrument, data_rate, level, data_type, tid, _) = file.split(
            "_"
        )
        group = f"/{instrument}_{data_type}/{data_rate}/{probe}/{tid}"
        pfx = f"{probe}_{instrument}"
        sfx = f"{data_rate}_{level}"
        if instrument != "edp":
            return f"File {file} is not in EDP dataset!"
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
            vars=[f"{pfx}_epoch_{sfx}"],
        ).reset_coords()

        if data_type == "dce":
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
        elif data_type == "scpot":
            ds = ds.drop_dims("dim0").rename(
                vars := {
                    f"{pfx}_epoch_{sfx}": "time",
                    f"{pfx}_scpot_{sfx}": "Phi_sc",
                }
            )
        else:
            raise NotImplementedError()

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
    d = LoadElectricDoubleProbes(data_type="dce")
    d.download()
