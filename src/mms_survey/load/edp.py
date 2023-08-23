__all__ = ["LoadEDP"]

import os
import warnings

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.directory import zarr_store, zarr_compressor

from .base import BaseLoader
from .download import download
from .utils import fix_epoch_metadata

warnings.filterwarnings("ignore")


class LoadEDP(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_type: str = "dce",
        data_level: str = "l2",
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
        )

    def process(self, file: str):
        # Extract some metadata from file name
        (
            probe, instrument, data_rate, level, data_type, tid, _
        ) = file.split("_")
        pfx = f"{probe}_{instrument}"
        sfx = f"{data_rate}_{level}"
        if instrument != "edp":
            return f"File {file} is not in EDP dataset!"

        # Download file into temporary file
        url = f"{self.server}/download/{self.query_type}?file={file}"
        temp_file = download(url)
        if temp_file is None:
            return f"Issue encountered! {file} was not processed!"

        # Fix dataset dimension & metadata
        ds = fix_epoch_metadata(
            cdf_to_xarray(temp_file, to_datetime=True, fillval_to_nan=True),
            vars=[f"{pfx}_epoch_{sfx}"],
        )

        if data_type == "dce":
            vars = {
                f"{pfx}_{var}_{sfx}": var
                for var in [
                    "dce_gse",
                    "dce_par_epar",
                    "dce_err",
                    "bitmask",
                ]
            }
        elif data_type == "scpot":
            vars = {f"{pfx}_{var}_{sfx}": var for var in ["scpot"]}
        else:
            raise NotImplementedError()

        ds = ds[vars.keys()].rename({f"{pfx}_epoch_{sfx}": "time", **vars})
        ds.to_zarr(
            store=zarr_store,
            group=f"/{probe}/{instrument}-{data_type}/{data_rate}/{tid}",
            mode="w",
            consolidated=False,
        )

        os.unlink(temp_file)
        return f"Processed {file}"


if __name__ == "__main__":
    d = LoadEDP(data_type="dce")
    d.download()
