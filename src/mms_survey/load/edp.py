import os
import warnings

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.directory import zarr_store

from .download import download_file
from .base import BaseDownload

warnings.filterwarnings("ignore")


class DownloadEDP(BaseDownload):

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
        )

    def process(self, file: str):
        temp_file = download_file(file)
        pfx = f"{self.probe}_edp"
        sfx = f"{self.data_rate}_{self.data_level}"

        ds = cdf_to_xarray(temp_file, to_datetime=True, fillval_to_nan=True)
        for key in ["units", "UNITS", "FILLVAL", "VALIDMAX", "VALIDMIN"]:
            if key in ds[f"{pfx}_epoch_{sfx}"].attrs:
                del ds[f"{pfx}_epoch_{sfx}"].attrs[key]

        # Split dataset into ephemeris and field data
        trange_id = ds.attrs["Logical_file_id"][0].split("_")[-2]

        if self.data_type == "dce":
            vars = {
                f"{pfx}_{var}_{sfx}": var for var in [
                    "dce_gse",
                    "dce_dsl",
                    "dce_par_epar",
                    "dce_err",
                    "dce_dsl_res",
                    "dce_dsl_offset",
                    "dce_gain",
                    "dce_dsl_base",
                    "bitmask",
                    "quality",
                    "deltap",
                ]
            }
        elif self.data_type == "scpot":
            vars = {
                f"{pfx}_{var}_{sfx}": var for var in [
                    "scpot",
                    "psp",
                    "dcv",
                ]
            }
        else:
            raise NotImplementedError()

        ds = ds[vars.keys()].rename({f"{pfx}_epoch_{sfx}": "time", **vars})
        ds.to_zarr(
            store=zarr_store,
            group=(
                f"/{self.probe}/edp-{self.data_type}/"
                f"{self.data_rate}/{trange_id}"
            ),
            mode="w",
        )

        os.unlink(temp_file)


if __name__ == "__main__":
    d = DownloadEDP(data_type="dce")
    d.download()
