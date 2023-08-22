import os
import warnings

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.directory import zarr_store

from .download import download_file
from .base import BaseDownload

warnings.filterwarnings("ignore")


class DownloadFGM(BaseDownload):

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
        )

    @staticmethod
    def process(file: str):
        temp_file = download_file(file)

        ds = cdf_to_xarray(temp_file, to_datetime=True, fillval_to_nan=True)
        file_name = ds.attrs["Logical_file_id"][0]
        (
            probe, instrument, data_rate, data_level, trange_id, _
        ) = file_name.split("_")
        pfx = f"{probe}_{instrument}"
        sfx = f"{data_rate}_{data_level}"
        for key in ["units", "UNITS", "FILLVAL", "VALIDMAX", "VALIDMIN"]:
            if key in ds.Epoch.attrs:
                del ds.Epoch.attrs[key]
            if key in ds.Epoch_state.attrs:
                del ds.Epoch_state.attrs[key]

        # Split dataset into ephemeris and field data
        fgm_vars = {
            f"{pfx}_{var}_{sfx}": var for var in [
                "b_gse",
                "b_gsm",
                "flag",
                "hirange",
                "stemp",
                "etemp",
            ]
        }
        ds_fgm = ds[fgm_vars.keys()].rename(Epoch="time", **fgm_vars)
        ds_fgm.to_zarr(
            store=zarr_store,
            group=f"/{probe}/fgm/{data_rate}/{trange_id}",
            mode="w",
        )

        eph_vars = {
            f"{pfx}_{var}_{sfx}": var for var in ["r_gse", "r_gsm"]
        }
        ds_eph = ds[eph_vars.keys()].rename(Epoch_state="time", **eph_vars)
        ds_eph.to_zarr(
            store=zarr_store,
            group=f"/{probe}/fgm-eph/{data_rate}/{trange_id}",
            mode="w",
        )

        os.unlink(temp_file)
        return f"Processed {file_name}"


if __name__ == "__main__":
    #d = DownloadFGM(end_date="2017-07-30", probe=["mms1", "mms2","mms3",
    #                                              "mms4"])
    d = DownloadFGM(end_date="2017-08-05")
    d.download()
