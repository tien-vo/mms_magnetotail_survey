import os
import warnings

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.directory import zarr_store

from .base import BaseDownload
from .download import download_file
from .utils import fix_epoch_metadata

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
        if temp_file is None:
            return f"Issue processing {file}. File was not processed"

        ds = fix_epoch_metadata(
            cdf_to_xarray(temp_file, to_datetime=True, fillval_to_nan=True),
            vars=["Epoch", "Epoch_state"])
        probe, instrument, data_rate, data_level, tid, _ = file.split("_")
        pfx = f"{probe}_{instrument}"
        sfx = f"{data_rate}_{data_level}"

        # Split dataset into ephemeris and field data
        fgm_vars = {
            f"{pfx}_{var}_{sfx}": var for var in ["b_gse", "b_gsm", "flag"]
        }
        ds_fgm = ds[fgm_vars.keys()].rename(Epoch="time", **fgm_vars)
        ds_fgm.to_zarr(
            store=zarr_store,
            group=f"/{probe}/fgm/{data_rate}/{tid}",
            mode="w",
            consolidated=False,
        )

        eph_vars = {
            f"{pfx}_{var}_{sfx}": var for var in ["r_gse", "r_gsm"]
        }
        ds_eph = ds[eph_vars.keys()].rename(Epoch_state="time", **eph_vars)
        ds_eph.to_zarr(
            store=zarr_store,
            group=f"/{probe}/fgm-eph/{data_rate}/{tid}",
            mode="w",
            consolidated=False,
        )

        os.unlink(temp_file)
        return f"Processed {file}"


if __name__ == "__main__":
    d = DownloadFGM()
    d.download()
