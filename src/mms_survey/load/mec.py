import os
import warnings

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.directory import zarr_store

from .base import BaseDownload
from .download import download_file
from .utils import fix_epoch_metadata

warnings.filterwarnings("ignore")


class DownloadMEC(BaseDownload):

    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_level: str = "l2",
    ):
        super().__init__(
            instrument="mec",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_type="epht89q",
            data_level=data_level,
        )

    @staticmethod
    def process(file: str):
        temp_file = download_file(file)
        if temp_file is None:
            return f"Issue processing {file}. File was not processed"

        ds = fix_epoch_metadata(
            cdf_to_xarray(temp_file, to_datetime=True, fillval_to_nan=True),
            vars=["Epoch"])
        probe, instrument, data_rate, _, data_type, tid, _ = file.split("_")
        pfx = f"{probe}_{instrument}"

        vars = {
            f"{pfx}_{var}": var for var in [
                "dipole_tilt",
                "kp",
                "dst",
                "r_eci",
                "v_eci",
                "quat_eci_to_bcs",
                "quat_eci_to_dbcs",
                "quat_eci_to_dmpa",
                "quat_eci_to_dsl",
                "quat_eci_to_gse",
                "quat_eci_to_gsm",
            ]
        }
        ds = ds[vars.keys()].rename(Epoch="time", **vars)
        ds.to_zarr(
            store=zarr_store,
            group=f"/{probe}/mec-{data_type}/{data_rate}/{tid}",
            mode="w",
            consolidated=False,
        )

        os.unlink(temp_file)
        return f"Processed {file}"


if __name__ == "__main__":
    d = DownloadMEC()
    d.download()
