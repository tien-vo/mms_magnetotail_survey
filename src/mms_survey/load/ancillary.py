__all__ = ["DownloadANCIL"]

import os

import psutil
import requests
import numpy as np
import pandas as pd
import xarray as xr

from mms_survey.utils.directory import zarr_store

from .download import download_file
from .base import BaseDownload, server, BYTE_TO_GIGABYTE


class DownloadANCIL(BaseDownload):
    def get_remote_files(self, check_size=False) -> list:
        url = f"{server}/file_info/ancillary?{self._query_string}"
        response = requests.get(url)
        assert response.ok, "Query error!"

        files = response.json()["files"]
        remote_file_names = list(map(lambda x: x["file_name"], files))
        download_size = sum(list(map(lambda x: x["file_size"], files)))
        free_disk_space = psutil.disk_usage("/home").free
        print(
            f"{self._id}: Query found {len(files)} files with total size = "
            f"{BYTE_TO_GIGABYTE * download_size:.4f} GB"
        )
        if check_size:
            assert download_size < 0.95 * free_disk_space, "Download size too large"

        return remote_file_names

    @staticmethod
    def process(file: str):
        temp_file = download_file(file, data_type="ancillary")
        if temp_file is None:
            return f"Issue processing {file}. File was not processed"

        _, product, start_date, end_date = os.path.splitext(file)[0].split("_")

        f = np.loadtxt(temp_file, skiprows=11, dtype=str)
        time = pd.to_datetime(f[:, 0], format="%Y-%j/%H:%M:%S.%f")
        tqf = xr.DataArray(
            f[:, 2].astype("f4"),
            coords=(time,),
            dims=("time",),
            name="tqf",
        )
        tqf.to_zarr(
            store=zarr_store,
            group=f"/ancillary/{product}/{start_date}_{end_date}",
            mode="w",
        )

        os.unlink(temp_file)
        return f"Processed {file}"

    @property
    def _query_string(self) -> str:
        s = f"product=defq&start_date={self.start_date}&end_date={self.end_date}"
        return s


if __name__ == "__main__":
    d = DownloadANCIL(end_date="2017-07-27")
    d.download()
