__all__ = ["LoadAncillary"]

import os

import numpy as np
import pandas as pd
import requests
import xarray as xr
import zarr

from mms_survey.utils.io import compressor, dataset_is_ok, store
from mms_survey.utils.download import download

from .base import BaseLoader


class LoadAncillary(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        product: str | list = "defq",
        skip_ok_dataset: bool = False,
    ):
        super().__init__(
            instrument=None,
            start_date=start_date,
            end_date=end_date,
            probe=None,
            data_rate=None,
            data_type=None,
            data_level=None,
            query_type="ancillary",
            skip_ok_dataset=skip_ok_dataset,
        )
        self.product = product

    @property
    def product(self) -> str:
        return ",".join(self._product)

    @product.setter
    def product(self, product: str | list):
        assert isinstance(
            product, (str, list)
        ), "Incorrect type for `product` input"
        if isinstance(product, str):
            product = [product]

        self._product = product

    @property
    def _id(self) -> str:
        return f"({self.product})"

    @property
    def query_string(self) -> str:
        return f"{super().query_string}&product={self.product}"

    def process(self, file: str):
        # Extract some metadata from file name
        _, product, start, end = os.path.splitext(file)[0].split("_")
        group = f"/ancillary/{product.lower()}/{start}_{end}"
        if self.skip_ok_dataset and dataset_is_ok(group):
            return f"{file} already processed. Skipping..."

        # Download file into temporary file
        url = f"{self.server}/download/{self.query_type}?file={file}"
        temp_file = download(url)
        if temp_file is None:
            return f"Issue encountered! {file} was not processed!"

        # Read TQF
        f = np.loadtxt(temp_file, skiprows=11, dtype=str)
        time = pd.to_datetime(f[:, 0], format="%Y-%j/%H:%M:%S.%f")
        tqf = xr.DataArray(
            f[:, 2].astype("f4"),
            coords=dict(time=time),
            name="tqf",
        )

        # Save
        encoding = dict(tqf={"compressor": compressor})
        tqf.to_zarr(
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
    d = LoadAncillary(end_date="2017-07-27")
    d.download()
