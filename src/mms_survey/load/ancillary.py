import os

import numpy as np
import pandas as pd
import requests
import xarray as xr

from mms_survey.utils.io import compressor, raw_store

from .base import BaseLoader


class LoadTetrahedronQualityFactor(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        product: str | list = "defq",
        skip_processed_data: bool = False,
    ):
        super().__init__(
            instrument=None,
            start_date=start_date,
            end_date=end_date,
            probe=None,
            data_rate=None,
            data_type=None,
            data_level=None,
            product=product,
            query_type="ancillary",
            skip_processed_data=skip_processed_data,
        )

    def get_metadata(self, file: str) -> dict:
        _, product, start, end = os.path.splitext(file)[0].split("_")
        return {
            "product": product,
            "start": start,
            "end": end,
            "group": f"/ancillary/tqf/{start}_{end}",
        }

    def process_file(self, file: str, metadata: dict):
        # Read file and create dataset
        data = np.loadtxt(file, skiprows=11, dtype=str)
        time = pd.to_datetime(data[:, 0], format="%Y-%j/%H:%M:%S.%f")
        ds = xr.Dataset(
            data_vars={
                "tqf": (["time"], data[:, 2].astype("f4")),
                "scale": (["time"], data[:, 3].astype("i4")),
            },
            coords={
                "time": time,
                "tai_epoch": data[:, 1].astype("f4"),
            },
            attrs={"processed": True},
        )

        # Save
        ds.attrs["start_date"] = str(ds.time.values[0])
        ds.attrs["end_date"] = str(ds.time.values[-1])
        encoding = {x: {"compressor": compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=raw_store,
            group=metadata["group"],
            encoding=encoding,
            consolidated=False,
        )


if __name__ == "__main__":
    d = LoadTetrahedronQualityFactor(end_date="2017-07-27")
    d.download()
