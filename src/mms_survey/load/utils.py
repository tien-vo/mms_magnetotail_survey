__all__ = ["fix_epoch_metadata"]

import numpy as np
import pandas.api.types as pd


def fix_epoch_metadata(
    ds,
    vars=[
        "Epoch",
    ],
):
    for var in vars:
        for key_to_remove in ["units", "UNITS"]:
            if key_to_remove in ds[var].attrs:
                del ds[var].attrs[key_to_remove]

        for key, value in ds[var].attrs.items():
            if pd.is_datetime64_dtype(value):
                ds[var].attrs[key] = value.astype(str)

    return ds
