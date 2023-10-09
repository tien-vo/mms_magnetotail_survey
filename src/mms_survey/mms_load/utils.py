__all__ = [
    "process_epoch_metadata",
]

import numpy as np
import pandas.api.types as pd
import xarray as xr


def process_epoch_metadata(
    dataset: xr.Dataset, epoch_vars: list
) -> xr.Dataset:
    """
    Assuming CDF time conversion is handled correctly by cdflib. With this function
    we remove some unnecessary metadata.
    """
    keys_to_remove = [
        "units",
        "UNITS",
        "FIELDNAM",
        "LABLAXIS",
        "TIME_BASE",
        "TIME_SCALE",
        "long_name",
    ]
    for var in epoch_vars:
        for key in keys_to_remove:
            if key in dataset[var].attrs:
                del dataset[var].attrs[key]

        for key, value in dataset[var].attrs.items():
            if pd.is_datetime64_dtype(value):
                dataset[var].attrs[key] = value.astype(str)

        dataset[var].attrs["standard_name"] = "Time"

    return dataset
