import warnings

# Ignore cdflib.xarray.cdf_to_xarray timestamp precision warning
warnings.filterwarnings(
    "ignore",
    message=(
        "Converting non-nanosecond precision datetime values "
        "to nanosecond precision."
    ),
)
