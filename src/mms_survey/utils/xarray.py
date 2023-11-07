import astropy.units as u
import xarray as xr


def to_np(data: xr.DataArray) -> u.Quantity:
    return data.values * u.Unit(data.attrs["units"])


def to_da(data: u.Quantity) -> xr.DataArray:
    return xr.DataArray(data.value, attrs={"units": str(data.unit)})
