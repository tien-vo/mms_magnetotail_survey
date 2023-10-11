import zarr
import xarray as xr
import astropy.units as u
import astropy.constants as c

from pathos.pools import ThreadPool as Pool

from mms_survey.utils.io import raw_store, compressor


def process(group: str):
    ds = xr.open_zarr(raw_store, group=group, consolidated=False)
    ds = ds[["E_err", "E_gse", "bitmask"]]
    ds.to_zarr(mode="w", store=raw_store, group=group, consolidated=False)
    print(f"Processed {group}")


if __name__ == "__main__":
    f = zarr.open(raw_store)
    #process("/fast/edp/dce/mms1/20170726")
    with Pool() as pool:
        for probe in ["mms1", "mms2", "mms3", "mms4"]:
            for _ in pool.uimap(
                lambda group: process(group[1].name),
                f[f"/fast/edp/dce/{probe}"].groups(),
            ):
                pass
