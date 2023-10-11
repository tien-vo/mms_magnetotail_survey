import zarr
import xarray as xr
import astropy.units as u
import astropy.constants as c

from pathos.pools import ThreadPool as Pool

from mms_survey.utils.curlometer import curlometer
from mms_survey.utils.io import (
    raw_store,
    processed_store,
    compressor,
    dataset_is_processed,
)


def process(tid: str):
    if dataset_is_processed(
        group=f"/fast/edp/dce/barycenter/{tid}",
        store=processed_store
    ):
        print(f"Already processed {tid=}, skipping...")
        return

    kw = dict(store=raw_store, consolidated=False)
    try:
        ds_edp_mms1 = xr.open_zarr(group=f"/fast/edp/dce/mms1/{tid}", **kw)
        ds_edp_mms2 = xr.open_zarr(group=f"/fast/edp/dce/mms2/{tid}", **kw)
        ds_edp_mms3 = xr.open_zarr(group=f"/fast/edp/dce/mms3/{tid}", **kw)
        ds_edp_mms4 = xr.open_zarr(group=f"/fast/edp/dce/mms4/{tid}", **kw)
        ds_eph_mms1 = xr.open_zarr(group=f"/srvy/fgm_eph/mms1/{tid}", **kw)
        ds_eph_mms2 = xr.open_zarr(group=f"/srvy/fgm_eph/mms2/{tid}", **kw)
        ds_eph_mms3 = xr.open_zarr(group=f"/srvy/fgm_eph/mms3/{tid}", **kw)
        ds_eph_mms4 = xr.open_zarr(group=f"/srvy/fgm_eph/mms4/{tid}", **kw)
    except zarr.errors.GroupNotFoundError:
        return

    ds_clm = curlometer(
        Q_name="E",
        Q1=ds_edp_mms1.E_gsm,
        Q2=ds_edp_mms2.E_gsm,
        Q3=ds_edp_mms3.E_gsm,
        Q4=ds_edp_mms4.E_gsm,
        R1=ds_eph_mms1.R_gsm,
        R2=ds_eph_mms2.R_gsm,
        R3=ds_eph_mms3.R_gsm,
        R4=ds_eph_mms4.R_gsm,
    )
    ds_clm.attrs["processed"] = True

    # Save
    ds_clm = ds_clm.chunk(chunks={"time": 125_000})
    encoding = {var: {"compressor": compressor} for var in ds_clm}
    ds_clm.to_zarr(
        mode="w",
        store=processed_store,
        group=f"/fast/edp/dce/barycenter/{tid}",
        encoding=encoding,
        consolidated=False,
    )
    print(f"Processed barycentric edp quantities for {tid=}")


if __name__ == "__main__":
    f = zarr.open(raw_store)
    group_names = f[f"/fast/edp/dce/mms1"].groups()
    #process("20170726")
    with Pool() as pool:
        for _ in pool.uimap(lambda group: process(group[0]), group_names):
            pass
