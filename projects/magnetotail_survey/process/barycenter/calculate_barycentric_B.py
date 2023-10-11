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

J_unit = u.Unit("nA m-2")


def process(tid: str):
    if dataset_is_processed(
        group=f"/srvy/fgm/barycenter/{tid}",
        store=processed_store
    ):
        print(f"Already processed {tid=}, skipping...")
        return

    kw = dict(store=raw_store, consolidated=False)
    try:
        ds_mms1 = xr.open_zarr(group=f"/srvy/fgm/mms1/{tid}", **kw)
        ds_mms2 = xr.open_zarr(group=f"/srvy/fgm/mms2/{tid}", **kw)
        ds_mms3 = xr.open_zarr(group=f"/srvy/fgm/mms3/{tid}", **kw)
        ds_mms4 = xr.open_zarr(group=f"/srvy/fgm/mms4/{tid}", **kw)
        ds_eph_mms1 = xr.open_zarr(group=f"/srvy/fgm_eph/mms1/{tid}", **kw)
        ds_eph_mms2 = xr.open_zarr(group=f"/srvy/fgm_eph/mms2/{tid}", **kw)
        ds_eph_mms3 = xr.open_zarr(group=f"/srvy/fgm_eph/mms3/{tid}", **kw)
        ds_eph_mms4 = xr.open_zarr(group=f"/srvy/fgm_eph/mms4/{tid}", **kw)
    except zarr.errors.GroupNotFoundError:
        return

    # Calculate J_clm
    ds_clm = curlometer(
        Q_name="B",
        Q1=ds_mms1.B_gsm,
        Q2=ds_mms2.B_gsm,
        Q3=ds_mms3.B_gsm,
        Q4=ds_mms4.B_gsm,
        R1=ds_eph_mms1.R_gsm,
        R2=ds_eph_mms2.R_gsm,
        R3=ds_eph_mms3.R_gsm,
        R4=ds_eph_mms4.R_gsm,
    )
    div_B = ds_clm.div_B.values * u.Unit(ds_clm.div_B.attrs["units"])
    curl_B = ds_clm.curl_B.values * u.Unit(ds_clm.curl_B.attrs["units"])
    J_clm = (curl_B / c.si.mu0).to(J_unit).value
    J_clm_err = (div_B / c.si.mu0).to(J_unit).value
    ds_clm = ds_clm.assign(
        J_clm=(["time", "space"], J_clm, {"units": str(J_unit)}),
        J_clm_err=("time", J_clm_err, {"units": str(J_unit)})
    )
    ds_clm.attrs["processed"] = True

    # Save
    ds_clm = ds_clm.chunk(chunks={"time": 125_000})
    encoding = {var: {"compressor": compressor} for var in ds_clm}
    ds_clm.to_zarr(
        mode="w",
        store=processed_store,
        group=f"/srvy/fgm/barycenter/{tid}",
        encoding=encoding,
        consolidated=False,
    )
    print(f"Processed barycentric fgm quantities for {tid=}")


if __name__ == "__main__":
    f = zarr.open(raw_store)
    group_names = f[f"/srvy/fgm/mms1"].groups()
    #process("20170630")
    with Pool() as pool:
        for _ in pool.uimap(lambda group: process(group[0]), group_names):
            pass
