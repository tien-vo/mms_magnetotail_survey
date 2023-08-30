import zarr
import xarray as xr
import astropy.units as u
import astropy.constants as c

from pathos.pools import ProcessPool as Pool

from mms_survey.utils.curlometer import curlometer
from mms_survey.utils.io import store, compressor


def process(tid: str):
    kw = dict(store=store, consolidated=False)
    ds_mms1 = xr.open_zarr(group=f"/{data_rate}/fgm/mms1/{tid}", **kw)
    ds_mms2 = xr.open_zarr(group=f"/{data_rate}/fgm/mms2/{tid}", **kw)
    ds_mms3 = xr.open_zarr(group=f"/{data_rate}/fgm/mms3/{tid}", **kw)
    ds_mms4 = xr.open_zarr(group=f"/{data_rate}/fgm/mms4/{tid}", **kw)

    # Calculate J_clm
    ds_clm = curlometer(
        Q_name="B",
        Q1=ds_mms1.B_gsm,
        Q2=ds_mms2.B_gsm,
        Q3=ds_mms3.B_gsm,
        Q4=ds_mms4.B_gsm,
        R1=ds_mms1.R_gsm,
        R2=ds_mms2.R_gsm,
        R3=ds_mms3.R_gsm,
        R4=ds_mms4.R_gsm,
    )
    J_unit = "nA m-2"
    div_B = ds_clm.div_B.values * u.Unit(ds_clm.div_B.attrs["units"])
    curl_B = ds_clm.curl_B.values * u.Unit(ds_clm.curl_B.attrs["units"])
    J_clm = (curl_B / c.si.mu0).to(u.Unit(J_unit)).value
    J_clm_err = (div_B / c.si.mu0).to(u.Unit(J_unit)).value
    ds_clm = ds_clm.assign(J_clm=(["time", "space"], J_clm, {"units": J_unit}))
    ds_clm = ds_clm.assign(J_clm_err=("time", J_clm_err, {"units": J_unit}))

    # Save
    encoding = {var: {"compressor": compressor} for var in ds_clm}
    ds_clm.attrs["ok"] = True
    ds_clm.to_zarr(
        mode="w",
        store=store,
        group=f"/{data_rate}/fgm/barycenter/{tid}",
        encoding=encoding,
        consolidated=False,
    )
    print(f"Processed barycentric fgm quantities for {tid=}")

if __name__ == "__main__":
    data_rate = "srvy"

    f = zarr.open(store)
    #process("20170726")
    with Pool() as pool:
        for _ in pool.uimap(
            lambda group: process(group[0]),
            f[f"/{data_rate}/fgm/mms1"].groups(),
        ):
            pass
