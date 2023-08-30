import zarr
import xarray as xr
import astropy.units as u
import astropy.constants as c

from pathos.pools import ProcessPool as Pool

from mms_survey.utils.curlometer import curlometer
from mms_survey.utils.io import store, compressor


def process(tid: str):
    kw = dict(store=store, consolidated=False)
    ds_mms1_fgm = xr.open_zarr(group=f"/{fgm_rate}/fgm/mms1/{tid}", **kw)
    ds_mms2_fgm = xr.open_zarr(group=f"/{fgm_rate}/fgm/mms2/{tid}", **kw)
    ds_mms3_fgm = xr.open_zarr(group=f"/{fgm_rate}/fgm/mms3/{tid}", **kw)
    ds_mms4_fgm = xr.open_zarr(group=f"/{fgm_rate}/fgm/mms4/{tid}", **kw)
    ds_mms1_edp = xr.open_zarr(group=f"/{edp_rate}/edp/dce/mms1/{tid}", **kw)
    ds_mms2_edp = xr.open_zarr(group=f"/{edp_rate}/edp/dce/mms2/{tid}", **kw)
    ds_mms3_edp = xr.open_zarr(group=f"/{edp_rate}/edp/dce/mms3/{tid}", **kw)
    ds_mms4_edp = xr.open_zarr(group=f"/{edp_rate}/edp/dce/mms4/{tid}", **kw)

    # Calculate clm
    ds_clm = curlometer(
        Q_name="E",
        Q1=ds_mms1_edp.E_gsm,
        Q2=ds_mms2_edp.E_gsm,
        Q3=ds_mms3_edp.E_gsm,
        Q4=ds_mms4_edp.E_gsm,
        R1=ds_mms1_fgm.R_gsm,
        R2=ds_mms2_fgm.R_gsm,
        R3=ds_mms3_fgm.R_gsm,
        R4=ds_mms4_fgm.R_gsm,
    )

    # Save
    encoding = {var: {"compressor": compressor} for var in ds_clm}
    ds_clm.attrs["ok"] = True
    ds_clm.to_zarr(
        mode="w",
        store=store,
        group=f"/{edp_rate}/edp/dce/barycenter/{tid}",
        encoding=encoding,
        consolidated=False,
    )
    print(f"Processed barycentric edp quantities for {tid=}")

if __name__ == "__main__":
    edp_rate = "fast"
    fgm_rate = "srvy"

    f = zarr.open(store)
    #process("20170726")
    with Pool() as pool:
        for _ in pool.uimap(
            lambda group: process(group[0]),
            f[f"/{edp_rate}/edp/dce/mms1"].groups(),
        ):
            pass
