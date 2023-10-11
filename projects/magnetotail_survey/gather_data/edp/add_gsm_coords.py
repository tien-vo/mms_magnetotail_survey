import zarr
import numpy as np
import xarray as xr

from pathos.pools import ThreadPool as Pool

from mms_survey.utils.cotrans import quaternion_rotate
from mms_survey.utils.io import raw_store, compressor


def process(group: str):
    _, _, _, _, probe, tid = group.split("/")
    mec_group = f"/srvy/mec/{probe}/{tid}"
    ds_edp = xr.open_zarr(raw_store, group=group, consolidated=False)
    ds_mec = xr.open_zarr(raw_store, group=mec_group, consolidated=False)
    encoding = dict()

    kw = dict(time=ds_edp.time, kwargs=dict(fill_value=np.nan))
    Q_eci_to_gse = ds_mec.Q_eci_to_gse.interp(**kw)
    Q_eci_to_gsm = ds_mec.Q_eci_to_gsm.interp(**kw)

    if "E_gsm" not in ds_edp:
        E_eci = quaternion_rotate(ds_edp.E_gse, Q_eci_to_gse, inverse=True)
        E_gsm = quaternion_rotate(E_eci, Q_eci_to_gsm)
        E_gsm.name = "E_gsm"
        E_gsm.attrs.update(standard_name="E GSM")
        E_gsm.attrs.pop("FIELDNAM")
        encoding["E_gsm"] = {"compressor": compressor}
        ds_edp = ds_edp.assign(E_gsm=E_gsm)

    # Save
    ds_edp = ds_edp.chunk(chunks={"time": 250_000})
    ds_edp.to_zarr(
        mode="a",
        store=raw_store,
        group=group,
        encoding=encoding,
        consolidated=False,
    )
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
