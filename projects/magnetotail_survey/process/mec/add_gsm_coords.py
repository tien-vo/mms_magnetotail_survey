import zarr
import xarray as xr

from pathos.pools import ProcessPool as Pool

from mms_survey.utils.io import store, compressor
from mms_survey.utils.cotrans import quaternion_rotate


def process(group: str):
    ds = xr.open_zarr(store, group=group, consolidated=False)
    encoding = dict()

    if "R_gsm" not in ds:
        R_gsm = quaternion_rotate(ds.R_eci, ds.Q_eci_to_gsm)
        R_gsm.name = "R_gsm"
        R_gsm.attrs.update(
            CATDESC="GSM position vector",
            standard_name="R GSM",
        )
        R_gsm.attrs.pop("FIELDNAM")
        R_gsm.attrs.pop("LABL_PTR_1")
        encoding["R_gsm"] = {"compressor": compressor}
        ds = ds.assign(R_gsm=R_gsm)

    if "V_gsm" not in ds:
        V_gsm = quaternion_rotate(ds.V_eci, ds.Q_eci_to_gsm)
        V_gsm.name = "V_gsm"
        V_gsm.attrs.update(
            CATDESC="GSM velocity vector",
            standard_name="V GSM",
        )
        V_gsm.attrs.pop("FIELDNAM")
        V_gsm.attrs.pop("LABL_PTR_1")
        encoding["V_gsm"] = {"compressor": compressor}
        ds = ds.assign(V_gsm=V_gsm)

    ds.to_zarr(
        mode="a",
        store=store,
        group=group,
        encoding=encoding,
        consolidated=False,
    )
    print(f"Processed {group}")


if __name__ == "__main__":
    f = zarr.open(store)

    with Pool() as pool:
        for probe in ["mms1", "mms2", "mms3", "mms4"]:
            for _ in pool.uimap(
                lambda group: process(group[1].name),
                f[f"/srvy/mec/{probe}"].groups(),
            ):
                pass
