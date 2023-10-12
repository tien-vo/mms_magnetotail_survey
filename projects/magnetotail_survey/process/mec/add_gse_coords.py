import zarr
import xarray as xr

from pathos.pools import ProcessPool as Pool

from mms_survey.utils.io import store, compressor
from mms_survey.utils.cotrans import quaternion_rotate


def process(group: str):
    ds = xr.open_zarr(store, group=group, consolidated=False)
    encoding = dict()

    if "R_gse" not in ds:
        R_gse = quaternion_rotate(ds.R_eci, ds.Q_eci_to_gse)
        R_gse.name = "R_gse"
        R_gse.attrs.update(
            CATDESC="GSE position vector",
            standard_name="R GSE",
        )
        R_gse.attrs.pop("FIELDNAM")
        R_gse.attrs.pop("LABL_PTR_1")
        encoding["R_gse"] = {"compressor": compressor}
        ds = ds.assign(R_gse=R_gse)

    if "V_gse" not in ds:
        V_gse = quaternion_rotate(ds.V_eci, ds.Q_eci_to_gse)
        V_gse.name = "V_gse"
        V_gse.attrs.update(
            CATDESC="GSE velocity vector",
            standard_name="V GSE",
        )
        V_gse.attrs.pop("FIELDNAM")
        V_gse.attrs.pop("LABL_PTR_1")
        encoding["V_gse"] = {"compressor": compressor}
        ds = ds.assign(V_gse=V_gse)

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
