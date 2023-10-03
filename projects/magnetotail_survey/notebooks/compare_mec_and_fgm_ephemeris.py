import numpy as np
import xarray as xr
import dask.array as da

from dask.diagnostics import ProgressBar
from tvolib import mpl_utils as mu
from mms_survey.utils.io import data_dir, plot_dir


def fgm_preprocess(ds):
    ds = ds.drop_dims("time")
    ds = ds.rename({"ephemeris_time": "time"})
    ds = ds.drop_duplicates("time")
    return ds


kw = dict(
    combine="nested",
    engine="zarr",
    data_vars="minimal",
    parallel=True,
    consolidated=False)
with ProgressBar():
    ds_mec = xr.open_mfdataset(
        str(data_dir / "srvy" / "mec" / "mms1" / "*"),
        concat_dim="time",
        **kw)
    ds_fgm = xr.open_mfdataset(
        str(data_dir / "srvy" / "fgm" / "mms1" / "*"),
        concat_dim="ephemeris_time",
        preprocess=preprocess,
        **kw)

R_mec = ds_mec.R_gsm.drop_duplicates("time").sel(space=["x", "y", "z"])
R_fgm = ds_fgm.R_gsm.rename({"ephemeris_time": "time"})
R_fgm = R_fgm.drop_duplicates("time")
R_fgm = R_fgm.interp(time=R_mec.time, kwargs={"fill_value": np.nan})
R_fgm = R_fgm.sel(space=["x", "y", "z"])
dR = da.linalg.norm(R_mec - R_fgm, axis=1)

bins = np.arange(0, 501, 1)
H = da.histogram(dR, bins=bins)[0]
H = H.compute()

fig, ax = mu.plt.subplots(1, 1, figsize=(8, 6))

ax.plot(bins[:-1], H, "-k")

ax.set_xlabel("$\\Delta R$ (km)")
ax.set_ylabel("Counts")
ax.set_yscale("log")

save_dir = plot_dir / "analysis" / "technical"
save_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(save_dir / "compare_mec_and_fgm_ephemeris.png")
mu.plt.close(fig)
