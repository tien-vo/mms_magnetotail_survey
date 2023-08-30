import zarr
import numpy as np
import xarray as xr
import astropy.units as u
import astropy.constants as c

from tvolib import mpl_utils as mu

from mms_survey.utils.io import store, compressor
from mms_survey.utils.cotrans import quaternion_rotate

ds_fgm = xr.open_zarr(
    store,
    group="/srvy/fgm/mms1/20170726",
    consolidated=False,
)
ds_mec = xr.open_zarr(
    store,
    group="/srvy/mec/mms1/20170726",
    consolidated=False,
)

Q_eci_to_gse = ds_mec.Q_eci_to_gse.interp(
    time=ds_fgm.time,
    kwargs=dict(fill_value=np.nan),
)
Q_eci_to_gsm = ds_mec.Q_eci_to_gsm.interp(
    time=ds_fgm.time,
    kwargs=dict(fill_value=np.nan),
)
B_eci = quaternion_rotate(ds_fgm.B_gse, Q_eci_to_gse, inverse=True)
B_gsm = quaternion_rotate(B_eci, Q_eci_to_gsm)
dB = np.linalg.norm(
    B_gsm - ds_fgm.B_gsm.sel(space=["x", "y", "z"]),
    axis=1,
)

fig, axes = mu.plt.subplots(5, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
ds_fgm.B_gsm.sel(space="x").plot.line(ax=ax, c="k", ls="-")
B_gsm.sel(space="x").plot.line(ax=ax, c="r", ls="--")

ax = axes[1]
ds_fgm.B_gsm.sel(space="y").plot.line(ax=ax, c="k", ls="-")
B_gsm.sel(space="y").plot.line(ax=ax, c="r", ls="--")

ax = axes[2]
ds_fgm.B_gsm.sel(space="z").plot.line(ax=ax, c="k", ls="-")
B_gsm.sel(space="z").plot.line(ax=ax, c="r", ls="--")

ax = axes[3]
ax.plot(B_gsm.time, dB, "-k")

ax = axes[4]
(ds_fgm.R_gsm.sel(space="x") / c.R_earth.to(u.km).value).plot.line(ax=ax, c="b")
(ds_fgm.R_gsm.sel(space="y") / c.R_earth.to(u.km).value).plot.line(ax=ax, c="g")
(ds_fgm.R_gsm.sel(space="z") / c.R_earth.to(u.km).value).plot.line(ax=ax, c="r")

for (i, ax) in enumerate(axes):
    mu.format_datetime_axis(ax)

fig.savefig("test.png")
