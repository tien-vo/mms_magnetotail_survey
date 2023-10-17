"""
Check whether phi is varying in DBCS
"""

import zarr
import numpy as np
import xarray as xr
from tvolib import mpl_utils as mu
from dask.diagnostics import ProgressBar
from mms_survey.utils.io import data_dir

with ProgressBar():
    ds_ion = xr.open_mfdataset(
        str(
            data_dir / "20170726_iaw_event" / "mms2"
            / "fpi_ion_distribution" / "brst" / "l2" / "*"
        ),
        engine="zarr",
        combine="nested",
        parallel=True,
        consolidated=False,
    )
    ds_elc = xr.open_mfdataset(
        str(
            data_dir / "20170726_iaw_event" / "mms2"
            / "fpi_elc_distribution" / "brst" / "l2" / "*"
        ),
        engine="zarr",
        combine="nested",
        parallel=True,
        consolidated=False,
    )

eyes = np.arange(1, 33)

fig, axes = mu.plt.subplots(2, 1, figsize=(12, 8), sharex=True)

t = ds_ion.time.values
T, E = np.meshgrid(t, eyes, indexing="ij")
cax = mu.add_colorbar(ax := axes[0])
im = ax.pcolormesh(T, E, ds_ion.phi.values, vmin=0, vmax=360, cmap="jet")
cb = fig.colorbar(im, cax=cax)
cb.set_label("(deg)")

t = ds_elc.time.values
T, E = np.meshgrid(t, eyes, indexing="ij")
cax = mu.add_colorbar(ax := axes[1])
im = ax.pcolormesh(T, E, ds_elc.phi.values, vmin=0, vmax=360, cmap="jet")
cb = fig.colorbar(im, cax=cax)
cb.set_label("(deg)")

axes[0].set_ylabel("Ion phi label")
axes[1].set_ylabel("Elc phi label")
for (i, ax) in enumerate(axes):
    mu.format_datetime_axis(ax)
    ax.set_ylim(eyes[0], eyes[-1])

mu.plt.show()
