import numpy as np
import xarray as xr
from tvolib import mpl_utils as mu

from mms_survey.calc.integrator import integrate, apply_background_filter
from mms_survey.utils.units import to_np
from mms_survey.utils.io import data_dir


def get(species):
    kw = dict(
        engine="zarr",
        combine="nested",
        consolidated=False,
        combine_attrs="drop_conflicts",
    )
    ds = xr.open_mfdataset(
        str(
            data_dir /
            "20170726_iaw_event" /
            probe /
            f"fpi_{species}_distribution" /
            data_rate /
            data_level /
            "*"
        ),
        **kw,
    )
    ds_moms = xr.open_mfdataset(
        str(
            data_dir /
            "20170726_iaw_event" /
            probe /
            f"fpi_{species}_moments" /
            data_rate /
            data_level /
            "*"
        ),
        **kw,
    )
    ds_pmoms = xr.open_mfdataset(
        str(
            data_dir /
            "20170726_iaw_event" /
            probe /
            f"fpi_{species}_partial_moments" /
            data_rate /
            data_level /
            "*"
        ),
        **kw,
    )

    f3d = apply_background_filter(species, ds.f3d, ds.W, ds.theta, ds.phi,
                                  V_sc=ds_pmoms.V_sc)
    t, N, Tg, Wg, f1d = integrate(species, f3d, ds.W, ds.theta, ds.phi,
                                  V_sc=ds_pmoms.V_sc)
    #t = ds.time.values
    #W = ds.energy.values
    #Tg, Wg = np.meshgrid(t, W, indexing="ij")


    #f_omni = integrate_omni(species, f3d, ds.W, ds.theta, ds.phi,
    #                        V_sc=ds_pmoms.V_sc)
    return t, N, Tg, Wg, f1d, to_np(ds_moms.f_omni)


probe = "mms2"
data_rate = "brst"
data_level = "l2"

t_ion, N_ion, Tg_ion, Wg_ion, f1d_ion, f1d_ion_ref = get("ion")
t_elc, N_elc, Tg_elc, Wg_elc, f1d_elc, f1d_elc_ref = get("elc")

fig, axes = mu.plt.subplots(4, 1, figsize=(12, 8), sharex=True)
kw = dict(cmap="nipy_spectral", norm=mu.mplc.LogNorm(1e4, 1e8))

cax = mu.add_colorbar(ax := axes[0])
im = ax.pcolormesh(Tg_ion, Wg_ion, f1d_ion.value, **kw)
fig.colorbar(im, cax=cax)

cax = mu.add_colorbar(ax := axes[1])
im = ax.pcolormesh(Tg_ion, Wg_ion, f1d_ion_ref.value, **kw)
fig.colorbar(im, cax=cax)

cax = mu.add_colorbar(ax := axes[2])
im = ax.pcolormesh(Tg_elc, Wg_elc, f1d_elc.value, **kw)
fig.colorbar(im, cax=cax)

cax = mu.add_colorbar(ax := axes[3])
im = ax.pcolormesh(Tg_elc, Wg_elc, f1d_elc_ref.value, **kw)
fig.colorbar(im, cax=cax)

for (i, ax) in enumerate(axes):
    mu.format_datetime_axis(ax)
    ax.set_ylim(0, 32)

fig, axes = mu.plt.subplots(3, 1, figsize=(12, 8), sharex=True)
kw = dict(cmap="nipy_spectral", norm=mu.mplc.LogNorm(1e4, 1e8))

cax = mu.add_colorbar(ax := axes[0])
im = ax.pcolormesh(Tg_ion, Wg_ion, f1d_ion.value, **kw)
fig.colorbar(im, cax=cax)
ax.set_ylim(0, 32)

cax = mu.add_colorbar(ax := axes[1])
im = ax.pcolormesh(Tg_elc, Wg_elc, f1d_elc.value, **kw)
fig.colorbar(im, cax=cax)
ax.set_ylim(0, 32)

mu.add_colorbar(ax := axes[2]).remove()
ax.plot(t_ion, N_ion, "-r")
ax.plot(t_elc, N_elc, "-b")

for (i, ax) in enumerate(axes):
    mu.format_datetime_axis(ax)
    ax.set_xlim(
        np.datetime64("2017-07-26T07:28:41"),
        np.datetime64("2017-07-26T07:28:49"),
    )

mu.plt.show()
