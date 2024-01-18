import numpy as np
import xarray as xr
import astropy.units as u
from tvolib import mpl_utils as mu

from mms_survey.utils.xarray import to_np
from mms_survey.utils.io import data_dir
from mms_survey.calc.integrator import integrate_omni
from mms_survey.calc.distribution import apply_background_filter


def where(species, dtype, probe="mms2", data_rate="brst", data_level="l2"):
    return str(
        data_dir /
        "20170726_iaw_event" /
        probe /
        f"fpi_{species}_{dtype}" /
        data_rate /
        data_level /
        "*"
    )


def calc(species):
    kw = dict(
        engine="zarr",
        combine="nested",
        consolidated=False,
        combine_attrs="drop_conflicts",
    )
    ds_dist = xr.open_mfdataset(where(species, "distribution"), **kw)
    ds_moms = xr.open_mfdataset(where(species, "moments"), **kw)
    ds_pmoms = xr.open_mfdataset(where(species, "partial_moments"), **kw)

    #t = ds_dist.time.values
    #W = ds_dist.energy.values
    #Tg, Wg = np.meshgrid(t, W, indexing="ij")

    #f3d = apply_background_filter(
    #    species,
    #    ds_dist.f3d,
    #    ds_dist.W,
    #    ds_dist.theta,
    #    ds_dist.phi,
    #    V_sc=ds_pmoms.V_sc,
    #    cutoff_energy=60 * u.eV if species == "elc" else 0,
    #)
    #f1d = integrate_omni(
    #    species,
    #    f3d,
    #    ds_dist.W,
    #    ds_dist.theta,
    #    ds_dist.phi,
    #    V_sc=ds_pmoms.V_sc,
    #)
    #return Tg, Wg, to_np(f1d), to_np(ds_moms.f_omni)


#Tg_ion, Wg_ion, f1d_ion, f1d_ion_ref = calc("ion")
#Tg_elc, Wg_elc, f1d_elc, f1d_elc_ref = calc("elc")
#
#fig, axes = mu.plt.subplots(4, 1, figsize=(12, 8), sharex=True)
#kw = dict(cmap="nipy_spectral", norm=mu.mplc.LogNorm(1e4, 1e8))
#
#cax = mu.add_colorbar(ax := axes[0])
#im = ax.pcolormesh(Tg_ion, Wg_ion, f1d_ion.value, **kw)
#fig.colorbar(im, cax=cax)
#
#cax = mu.add_colorbar(ax := axes[1])
#im = ax.pcolormesh(Tg_ion, Wg_ion, f1d_ion_ref.value, **kw)
#fig.colorbar(im, cax=cax)
#
#cax = mu.add_colorbar(ax := axes[2])
#im = ax.pcolormesh(Tg_elc, Wg_elc, f1d_elc.value, **kw)
#fig.colorbar(im, cax=cax)
#
#cax = mu.add_colorbar(ax := axes[3])
#im = ax.pcolormesh(Tg_elc, Wg_elc, f1d_elc_ref.value, **kw)
#fig.colorbar(im, cax=cax)
#
#for (i, ax) in enumerate(axes):
#    mu.format_datetime_axis(ax)
#    ax.set_ylim(0, 32)
#
#mu.plt.show()
