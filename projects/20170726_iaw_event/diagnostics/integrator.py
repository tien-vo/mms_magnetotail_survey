import numpy as np
import xarray as xr
import astropy.units as u
from tvolib import mpl_utils as mu

from mms_survey.utils.units import to_np
from mms_survey.utils.io import data_dir
from mms_survey.calc.integrator import integrate
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

    f3d = apply_background_filter(
        species,
        ds_dist.f3d,
        ds_dist.W,
        ds_dist.theta,
        ds_dist.phi,
        V_sc=ds_pmoms.V_sc,
        cutoff_energy=60 * u.eV if species == "elc" else 0,
    )
    ds = integrate(
        species,
        f3d,
        ds_dist.W,
        ds_dist.theta,
        ds_dist.phi,
        V_sc=ds_pmoms.V_sc,
    )
    return ds, ds_pmoms


ds_ion, ds_ion_ref = calc("ion")
ds_elc, ds_elc_ref = calc("elc")

fig, axes = mu.plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(ds_ion.time, ds_ion.N, "-k")
axes[0].plot(ds_ion_ref.time, ds_ion_ref.N.sel(energy=6), "-r")

axes[1].plot(ds_elc.time, ds_elc.N, "-k")
axes[1].plot(ds_elc_ref.time, ds_elc_ref.N.sel(energy=6), "-r")

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

for (i, ax) in enumerate(axes):
    mu.format_datetime_axis(ax)

mu.plt.show()
