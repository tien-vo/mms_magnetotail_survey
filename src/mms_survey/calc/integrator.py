import logging

import astropy.constants as c
import astropy.units as u
import numpy as np
import xarray as xr

from mms_survey.utils.xarray import to_da, to_np

q = dict(ion=c.si.e, elc=-c.si.e)
m = dict(ion=c.si.m_p, elc=c.si.m_e)


def precondition_distribution_function(ds: xr.Dataset, E0=100):
    new_ds = ds.copy()

    # Energy mapping
    new_ds = new_ds.assign(U=lambda x: x.W / (x.W + E0))

    # Angular boundary conditions
    new_ds = new_ds.reindex(
        azimuthal_sector=np.arange(33),
        zenith_sector=np.arange(-1, 17),
        fill_value=0.0,
    )
    phi_pad = dict(azimuthal_sector=32)
    new_ds.phi.loc[phi_pad] = new_ds.phi.sel(**phi_pad) + 360
    new_ds.f3d.loc[phi_pad] = new_ds.f3d.sel(**phi_pad)
    new_ds.theta.loc[dict(zenith_sector)] = 180.0

    # Transpose to desirable shape
    new_ds = new_ds.transpose("time", "energy_channel", "zenith_sector",
                              "azimuthal_sector", missing_dims="ignore")

    return new_ds.compute()


def integrate_energy_spectrogram():
    pass


#def precondition_distribution_function(
#    species: str,
#    f3d: xr.DataArray,
#    W: xr.DataArray,
#    theta: xr.DataArray,
#    phi: xr.DataArray,
#    V_sc: xr.DataArray = None,
#):
#    f3d, W, theta, phi = xr.broadcast(f3d, W, theta, phi)
#
#    dims = ("time", "energy", "zenith", "azimuth")
#    f3d = to_np(f3d.transpose(*dims))
#    theta = to_np(theta.transpose(*dims)).to(u.rad)
#    phi = to_np(phi.transpose(*dims)).to(u.rad)
#    if V_sc is not None:
#        W, V_sc = xr.broadcast(W, V_sc)
#        W = to_np(W.transpose(*dims)) + q[species] * to_np(
#            V_sc.transpose(*dims)
#        )
#        W[W < 0] = 0.0
#    else:
#        W = to_np(W.transpose(*dims))
#
#    f3d[np.isnan(f3d)] = 0.0
#    return f3d, W, theta, phi


#def integrate_energy_spectrogram(
#    species: str,
#    f3d: xr.DataArray,
#    W: xr.DataArray,
#    theta: xr.DataArray,
#    phi: xr.DataArray,
#    V_sc: xr.DataArray = None,
#):
#    f3d, W, theta, phi = precondition_distribution_function(
#        species, f3d, W, theta, phi, V_sc=V_sc
#    )
#
#    # Calculate auxillary variables
#    m = c.si.m_p if species == "ion" else c.si.m_e
#    V = np.sqrt(2 * W / m).to(u.Unit("1000 km/s"))
#
#    # Total solid angle
#    Omega = np.trapz(
#        np.trapz(
#            np.sin(theta),
#            x=phi,
#            axis=-1,
#        ),
#        x=theta[:, :, :, 0],
#        axis=-1,
#    ).to(u.sr)
#
#    # Omni-disitribution
#    f1d = (1 / Omega) * np.trapz(
#        np.trapz(
#            0.5 * V**4 * f3d * np.sin(theta) / u.sr,
#            x=phi,
#            axis=-1,
#        ),
#        x=theta[:, :, :, 0],
#        axis=-1,
#    )
#
#    return to_da(f1d.to(u.Unit("cm-2 s-1 sr-1")))
#
#
#def integrate(
#    species: str,
#    f3d: xr.DataArray,
#    W: xr.DataArray,
#    theta: xr.DataArray,
#    phi: xr.DataArray,
#    V_sc: xr.DataArray = None,
#):
#    ds = xr.Dataset(
#        coords=dict(
#            time=f3d.time.values,
#            energy=f3d.energy.values,
#        )
#    )
#
#    f1d = integrate_omni(species, f3d, W, theta, phi, V_sc=V_sc)
#
#    f3d, W, theta, phi = precondition_distribution_function(
#        species, f3d, W, theta, phi, V_sc=V_sc
#    )
#    m = c.si.m_p if species == "ion" else c.si.m_e
#    V = np.sqrt(2 * W / m).to(u.Unit("1000 km/s"))
#    f3d = f3d.to(u.Unit("1e-9 cm-3 km-3 s3"))
#
#    # Number density
#    N = np.trapz(
#        np.trapz(
#            np.trapz(
#                f3d * V**2 * np.sin(theta) / u.sr,
#                x=phi,
#                axis=-1,
#            ),
#            x=theta[:, :, :, 0],
#            axis=-1,
#        ),
#        x=V[:, :, 0, 0],
#        axis=-1,
#    ).to(u.Unit("cm-3"))
#
#    # Velocity
#    ux = (
#        (1 / N)
#        * np.trapz(
#            np.trapz(
#                np.trapz(
#                    -f3d * V**3 * np.sin(theta) ** 2 * np.cos(phi) / u.sr,
#                    x=phi,
#                    axis=-1,
#                ),
#                x=theta[:, :, :, 0],
#                axis=-1,
#            ),
#            x=V[:, :, 0, 0],
#            axis=-1,
#        )
#    ).to(u.Unit("km/s"))
#    uy = (
#        (1 / N)
#        * np.trapz(
#            np.trapz(
#                np.trapz(
#                    -f3d * V**3 * np.sin(theta) ** 2 * np.sin(phi) / u.sr,
#                    x=phi,
#                    axis=-1,
#                ),
#                x=theta[:, :, :, 0],
#                axis=-1,
#            ),
#            x=V[:, :, 0, 0],
#            axis=-1,
#        )
#    ).to(u.Unit("km/s"))
#    uz = (
#        (1 / N)
#        * np.trapz(
#            np.trapz(
#                np.trapz(
#                    -f3d * V**3 * np.sin(theta) * np.cos(theta) / u.sr,
#                    x=phi,
#                    axis=-1,
#                ),
#                x=theta[:, :, :, 0],
#                axis=-1,
#            ),
#            x=V[:, :, 0, 0],
#            axis=-1,
#        )
#    ).to(u.Unit("km/s"))
#
#    # Stress tensor
#    Pxx = np.trapz(
#        np.trapz(
#            np.trapz(
#                m
#                * V**4
#                * f3d
#                * np.sin(theta) ** 3
#                * np.cos(phi) ** 2
#                / u.sr,
#                x=phi,
#                axis=-1,
#            ),
#            x=theta[:, :, :, 0],
#            axis=-1,
#        ),
#        x=V[:, :, 0, 0],
#        axis=-1,
#    ).to(u.nPa)
#    Pyy = np.trapz(
#        np.trapz(
#            np.trapz(
#                m
#                * V**4
#                * f3d
#                * np.sin(theta) ** 3
#                * np.sin(phi) ** 2
#                / u.sr,
#                x=phi,
#                axis=-1,
#            ),
#            x=theta[:, :, :, 0],
#            axis=-1,
#        ),
#        x=V[:, :, 0, 0],
#        axis=-1,
#    ).to(u.nPa)
#    Pzz = np.trapz(
#        np.trapz(
#            np.trapz(
#                m * V**4 * f3d * np.sin(theta) * np.cos(theta) ** 2 / u.sr,
#                x=phi,
#                axis=-1,
#            ),
#            x=theta[:, :, :, 0],
#            axis=-1,
#        ),
#        x=V[:, :, 0, 0],
#        axis=-1,
#    ).to(u.nPa)
#    Pxy = np.trapz(
#        np.trapz(
#            np.trapz(
#                m
#                * V**4
#                * f3d
#                * np.sin(theta) ** 3
#                * np.sin(phi)
#                * np.cos(phi)
#                / u.sr,
#                x=phi,
#                axis=-1,
#            ),
#            x=theta[:, :, :, 0],
#            axis=-1,
#        ),
#        x=V[:, :, 0, 0],
#        axis=-1,
#    ).to(u.nPa)
#    Pxz = np.trapz(
#        np.trapz(
#            np.trapz(
#                m
#                * V**4
#                * f3d
#                * np.sin(theta)
#                * np.cos(theta)
#                * np.cos(phi)
#                / u.sr,
#                x=phi,
#                axis=-1,
#            ),
#            x=theta[:, :, :, 0],
#            axis=-1,
#        ),
#        x=V[:, :, 0, 0],
#        axis=-1,
#    ).to(u.nPa)
#    Pyz = np.trapz(
#        np.trapz(
#            np.trapz(
#                m
#                * V**4
#                * f3d
#                * np.sin(theta)
#                * np.cos(theta)
#                * np.sin(phi)
#                / u.sr,
#                x=phi,
#                axis=-1,
#            ),
#            x=theta[:, :, :, 0],
#            axis=-1,
#        ),
#        x=V[:, :, 0, 0],
#        axis=-1,
#    ).to(u.nPa)
#
#    ds = ds.assign(
#        N=xr.DataArray(
#            N.value,
#            dims=("time"),
#            attrs=dict(units=str(N.unit)),
#        ),
#        V=xr.DataArray(
#            np.array([ux, uy, uz]).T,
#            dims=("time", "space"),
#            coords=dict(space=["x", "y", "z"]),
#            attrs=dict(units=str(ux.unit)),
#        ),
#        Pxx=xr.DataArray(
#            Pxx.value,
#            dims=("time"),
#            attrs=dict(units=str(Pxx.unit)),
#        ),
#        Pyy=xr.DataArray(
#            Pyy.value,
#            dims=("time"),
#            attrs=dict(units=str(Pxx.unit)),
#        ),
#        Pzz=xr.DataArray(
#            Pzz.value,
#            dims=("time"),
#            attrs=dict(units=str(Pxx.unit)),
#        ),
#        Pxy=xr.DataArray(
#            Pxy.value,
#            dims=("time"),
#            attrs=dict(units=str(Pxx.unit)),
#        ),
#        Pxz=xr.DataArray(
#            Pxz.value,
#            dims=("time"),
#            attrs=dict(units=str(Pxx.unit)),
#        ),
#        Pyz=xr.DataArray(
#            Pyz.value,
#            dims=("time"),
#            attrs=dict(units=str(Pxx.unit)),
#        ),
#    )
#    return ds
