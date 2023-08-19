import numpy as np
import tvolib as tv
import numpy.ma as ma
import astropy.units as u
from tvolib import mpl_utils as mu

from scipy.interpolate import UnivariateSpline

import lib
from lib.utils import read_data


where = "/analysis/dipole_tilt"
Y = read_data(f"{where}/Y").value
tilt = read_data(f"{where}/tilt_Y").value
H_Y_tilt = read_data(f"{where}/H_Y_tilt")
y_tilt = Y[:, 0]
# Calculate avg & std
tilt_avg = np.sum(tilt * H_Y_tilt, axis=1) / np.sum(H_Y_tilt, axis=1)
tilt_avg_tiled = np.tile(tilt_avg, (tilt.shape[1], 1)).T
tilt_std = np.sqrt(np.sum((tilt - tilt_avg_tiled) ** 2 * H_Y_tilt, axis=1) / np.sum(H_Y_tilt, axis=1))
# Fix nan
y_tilt = y_tilt[~np.isnan(tilt_avg)]
tilt_std = tilt_std[~np.isnan(tilt_avg)]
tilt_avg = tilt_avg[~np.isnan(tilt_avg)]
# Interpolate +-
f_tilt_avg = UnivariateSpline(y_tilt, tilt_avg, check_finite=True)
f_tilt_plus = UnivariateSpline(y_tilt, tilt_avg + tilt_std, check_finite=True)
f_tilt_minus = UnivariateSpline(y_tilt, tilt_avg - tilt_std, check_finite=True)


def ns_model(ymin=-20, ymax=20, Pobs=2, X=-20, chi=np.radians(20), tilt_model=f_tilt_avg):
    # Auxilliary parameters
    ksp = (2 / Pobs) ** (1 / 6)
    kmf = 1.06 * np.arctan(np.sqrt((10 - X) / 15.9))
    H0 = 9.98 * ksp
    Y0 = 18.44 * ksp * kmf
    D = 15.10 * ksp

    Y = np.linspace(max(ymin, y_tilt.min()), min(ymax, y_tilt.max()), 1000)
    chi = np.radians(tilt_model(Y))
    Z = np.sin(chi) * ((H0 + D) * np.sqrt(1 - (Y / Y0) ** 2) - D)
    return Y, Z


where = "/analysis/XYZ_distribution"
H = read_data(f"{where}/H")
H[H < 100] = np.nan
H_Bx = read_data(f"{where}/H_Bx")
H_tilt = read_data(f"{where}/H_tilt")
H_Bxy = read_data(f"{where}/H_Bxy")
H_beta = read_data(f"{where}/H_beta")
Xg = read_data(f"{where}/Xg")
Yg = read_data(f"{where}/Yg")
Zg = read_data(f"{where}/Zg")

beta = H_beta / H
Bxy = H_Bxy / H
Bx = H_Bx / H
tilt = H_tilt / H

beta_YZ = np.nanmean(beta, axis=0)
Bxy_YZ = np.nanmean(Bxy, axis=0)
mask = (beta_YZ >= 0.2) & (Bxy_YZ <= 14 * u.nT)
Y_ma = ma.array(Yg[0, :, :].value, mask=~mask).ravel()
Z_ma = ma.array(Zg[0, :, :].value, mask=~mask).ravel()

Y_ns, Z_ns = ns_model(Pobs=2, tilt_model=f_tilt_avg)
_, Z_ns_plus = ns_model(Pobs=2, tilt_model=f_tilt_plus)
_, Z_ns_minus = ns_model(Pobs=2, tilt_model=f_tilt_minus)

fig, axes = mu.plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)
tkw = dict(color="k", fontsize="medium", bbox=dict(facecolor="wheat", alpha=0.9))
skw = dict(s=4, fc="k", ec="none")

cax = mu.add_colorbar(ax := axes[0])
im = ax.pcolormesh(
    Yg[0, :, :].value, Zg[0, :, :].value, np.nanmean(Bx.value, axis=0),
    vmin=-30, vmax=30, cmap="seismic")
cb = fig.colorbar(im, cax)
cb.set_label("nT")
ax.scatter(Y_ma, Z_ma, **skw)
ax.plot(Y_ns, Z_ns, c="lime", lw=2, zorder=999)
ax.text(0.05, 0.1, "(a) $B_x$", transform=ax.transAxes, **tkw)

cax = mu.add_colorbar(ax := axes[1])
im = ax.pcolormesh(
    Yg[0, :, :].value, Zg[0, :, :].value, np.nanmean(tilt.value, axis=0),
    vmin=-30, vmax=30, cmap="seismic")
cb = fig.colorbar(im, cax)
cb.set_label("deg")
ax.text(0.05, 0.1, "(b) Dipole tilt", transform=ax.transAxes, **tkw)

for (i, ax) in enumerate(axes):
    x = np.linspace(-30, 0, 1000)
    ax.locator_params(axis="both", nbins=5)
    ax.set_aspect("equal")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 10)
    ax.set_facecolor("darkgray")
    ax.set_ylabel("$Z$ ($R_E$)")
    if i == 1:
        ax.set_xlabel("$Y$ ($R_E$)")

fig.tight_layout(h_pad=0.01)
fig.savefig(lib.plot_dir / "Fig7_YZ_distribution.png", dpi=600, transparent=True)
