import numpy as np
import tvolib as tv
import numpy.ma as ma
import astropy.units as u
from tvolib import mpl_utils as mu

import lib
from lib.utils import read_data

where = "/analysis/XYZ_distribution"
H = read_data(f"{where}/H")
H[H < 100] = np.nan
H_beta = read_data(f"{where}/H_beta")
H_Bxy = read_data(f"{where}/H_Bxy")
H_Ez = read_data(f"{where}/H_Ez")
Xg = read_data(f"{where}/Xg")
Zg = read_data(f"{where}/Zg")

beta = H_beta / H
Bxy = H_Bxy / H
Ez = H_Ez / H
beta_XZ = np.nanmean(beta, axis=1)
Bxy_XZ = np.nanmean(Bxy, axis=1)

b_ma = (beta_XZ >= 0.2)
X_bma = ma.array(Xg[:, 0, :].value, mask=~b_ma).ravel()
Z_bma = ma.array(Zg[:, 0, :].value, mask=~b_ma).ravel()

B_ma = (Bxy_XZ <= 14 * u.nT)
X_Bma = ma.array(Xg[:, 0, :].value, mask=~B_ma).ravel()
Z_Bma = ma.array(Zg[:, 0, :].value, mask=~B_ma).ravel()

s_ma = b_ma & B_ma
X_sma = ma.array(Xg[:, 0, :].value, mask=~s_ma).ravel()
Z_sma = ma.array(Zg[:, 0, :].value, mask=~s_ma).ravel()

fig, axes = mu.plt.subplots(1, 3, figsize=(16, 6.5), sharex=True, sharey=True)
tkw = dict(color="k", fontsize="x-large", bbox=dict(facecolor="wheat", alpha=0.9))
ckw = dict(location="top", pad=0.1, fraction=0.025)
skw = dict(s=4, fc="k", ec="k")

axes[0].set_title("$\\beta_i$", pad=10, fontsize="x-large")
im = axes[0].pcolormesh(
    Xg[:, 0, :].value, Zg[:, 0, :].value, np.nanmean(beta, axis=1),
    norm=mu.mplc.LogNorm(1e-3, 1e3), cmap="hot")
axes[0].scatter(X_bma, Z_bma, label=f"$\\beta_i\\geq{0.2}$", **skw)
fig.colorbar(im, ax=axes[0], ticks=np.logspace(-3, 3, 5), **ckw)

axes[1].set_title("$B_{xy}$ (nT)", pad=10, fontsize="x-large")
im = axes[1].pcolormesh(
    Xg[:, 0, :].value, Zg[:, 0, :].value, np.nanmean(Bxy.value, axis=1),
    vmin=0, vmax=35, cmap="hot")
axes[1].scatter(X_Bma, Z_Bma, label=f"$B_{{xy}}\\leq 14$ nT", **skw)
fig.colorbar(im, ax=axes[1], ticks=np.arange(5, 36, 10), **ckw)

axes[2].set_title("$E_z$ (mV/m)", pad=10, fontsize="x-large")
im = axes[2].pcolormesh(
    Xg[:, 0, :].value, Zg[:, 0, :].value, np.nanmean(Ez.value, axis=1),
    vmin=-2, vmax=2, cmap="seismic")
axes[2].scatter(X_sma, Z_sma, label=f"$\\beta_i\\geq 0.2$ & $B_{{xy}}\\leq 14$ nT", **skw)
fig.colorbar(im, ax=axes[2], ticks=np.linspace(-2, 2, 5), **ckw)

x = np.linspace(-30, 0, 1000)
texts = ["(a)", "(b)", "(c)"]
for (i, ax) in enumerate(axes):
    ax.legend(frameon=True, loc="lower left", fontsize="large", handlelength=1.5,
              handletextpad=0.5, borderaxespad=0.2, handleheight=0.5)
    ax.text(0.05, 0.9, texts[i], transform=ax.transAxes, **tkw)
    ax.locator_params(axis="both", nbins=5)
    ax.set_aspect("equal")
    ax.set_xlim(-8, -30)
    ax.set_ylim(-12, 10)
    ax.set_facecolor("darkgray")
    ax.set_xlabel("$X$ ($R_E$)", fontsize="x-large")
    if i == 0:
        ax.set_ylabel("$Z$ ($R_E$)", fontsize="x-large")

fig.tight_layout()
fig.savefig(lib.plot_dir / "Fig6_XZ_distribution.pdf", dpi=600)
fig.savefig(lib.plot_dir / "Fig6_XZ_distribution.png", dpi=600)
