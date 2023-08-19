import numpy as np
from tvolib import mpl_utils as mu
from tvolib.models.magnetopause_model import Lin10MagnetopauseModel

import lib
from lib.utils import read_data

where = "/analysis/inner_tail"
X = read_data(f"{where}/Xg").mean(axis=2)
Y = read_data(f"{where}/Yg").mean(axis=2)
H = read_data(f"{where}/H")
H[H < 100] = np.nan
H_N_ion = read_data(f"{where}/H_N_ion")
H_T_elc = read_data(f"{where}/H_T_elc")
H_dB = read_data(f"{where}/H_dB")

counts = np.nanmean(H, axis=2)
N_ion = np.nanmean(H_N_ion / H, axis=2)
dB = np.nanmean(H_dB / H, axis=2)
T_elc = np.nanmean(H_T_elc / H, axis=2)

ckw = dict(location="top", pad=0.1, fraction=0.015)
tkw = dict(color="k", fontsize="large", bbox=dict(facecolor="wheat", alpha=0.9))
pkw = dict()

fig, axes = mu.plt.subplots(1, 4, figsize=(12, 6), sharex=True, sharey=True)

axes[0].set_title("Coverage (counts)", pad=10)
im = axes[0].pcolormesh(X.value, Y.value, counts, vmax=1500, cmap="hot", **pkw)
fig.colorbar(im, ax=axes[0], ticks=np.linspace(0, 1500, 3), extend="max", **ckw)

axes[1].set_title(f"$n_i$ ({N_ion.unit:latex_inline})", pad=10)
im = axes[1].pcolormesh(X.value, Y.value, N_ion.value, norm=mu.mplc.LogNorm(1e-2, 1e2), cmap="hot", **pkw)
fig.colorbar(im, ax=axes[1], ticks=np.logspace(-2, 2, 3), extend="max", **ckw)

axes[3].set_title(f"$\\sigma_B$ ({dB.unit:latex_inline})", pad=10)
im = axes[3].pcolormesh(X.value, Y.value, dB.value, vmin=0, vmax=1, cmap="hot", **pkw)
fig.colorbar(im, ax=axes[3], ticks=np.linspace(0, 1, 3), extend="max", **ckw)

axes[2].set_title(f"$T_{{e}}$ ({T_elc.unit:latex_inline})", pad=10)
im = axes[2].pcolormesh(X.value, Y.value, T_elc.value, vmin=0, vmax=1, cmap="hot", **pkw)
fig.colorbar(im, ax=axes[2], ticks=np.linspace(0, 1, 3), extend="max", **ckw)

x = X[X[:, 0] < 0, 0].value
texts = ["(a)", "(b)", "(c)", "(d)"]
model = Lin10MagnetopauseModel(pressure=20, bfield=0, tilt_angle=0)
for (i, ax) in enumerate(axes):
    mu.draw_earth(ax)
    ax.set_aspect("equal")
    ax.locator_params(axis="y", nbins=6)
    ax.set_xlim(X.value.max(), X.value.min())
    ax.set_ylim(Y.value.max(), Y.value.min())
    ax.set_xticks(np.arange(-25, 6, 10, dtype=int))
    ax.set_facecolor("grey")
    ax.set_xlabel("$X$ ($R_E$)")
    ax.text(0.05, 0.9, texts[i], transform=ax.transAxes, **tkw)
    if i == 0:
        ax.set_ylabel("$Y$ ($R_E$)")
    else:
        model.pressure = 20
        model.construct_model()
        model.add_mpause_model(ax, which="XY", rotation_angle=np.radians(-5), ls="-", lw=2, c="lime", zorder=9)
        model.pressure = 5
        model.construct_model()
        model.add_mpause_model(ax, which="XY", rotation_angle=np.radians(-5), ls="--", lw=2, c="lime", zorder=9)
        ax.axhline(-15, c="k", ls="--", lw=2)
        ax.axhline(15, c="k", ls="--", lw=2)
        ax.plot(x, x, "--k", lw=2)
        ax.plot(x, -x, "--k", lw=2)

fig.tight_layout()
fig.savefig(lib.plot_dir / "Fig2_inner_tail.pdf", dpi=600)
fig.savefig(lib.plot_dir / "Fig2_inner_tail.png", dpi=600)
