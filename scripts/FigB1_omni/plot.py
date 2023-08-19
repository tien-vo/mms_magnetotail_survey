import numpy as np
from tvolib import mpl_utils as mu

import lib
from lib.utils import read_data

where = "/analysis/omni_stats"
Pg = read_data(f"{where}/Pg")
Bg = read_data(f"{where}/Bg")
H_P_By = read_data(f"{where}/H_P_By")
H_P_Bz = read_data(f"{where}/H_P_Bz")
H_P_By /= np.nansum(H_P_By)
H_P_Bz /= np.nansum(H_P_Bz)

fig, axes = mu.plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
kw = dict(cmap="jet")

cax = mu.add_colorbar(axes[1])
im = axes[0].pcolormesh(Pg.value, Bg.value, H_P_By * 1e3, **kw)
im = axes[1].pcolormesh(Pg.value, Bg.value, H_P_Bz * 1e3, **kw)
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"PDF ($\times 10^{-3}$)")

axes[0].set_ylabel("nT", fontsize="x-large")
tkw = dict(color="k", fontsize="x-large", bbox=dict(facecolor="wheat", alpha=0.9))
axes[0].text(0.05, 0.9, "(a) IMF $B_y$", transform=axes[0].transAxes, **tkw)
axes[1].text(0.05, 0.9, "(b) IMF $B_z$", transform=axes[1].transAxes, **tkw)
for (i, ax) in enumerate(axes):
    ax.tick_params(which="major", color="w", labelcolor="k", width=2, length=4)
    ax.tick_params(which="minor", color="w", labelcolor="k", width=2, length=2)
    ax.set_xlim(Pg.value.min(), Pg.value.max())
    ax.set_ylim(Bg.value.min(), Bg.value.max())
    ax.set_xlabel("$P_{sw}$ (nPa)", fontsize="x-large")

fig.tight_layout(w_pad=0.2)
fig.savefig(lib.plot_dir / "FigB1_omni_stats.png", dpi=600)
