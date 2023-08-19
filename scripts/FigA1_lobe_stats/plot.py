import numpy as np
import astropy.units as u
from tvolib import mpl_utils as mu

from background import f_omni_background

import lib
from lib.utils import read_data

where = "/analysis/lobe_stats"
W = read_data(f"{where}/Wg")
f = read_data(f"{where}/fg")
H_ion = read_data(f"{where}/H_ion")
H_elc = read_data(f"{where}/H_elc")

W_arr = np.logspace(np.log10(W[0, 0].value), np.log10(W[-1, 0].value), 1000) * W.unit
f_ion_bg = f_omni_background(W_arr, species="ion")
f_elc_bg = f_omni_background(W_arr, species="elc")

fig, axes = mu.plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
kw = dict(cmap="jet", norm=mu.mplc.LogNorm(1e0, 1e4))

cax = mu.add_colorbar(axes[1])
im = axes[0].pcolormesh(W.value, f.value, H_ion, **kw)
im = axes[1].pcolormesh(W.value, f.value, H_elc, **kw)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Counts")

axes[0].plot(W_arr, f_ion_bg, "-k")
axes[1].plot(W_arr, f_elc_bg, "-k")

icutoffs = np.array([28.3, 76.8]) * u.keV
ecutoffs = np.array([27.5, 65.9]) * u.keV
axes[0].axvline(icutoffs[0].value, c="magenta", ls="--", lw=2)
axes[0].axvline(icutoffs[1].value, c="magenta", ls="--", lw=2)
axes[1].axvline(ecutoffs[0].value, c="magenta", ls="--", lw=2)
axes[1].axvline(ecutoffs[1].value, c="magenta", ls="--", lw=2)

axes[0].set_ylabel(f"Energy flux ({f.unit:latex_inline})")
tkw = dict(color="k", fontsize="large", bbox=dict(facecolor="wheat", alpha=0.9))
(ax := axes[0]).text(0.05, 0.05, "(a) Ion", transform=ax.transAxes, **tkw)
(ax := axes[1]).text(0.05, 0.05, "(b) Electron", transform=ax.transAxes, **tkw)
for (i, ax) in enumerate(axes):
    ax.set_xlabel(f"Energy ({W.unit:latex_inline})")
    #ax.set_xlim(W.value.min(), W.value.max())
    ax.set_xlim(5e-2, W.value.max())
    #ax.set_ylim(f.value.min(), f.value.max())
    ax.set_ylim(1e1, 1e6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_facecolor("darkgray")

fig.tight_layout(w_pad=0.2)
fig.savefig(lib.plot_dir / "FigA1_lobe_stats.png", dpi=600)
