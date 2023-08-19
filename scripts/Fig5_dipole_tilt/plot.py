import numpy as np
import tvolib as tv
from tvolib import mpl_utils as mu

import lib
from lib.utils import read_data

where = "/analysis/dipole_tilt"
doy = read_data(f"{where}/doy")
Y = read_data(f"{where}/Y")
tilt_doy = read_data(f"{where}/tilt_doy")
tilt_Y = read_data(f"{where}/tilt_Y")
H_doy_tilt = read_data(f"{where}/H_doy_tilt")
H_Y_tilt = read_data(f"{where}/H_Y_tilt")

P_doy_tilt = H_doy_tilt / H_doy_tilt.sum()
P_Y_tilt = H_Y_tilt / H_Y_tilt.sum()
tilt_avg = np.sum(tilt_Y * H_Y_tilt, axis=1) / np.sum(H_Y_tilt, axis=1)
tilt_avg_tiled = np.tile(tilt_avg, (tilt_Y.shape[1], 1)).T
tilt_std = np.sqrt(np.sum((tilt_Y - tilt_avg_tiled) ** 2 * H_Y_tilt, axis=1) / np.sum(H_Y_tilt, axis=1))

dates = np.array([f"2017-{m}-01" for m in ["05", "06", "07", "08", "09", "10"]], dtype="datetime64[ns]")
dates = (dates.astype("datetime64[D]") - dates.astype("datetime64[Y]")).astype("i8")
texts = np.array(["May", "Jun", "Jul", "Aug", "Sep", "Oct"])

fig, axes = mu.plt.subplots(1, 2, figsize=(12, 6), sharey=True)
tkw = dict(color="k", fontsize="x-large", bbox=dict(facecolor="wheat", alpha=0.9))
kw = dict(cmap="jet")

im = axes[0].pcolormesh(doy, tilt_doy.value, P_doy_tilt * 1e2, vmax=0.5, **kw)
axes[0].set_ylim(tilt_doy.value.min(), tilt_doy.value.max())
axes[0].set_xlim(doy.min(), doy.max())
axes[0].set_ylabel("Dipole tilt ($^\\circ$)", fontsize="x-large")
axes[0].axhline(0, c="w", ls="--")
axes[0].set_xticks(dates)
axes[0].set_xticklabels(texts)
axes[0].text(0.05, 0.9, "(a)", transform=axes[0].transAxes, **tkw)

cax = mu.add_colorbar(axes[1])
im = axes[1].pcolormesh(Y.value, tilt_Y.value, P_Y_tilt * 1e2, vmax=0.5, **kw)
cb = fig.colorbar(im, cax=cax)
cb.set_label("PDF ($\\times10^2$)")
axes[1].set_ylim(tilt_Y.value.min(), tilt_Y.value.max())
axes[1].set_xlim(Y.value.min(), Y.value.max())
axes[1].axhline(0, c="w", ls="--")
axes[1].set_xlabel("$Y$ ($R_E$)", fontsize="x-large")
axes[1].plot(Y[:, 0].value, tv.numeric.move_avg(tilt_avg.value, (5,)), "-w")
axes[1].fill_between(
    Y[:, 0].value,
    tv.numeric.move_avg((tilt_avg + tilt_std).value, (5,)),
    tv.numeric.move_avg((tilt_avg - tilt_std).value, (5,)), color="w", alpha=0.3)
axes[1].text(0.05, 0.9, "(b)", transform=axes[1].transAxes, **tkw)

for (i, ax) in enumerate(axes):
    ax.tick_params(which="major", color="w", labelcolor="k", width=2, length=4)
    ax.tick_params(which="minor", color="w", labelcolor="k", width=2, length=2)

fig.tight_layout(h_pad=0.01)
fig.savefig(lib.plot_dir / "Fig5_dipole_tilt.png", dpi=600, transparent=True)
