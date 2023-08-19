import numpy as np
import astropy.units as u
from tvolib import mpl_utils as mu

import lib
from lib.utils import read_data

# Event ID
interval = 418
trange = np.array(["2017-07-26T07:10", "2017-07-26T07:45"], dtype="datetime64[ns]")
t1 = np.datetime64("2017-07-26T07:18:30")
t2 = np.datetime64("2017-07-26T07:29:30")
t3 = np.datetime64("2017-07-26T07:40:00")
icutoffs = np.array([28.3, 76.8]) * u.keV
ecutoffs = np.array([27.5, 65.9]) * u.keV

# Create figure
mu.plt.rc("xtick", labelsize="small")
mu.plt.rc("ytick", labelsize="small")
mu.plt.rc("axes", labelsize="medium")
fig = mu.plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(8, 5, width_ratios=(1, 1, 0.025, 0.5, 1))
fig.subplots_adjust(hspace=0.1, wspace=0.08, bottom=0.06, top=0.98, right=0.98)

# Barycentric magnetic field
t = read_data(f"/postprocess/interval_{interval}/barycenter/t").astype("datetime64[ns]")
B_bc = read_data(f"/postprocess/interval_{interval}/barycenter/B_bc")
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.set_ylabel(f"{B_bc.unit:latex_inline}")
ax1.set_ylim(-30, 30)
ax1.set_yticks(np.arange(-20, 21, 10))
ax1.plot(t, B_bc[:, 0], "-b")
ax1.plot(t, B_bc[:, 1], "-g")
ax1.plot(t, B_bc[:, 2], "-r")
ax1.plot(t, np.linalg.norm(B_bc, axis=1), "-k")
kw = dict(x=1.02, transform=ax1.transAxes, fontsize="small")
ax1.text(y=0.8, s="x", c="b", **kw)
ax1.text(y=0.6, s="y", c="g", **kw)
ax1.text(y=0.4, s="z", c="r", **kw)
ax1.text(y=0.2, s="mag", c="k", **kw)

# Barycentric electric field
E_bc = read_data(f"/postprocess/interval_{interval}/barycenter/E_bc")
ax2 = fig.add_subplot(gs[1, 0:2])
ax2.set_ylabel(f"{E_bc.unit:latex_inline}")
ax2.set_ylim(-150, 150)
ax2.set_yticks(np.arange(-100, 101, 50))
ax2.plot(t, E_bc[:, 0], "-b")
ax2.plot(t, E_bc[:, 1], "-g")
ax2.plot(t, E_bc[:, 2], "-r")
kw = dict(x=1.02, transform=ax2.transAxes, fontsize="small")
ax2.text(y=0.8, s="x", c="b", **kw)
ax2.text(y=0.5, s="y", c="g", **kw)
ax2.text(y=0.2, s="z", c="r", **kw)

# MMS1 ion velocity
t = read_data(f"/mms1/ion-fpi-moms/interval_{interval}/t").astype("datetime64[ns]")
Vi = read_data(f"/mms1/ion-fpi-moms/interval_{interval}/V_gsm").to(u.Unit("1000 km/s"))
ax3 = fig.add_subplot(gs[2, 0:2])
ax3.set_ylabel(f"{Vi.unit:latex_inline}")
ax3.set_ylim(-1.5, 1.5)
ax3.set_yticks(np.arange(-1, 1.1, 0.5))
ax3.plot(t, Vi[:, 0], "-b")
ax3.plot(t, Vi[:, 1], "-g")
ax3.plot(t, Vi[:, 2], "-r")
kw = dict(x=1.02, transform=ax3.transAxes, fontsize="small")
ax3.text(y=0.8, s="x", c="b", **kw)
ax3.text(y=0.5, s="y", c="g", **kw)
ax3.text(y=0.2, s="z", c="r", **kw)

# MMS1 ion energy spectrum
t_ion = read_data(f"/postprocess/interval_{interval}/ion/t").astype("datetime64[ns]")
tg_ion = read_data(f"/postprocess/interval_{interval}/ion/tg").astype("datetime64[ns]")
Wg_ion = read_data(f"/postprocess/interval_{interval}/ion/f_omni_energy")
f_ion = read_data(f"/postprocess/interval_{interval}/ion/f_omni")
ax4 = fig.add_subplot(gs[3, 0:2])
cax4_r = fig.add_subplot(gs[3, 2])
ax4.set_ylabel(f"{Wg_ion.unit:latex_inline}")
ax4.set_yscale("log")
ax4.set_ylim(1e-1, 1e3)
ax4.set_yticks(np.power(10.0, np.arange(-1, 3, 1)))
im = ax4.pcolormesh(tg_ion, Wg_ion.value, f_ion.value, norm=mu.mplc.LogNorm(1e2, 1e8), cmap="jet")
cb = fig.colorbar(im, ax=ax4, cax=cax4_r, ticks=np.logspace(0, 6, 3))
cb.set_label(f"{f_ion.unit:latex_inline}", fontsize="x-small")
ax4.axhline(icutoffs[0].value, c="magenta", ls="--", lw=2)
ax4.axhline(icutoffs[1].value, c="magenta", ls="--", lw=2)
ax4.set_facecolor("silver")

# MMS1 elc energy spectrum
t_elc = read_data(f"/postprocess/interval_{interval}/elc/t").astype("datetime64[ns]")
tg_elc = read_data(f"/postprocess/interval_{interval}/elc/tg").astype("datetime64[ns]")
Wg_elc = read_data(f"/postprocess/interval_{interval}/elc/f_omni_energy")
f_elc = read_data(f"/postprocess/interval_{interval}/elc/f_omni")
ax5 = fig.add_subplot(gs[4, 0:2])
cax5_r = fig.add_subplot(gs[4, 2])
ax5.set_ylabel(f"{Wg_elc.unit:latex_inline}")
ax5.set_yscale("log")
ax5.set_ylim(1e-1, 1e3)
ax5.set_yticks(np.power(10.0, np.arange(-1, 3, 1)))
im = ax5.pcolormesh(tg_elc, Wg_elc.value, f_elc.value, norm=mu.mplc.LogNorm(1e2, 1e8), cmap="jet")
cb = fig.colorbar(im, ax=ax5, cax=cax5_r, ticks=np.logspace(0, 6, 3))
cb.set_label(f"{f_elc.unit:latex_inline}", fontsize="x-small")
ax5.axhline(ecutoffs[0].value, c="magenta", ls="--", lw=2)
ax5.axhline(ecutoffs[1].value, c="magenta", ls="--", lw=2)
ax5.set_facecolor("silver")

# MMS1 densities
N_ion = read_data(f"/postprocess/interval_{interval}/ion/N")
N_elc = read_data(f"/postprocess/interval_{interval}/elc/N")
ax6 = fig.add_subplot(gs[5, 0:2])
ax6.set_ylabel(f"{N_ion.unit:latex_inline}")
ax6.set_ylim(0, 0.35)
ax6.set_yticks(np.arange(0, 0.36, 0.1))
ax6.plot(t_ion, N_ion, "-r")
ax6.plot(t_elc, N_elc, "-b")
kw = dict(x=1.02, transform=ax6.transAxes, fontsize="small")
ax6.text(y=0.8, s="Ion", c="r", **kw)
ax6.text(y=0.2, s="Elc", c="b", **kw)

# MMS1 pressures
P_ion = read_data(f"/postprocess/interval_{interval}/ion/P_scalar")
P_elc = read_data(f"/postprocess/interval_{interval}/elc/P_scalar")
ax7 = fig.add_subplot(gs[6, 0:2])
ax7.set_ylabel(f"{P_ion.unit:latex_inline}")
ax7.set_ylim(0, 3)
ax7.set_yticks(np.arange(0, 3, 0.5))
ax7.plot(t_ion, P_ion, "-r")
ax7.plot(t_elc, P_elc, "-b")
kw = dict(x=1.02, transform=ax7.transAxes, fontsize="small")
ax7.text(y=0.8, s="Ion", c="r", **kw)
ax7.text(y=0.2, s="Elc", c="b", **kw)

# MMS1 nonthermal pressures
Pnt_ion = read_data(f"/postprocess/interval_{interval}/ion/P_scalar_nt")
Pnt_elc = read_data(f"/postprocess/interval_{interval}/elc/P_scalar_nt")
ax8 = fig.add_subplot(gs[7, 0:2])
ax8.set_ylabel(f"{Pnt_ion.unit:latex_inline}")
ax8.set_ylim(0, 0.8)
ax8.set_yticks(np.arange(0, 0.8, 0.2))
ax8.plot(t_ion, Pnt_ion, "-r")
ax8.plot(t_elc, Pnt_elc, "-b")
kw = dict(x=1.02, transform=ax8.transAxes, fontsize="small")
ax8.text(y=0.8, s="Ion", c="r", **kw)
ax8.text(y=0.2, s="Elc", c="b", **kw)

ts_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
ts_labels = [
    "(a) B GSM",
    "(b) E GSM",
    "(c) $V_i$ GSM",
    "(d) Ion eflux",
    "(e) Elc eflux",
    "(f) Density",
    "(g) Pressure",
    "(h) Non-thermal pressure",
]
fig.align_ylabels(ts_axes)
ii1 = np.argmin(np.abs(t_ion - t1))
ii2 = np.argmin(np.abs(t_ion - t2))
ii3 = np.argmin(np.abs(t_ion - t3))
for (i, ax) in enumerate(ts_axes):
    ax.axvline(t_ion[ii1], color="b", lw=2)
    ax.axvline(t_ion[ii2], color="g", lw=2)
    ax.axvline(t_ion[ii3], color="r", lw=2)
    ax.set_xlim(*trange)
    mu.format_datetime_axis(ax)
    ax.text(0.02, 0.75, ts_labels[i], transform=ax.transAxes, bbox=dict(facecolor="wheat", alpha=0.9))
    if i < len(ts_axes) - 1:
        ax.set_xticklabels([])

skw = dict(s=15)

# Ion & elc distribution at different times
Wg_ion_raw = read_data(f"/mms1/ion-fpi-moms/interval_{interval}/f_omni_energy")
f_ion_raw = read_data(f"/mms1/ion-fpi-moms/interval_{interval}/f_omni")
Vsc = read_data(f"mms1/ion-fpi-moms/interval_{interval}/Vsc")
f_ion_raw[(Wg_ion_raw < 60 * u.eV) | (Wg_ion_raw < np.abs(Vsc)[:, np.newaxis])] = np.nan
f_sorted = np.take_along_axis(f_ion_raw, np.argsort(f_ion_raw, axis=1), axis=1)
f_ion_raw = f_ion_raw - np.nanmean(f_sorted[:, :5], axis=1)[:, np.newaxis]
axi = fig.add_subplot(gs[0:4, 4])
axi.scatter(Wg_ion_raw[ii1, :].value, f_ion_raw[ii1, :].value, ec="b", fc="none", **skw)
axi.scatter(Wg_ion_raw[ii2, :].value, f_ion_raw[ii2, :].value, ec="g", fc="none", **skw)
axi.scatter(Wg_ion_raw[ii3, :].value, f_ion_raw[ii3, :].value, ec="r", fc="none", **skw)
axi.scatter(Wg_ion[ii1, :].value, f_ion[ii1, :].value, fc="b", ec="b", **skw)
axi.scatter(Wg_ion[ii2, :].value, f_ion[ii2, :].value, fc="g", ec="g", **skw)
axi.scatter(Wg_ion[ii3, :].value, f_ion[ii3, :].value, fc="r", ec="r", **skw)

Wg_elc_raw = read_data(f"/mms1/elc-fpi-moms/interval_{interval}/f_omni_energy")
f_elc_raw = read_data(f"/mms1/elc-fpi-moms/interval_{interval}/f_omni")
axe = fig.add_subplot(gs[4:, 4])
axe.scatter(Wg_elc_raw[ii1, :].value, f_elc_raw[ii1, :].value, ec="b", fc="none", **skw)
axe.scatter(Wg_elc_raw[ii2, :].value, f_elc_raw[ii2, :].value, ec="g", fc="none", **skw)
axe.scatter(Wg_elc_raw[ii3, :].value, f_elc_raw[ii3, :].value, ec="r", fc="none", **skw)
axe.scatter(Wg_elc[ii1, :].value, f_elc[ii1, :].value, fc="b", ec="b", **skw)
axe.scatter(Wg_elc[ii2, :].value, f_elc[ii2, :].value, fc="g", ec="g", **skw)
axe.scatter(Wg_elc[ii3, :].value, f_elc[ii3, :].value, fc="r", ec="r", **skw)

vc_axes = [axi, axe]
vc_labels = ["(i) Ion eflux", "(j) Elc eflux"]
fig.align_ylabels(vc_axes)
kw = dict(x=0.1, transform=axi.transAxes, fontsize="small")
axi.text(y=0.9, s=t_ion[ii1].astype(str)[11:19], color="b", **kw)
axi.text(y=0.8, s=t_ion[ii2].astype(str)[11:19], color="g", **kw)
axi.text(y=0.7, s=t_ion[ii3].astype(str)[11:19], color="r", **kw)

for (i, ax) in enumerate(vc_axes):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-1, 1e3)
    ax.set_ylim(1e2, 1e8)
    ax.set_yticks(10 ** np.arange(2, 8, 1))
    ax.set_ylabel(f"{f_ion.unit:latex_inline}")
    ax.text(0.05, 0.04, vc_labels[i], transform=ax.transAxes, bbox=dict(facecolor="wheat", alpha=0.9))
    for j in range(2):
        ax.axvline(icutoffs[j].value if i == 0 else ecutoffs[j].value, c="magenta", ls="--", lw=2)

    if i == 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(f"Energy ({Wg_ion.unit:latex_inline})", fontsize="small")

fig.savefig(lib.plot_dir / "Fig1_fpi_feeps_combined_example.png", dpi=600, transparent=True)
