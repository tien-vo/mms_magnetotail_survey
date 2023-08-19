import warnings

import numpy as np
import astropy.units as u
from tvolib import mpl_utils as mu
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian2DKernel

import lib
from lib.utils import read_data

warnings.filterwarnings("ignore")

where = "/analysis/compare_B14"
bg = read_data(f"{where}/bg")
Bg = read_data(f"{where}/Bg")
Ng = read_data(f"{where}/Ng")
Jg = read_data(f"{where}/Jg")
H_beta_N_elc = read_data(f"{where}/H_beta_N_elc")
H_beta_B_xy = read_data(f"{where}/H_beta_B_xy")
H_beta_J_para = read_data(f"{where}/H_beta_J_para")
H_beta_J_perp = read_data(f"{where}/H_beta_J_perp")
H_beta_J_err = read_data(f"{where}/H_beta_J_err")

T = 30 * u.s
dt_ptcl = 5 * u.s
dt_fields = 1 / 16 * u.s
H_beta_N_elc[H_beta_N_elc < int((T / dt_ptcl).decompose())] = np.nan
H_beta_B_xy[H_beta_B_xy < int((T / dt_ptcl).decompose())] = np.nan
H_beta_J_para[H_beta_J_para < int((T / dt_fields).decompose())] = np.nan
H_beta_J_perp[H_beta_J_perp < int((T / dt_fields).decompose())] = np.nan
H_beta_J_err[H_beta_J_err < int((T / dt_fields).decompose())] = np.nan
P_beta_N_elc = H_beta_N_elc / np.nansum(H_beta_N_elc) * 100
P_beta_B_xy = H_beta_B_xy / np.nansum(H_beta_B_xy) * 100
P_beta_J_para = H_beta_J_para / np.nansum(H_beta_J_para) * 100
P_beta_J_perp = H_beta_J_perp / np.nansum(H_beta_J_perp) * 100
P_beta_J_err = H_beta_J_err / np.nansum(H_beta_J_err) * 100

P_beta_J_para_err = (
    P_beta_J_err * np.nanmax(P_beta_J_para, axis=1)[:, np.newaxis] / np.nanmax(P_beta_J_err, axis=1)[:, np.newaxis]
)
P_beta_J_perp_err = (
    P_beta_J_err * np.nanmax(P_beta_J_perp, axis=1)[:, np.newaxis] / np.nanmax(P_beta_J_err, axis=1)[:, np.newaxis]
)

J_arr = Jg[0, :]
beta_arr = bg[:, 0]
i1 = np.argmin(np.abs(beta_arr - 1e-2))
i2 = np.argmin(np.abs(beta_arr - 2e0))

H_para_L = P_beta_J_para[i1, :]
H_para_err_L = P_beta_J_para_err[i1, :]
H_para_S = P_beta_J_para[i2, :]
H_para_err_S = P_beta_J_para_err[i2, :]
J_para_L = J_arr[cond := (~np.isnan(H_para_L))]
H_para_L = H_para_L[cond]
J_para_err_L = J_arr[cond := (~np.isnan(H_para_err_L))]
H_para_err_L = H_para_err_L[cond]
J_para_S = J_arr[cond := (~np.isnan(H_para_S))]
H_para_S = H_para_S[cond]
J_para_err_S = J_arr[cond := (~np.isnan(H_para_err_S))]
H_para_err_S = H_para_err_S[cond]

H_perp_L = P_beta_J_perp[i1, :]
H_perp_err_L = P_beta_J_perp_err[i1, :]
H_perp_S = P_beta_J_perp[i2, :]
H_perp_err_S = P_beta_J_perp_err[i2, :]
J_perp_L = J_arr[cond := (~np.isnan(H_perp_L))]
H_perp_L = H_perp_L[cond]
J_perp_err_L = J_arr[cond := (~np.isnan(H_perp_err_L))]
H_perp_err_L = H_perp_err_L[cond]
J_perp_S = J_arr[cond := (~np.isnan(H_perp_S))]
H_perp_S = H_perp_S[cond]
J_perp_err_S = J_arr[cond := (~np.isnan(H_perp_err_S))]
H_perp_err_S = H_perp_err_S[cond]

J_int = np.linspace(Jg.min(), Jg.max(), 1000)
H_para_int_L = interp1d(J_para_L, H_para_L, fill_value=np.nan, kind="cubic", bounds_error=False)(J_int)
H_para_err_int_L = interp1d(J_para_err_L, H_para_err_L, fill_value=np.nan, kind="cubic", bounds_error=False)(J_int)
H_para_int_S = interp1d(J_para_S, H_para_S, fill_value=np.nan, kind="cubic", bounds_error=False)(J_int)
H_para_err_int_S = interp1d(J_para_err_S, H_para_err_S, fill_value=np.nan, kind="cubic", bounds_error=False)(J_int)
H_perp_int_L = interp1d(J_perp_L, H_perp_L, fill_value=np.nan, kind="cubic", bounds_error=False)(J_int)
H_perp_err_int_L = interp1d(J_perp_err_L, H_perp_err_L, fill_value=np.nan, kind="cubic", bounds_error=False)(J_int)
H_perp_int_S = interp1d(J_perp_S, H_perp_S, fill_value=np.nan, kind="cubic", bounds_error=False)(J_int)
H_perp_err_int_S = interp1d(J_perp_err_S, H_perp_err_S, fill_value=np.nan, kind="cubic", bounds_error=False)(J_int)
P1 = np.nansum(H_para_int_L[(H_para_int_L >= H_para_err_int_L) | np.isnan(H_para_err_int_L)]) / np.nansum(H_para_int_L)
P2 = np.nansum(H_para_int_S[(H_para_int_S >= H_para_err_int_S) | np.isnan(H_para_err_int_S)]) / np.nansum(H_para_int_S)
P3 = np.nansum(H_perp_int_L[(H_perp_int_L >= H_perp_err_int_L) | np.isnan(H_perp_err_int_L)]) / np.nansum(H_perp_int_L)
P4 = np.nansum(H_perp_int_S[(H_perp_int_S >= H_perp_err_int_S) | np.isnan(H_perp_err_int_S)]) / np.nansum(H_perp_int_S)
H_para_err_int_L[np.isnan(H_para_err_int_L) | (H_para_err_int_L < np.nanmin(H_para_int_L))] = np.nanmin(H_para_int_L)
H_para_err_int_S[np.isnan(H_para_err_int_S) | (H_para_err_int_S < np.nanmin(H_para_int_S))] = np.nanmin(H_para_int_S)
H_perp_err_int_L[np.isnan(H_perp_err_int_L) | (H_perp_err_int_L < np.nanmin(H_perp_int_L))] = np.nanmin(H_perp_int_L)
H_perp_err_int_S[np.isnan(H_perp_err_int_S) | (H_perp_err_int_S < np.nanmin(H_perp_int_S))] = np.nanmin(H_perp_int_S)
print(P1, P2, P3, P4)

fig, axes = mu.plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))

axes[1].plot(J_perp_L, H_perp_L, "-k", label="$\\beta_i=10^{-2}$")
axes[1].plot(J_perp_S, H_perp_S, "-r", label="$\\beta_i=2$")
fig.legend(frameon=False, loc="upper right", fontsize="x-small", bbox_to_anchor=(0.97, 0.95))

axes[0].plot(J_para_L, H_para_L, "-k")
axes[0].plot(J_para_err_L, H_para_err_L, "--k")
axes[0].plot(J_para_S, H_para_S, "-r")
axes[0].plot(J_para_err_S, H_para_err_S, "--r")
axes[0].fill_between(
    J_int.value, H_para_int_L, H_para_err_int_L,
    color="k", alpha=0.1, where=H_para_int_L > H_para_err_int_L, linewidth=0,
    label=f"Probability = {P1:.2f}",
)
axes[0].fill_between(
    J_int.value, H_para_int_S, H_para_err_int_S,
    color="r", alpha=0.1, where=H_para_int_S > H_para_err_int_S, linewidth=0,
    label=f"Probability = {P2:.2f}",
)
axes[0].text(3.0, 2e-4, "Noise", c="k", fontsize="x-small")
axes[0].text(9, 1.6e-4, "Noise", c="r", fontsize="x-small")
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles, labels=labels, loc="upper left", frameon=False, fontsize="x-small")

axes[1].plot(J_perp_err_L, H_perp_err_L, "--k")
axes[1].plot(J_perp_err_S, H_perp_err_S, "--r")
axes[1].fill_between(
    J_int.value, H_perp_int_L, H_perp_err_int_L,
    color="k", alpha=0.1, where=(H_perp_int_L - H_perp_err_int_L) >= 1e-4, linewidth=0,
    label=f"Probability = {P3:.2f}",
)
axes[1].fill_between(
    J_int.value, H_perp_int_S, H_perp_err_int_S,
    color="r", alpha=0.1, where=(H_perp_int_S - H_perp_err_int_S) >= 1e-4, linewidth=0,
    label=f"Probability = {P4:.2f}",
)
axes[1].text(4, 1.6e-4, "Noise", c="k", fontsize="x-small")
axes[1].text(9, 1.6e-4, "Noise", c="r", fontsize="x-small")
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles[2:], labels=labels[2:], loc="upper left", frameon=False, fontsize="x-small")


axes[0].set_ylabel("Occurrence rate (%)")
axes[0].set_xlabel(f"$J_\|$ ({u.Unit('nA m-2'):latex_inline})")
axes[1].set_xlabel(f"$J_\perp$ ({u.Unit('nA m-2'):latex_inline})")
for (i, ax) in enumerate(axes):
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e-1)
    ax.axvline(0, ls=":")

fig.tight_layout()
fig.savefig(lib.plot_dir / "Fig4_J_slices.png", dpi=600, transparent=True)

fig, axes = mu.plt.subplots(2, 2, figsize=(12, 10))
kw = dict(cmap="jet", norm=(norm := mu.mplc.LogNorm(1e-4, 1e-1)), alpha=0.5)
ctkw = dict(linewidths=2, cmap="jet", levels=np.logspace(np.log10(5e-4), np.log10(5e-2), 5), norm=norm,)

k1 = Gaussian2DKernel(2)
k2 = Gaussian2DKernel(3)

mu.add_colorbar(ax := axes[0, 0]).remove()
ax.pcolormesh(bg, Jg.value, P_beta_J_para, **kw)
ax.contour(bg, Jg.value, convolve(P_beta_J_para, k1), **ctkw)
ax.contour(bg, Jg.value, convolve(P_beta_J_para_err, k1), linestyles="dashed", **ctkw)

mu.add_colorbar(ax := axes[0, 1]).remove()
ax.pcolormesh(bg, Jg.value, P_beta_J_perp, **kw)
ax.contour(bg, Jg.value, convolve(P_beta_J_perp, k1), **ctkw)
ax.contour(bg, Jg.value, convolve(P_beta_J_perp_err, k1), linestyles="dashed", **ctkw)

mu.add_colorbar(ax := axes[1, 0]).remove()
ax.pcolormesh(bg, Ng.value, P_beta_N_elc, **kw)
ax.contour(bg, Ng.value, convolve(P_beta_N_elc, k2), **ctkw)
ax.set_yscale("log")

cax = mu.add_colorbar(ax := axes[1, 1])
im = ax.pcolormesh(bg, Bg.value, P_beta_B_xy, **kw)
ax.contour(bg, Bg.value, convolve(P_beta_B_xy, k2), **ctkw)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Occurrence rate (%)")
cb.set_alpha(1)
fig.draw_without_rendering()

# Annotate
akw = dict(color="r", arrowprops=dict(color="r", width=0.5), fontsize="small")
axes[1, 0].annotate("Lobe", (4e-2, 6e-3), xytext=(1e1, 1.5e-3), **akw)
axes[1, 0].annotate("Sheet", (2e1, 8e-2), xytext=(1e2, 8e-3), **akw)
axes[1, 1].annotate("Lobe", (1e-2, 14), xytext=(1e-3, 5), **akw)
axes[1, 1].annotate("Inner PS", (3e1, 6), xytext=(1e2, 10), **akw)
axes[1, 1].annotate("PSBL", (5e0, 20), xytext=(5e2, 25), **akw)

axes[0, 0].set_ylabel(f"$J_\\|$ ({u.Unit('nA m-2'):latex_inline})", fontsize="large")
axes[0, 1].set_ylabel(f"$J_\\perp$ ({u.Unit('nA m-2'):latex_inline})", fontsize="large")
axes[1, 0].set_ylabel(f"$n_e$ ({u.Unit('cm-3'):latex_inline})", fontsize="large")
axes[1, 1].set_ylabel(f"$B_{{xy}}$ ({u.Unit('nT'):latex_inline})", fontsize="large")
tkw = dict(color="k", fontsize="large", bbox=dict(facecolor="wheat", alpha=0.9))
axes[0, 0].text(0.05, 0.88, "(a)", transform=axes[0, 0].transAxes, **tkw)
axes[0, 1].text(0.05, 0.88, "(b)", transform=axes[0, 1].transAxes, **tkw)
axes[1, 0].text(0.05, 0.88, "(c)", transform=axes[1, 0].transAxes, **tkw)
axes[1, 1].text(0.05, 0.88, "(d)", transform=axes[1, 1].transAxes, **tkw)
axes[1, 1].axhline(14, color="k", ls="--", lw=2)
axes[1, 0].axhline(5e-2, color="k", ls="--", lw=2)
axes[0, 0].set_ylim(Jg.value.min(), Jg.value.max())
axes[0, 1].set_ylim(Jg.value.min(), Jg.value.max())
axes[1, 0].set_ylim(Ng.value.min(), Ng.value.max())
axes[1, 1].set_ylim(Bg.value.min(), Bg.value.max())
for (i, j) in np.ndindex(axes.shape):
    ax = axes[i, j]
    ax.set_facecolor("darkgray")
    ax.set_xlim(bg.min(), bg.max())
    ax.set_xscale("log")
    ax.axvline(0.2, color="k", ls="--", lw=2)
    ax.set_xlabel("$\\beta_i$", fontsize="large")

fig.tight_layout(h_pad=0.1, w_pad=1)
fig.savefig(lib.plot_dir / "Fig3_compare_B14.png", dpi=600, transparent=True)
