import os
import sys

import numpy as np
import astropy.units as u
from pathos.pools import ProcessPool as Pool

import lib
from lib.utils import read_data


def plot(interval):

    import matplotlib
    matplotlib.use("Agg")
    from tvolib import mpl_utils as mu
    mu.setup(cache=True)

    icutoffs = np.array([28.3, 76.8]) * u.keV
    ecutoffs = np.array([27.5, 65.9]) * u.keV

    # Create figure
    fig, axes = mu.plt.subplots(8, 1, figsize=(12, 11), sharex=True)

    # Barycentric magnetic field
    t = read_data(f"/postprocess/interval_{interval}/barycenter/t").astype("datetime64[ns]")
    B_bc = read_data(f"/postprocess/interval_{interval}/barycenter/B_bc")
    mu.add_colorbar(ax := axes[0]).remove()
    ax.set_ylabel(f"{B_bc.unit:latex_inline}")
    ax.plot(t, B_bc[:, 0], "-b")
    ax.plot(t, B_bc[:, 1], "-g")
    ax.plot(t, B_bc[:, 2], "-r")
    ax.plot(t, np.linalg.norm(B_bc, axis=1), "-k")

    # Barycentric electric field
    E_bc = read_data(f"/postprocess/interval_{interval}/barycenter/E_bc")
    mu.add_colorbar(ax := axes[1]).remove()
    ax.set_ylabel(f"{E_bc.unit:latex_inline}")
    ax.plot(t, E_bc[:, 0], "-b")
    ax.plot(t, E_bc[:, 1], "-g")
    ax.plot(t, E_bc[:, 2], "-r")

    # MMS1 ion velocity
    t = read_data(f"/mms1/ion-fpi-moms/interval_{interval}/t").astype("datetime64[ns]")
    Vi = read_data(f"/mms1/ion-fpi-moms/interval_{interval}/V_gsm").to(u.Unit("1000 km/s"))
    mu.add_colorbar(ax := axes[2]).remove()
    ax.set_ylabel(f"{Vi.unit:latex_inline}")
    ax.plot(t, Vi[:, 0], "-b")
    ax.plot(t, Vi[:, 1], "-g")
    ax.plot(t, Vi[:, 2], "-r")

    # MMS1 ion energy spectrum
    t_ion = read_data(f"/postprocess/interval_{interval}/ion/t").astype("datetime64[ns]")
    tg_ion = read_data(f"/postprocess/interval_{interval}/ion/tg").astype("datetime64[ns]")
    Wg_ion = read_data(f"/postprocess/interval_{interval}/ion/f_omni_energy")
    f_ion = read_data(f"/postprocess/interval_{interval}/ion/f_omni")
    cax = mu.add_colorbar(ax := axes[3])
    ax.set_ylabel(f"{Wg_ion.unit:latex_inline}")
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e3)
    ax.set_yticks(np.power(10.0, np.arange(-1, 3, 1)))
    im = ax.pcolormesh(tg_ion, Wg_ion.value, f_ion.value, norm=mu.mplc.LogNorm(1e2, 1e8), cmap="jet", rasterized=True)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"{f_ion.unit:latex_inline}", fontsize="x-small")
    ax.axhline(icutoffs[0].value, c="magenta", ls="--", lw=2)
    ax.axhline(icutoffs[1].value, c="magenta", ls="--", lw=2)
    ax.set_facecolor("silver")

    # MMS1 elc energy spectrum
    t_elc = read_data(f"/postprocess/interval_{interval}/elc/t").astype("datetime64[ns]")
    tg_elc = read_data(f"/postprocess/interval_{interval}/elc/tg").astype("datetime64[ns]")
    Wg_elc = read_data(f"/postprocess/interval_{interval}/elc/f_omni_energy")
    f_elc = read_data(f"/postprocess/interval_{interval}/elc/f_omni")
    cax = mu.add_colorbar(ax := axes[4])
    ax.set_ylabel(f"{Wg_elc.unit:latex_inline}")
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e3)
    ax.set_yticks(np.power(10.0, np.arange(-1, 3, 1)))
    im = ax.pcolormesh(tg_elc, Wg_elc.value, f_elc.value, norm=mu.mplc.LogNorm(1e2, 1e8), cmap="jet", rasterized=True)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"{f_elc.unit:latex_inline}", fontsize="x-small")
    ax.axhline(ecutoffs[0].value, c="magenta", ls="--", lw=2)
    ax.axhline(ecutoffs[1].value, c="magenta", ls="--", lw=2)
    ax.set_facecolor("silver")

    # MMS1 densities
    N_ion = read_data(f"/postprocess/interval_{interval}/ion/N")
    N_elc = read_data(f"/postprocess/interval_{interval}/elc/N")
    mu.add_colorbar(ax := axes[5]).remove()
    ax.set_ylabel(f"{N_ion.unit:latex_inline}")
    ax.plot(t_elc, N_elc, "-k", lw=2)
    ax.plot(t_ion, N_ion, "--r", lw=2)
    kw = dict(x=1.02, transform=ax.transAxes, fontsize="small")
    ax.text(y=0.8, s="Ion", c="r", **kw)
    ax.text(y=0.2, s="Elc", c="k", **kw)

    # MMS1 pressures
    P_ion = read_data(f"/postprocess/interval_{interval}/ion/P_scalar")
    P_elc = read_data(f"/postprocess/interval_{interval}/elc/P_scalar")
    mu.add_colorbar(ax := axes[6]).remove()
    ax.set_ylabel(f"{P_ion.unit:latex_inline}")
    ax.plot(t_elc, P_elc, "-k", lw=2)
    ax.plot(t_ion, P_ion, "--r", lw=2)
    kw = dict(x=1.02, transform=ax.transAxes, fontsize="small")
    ax.text(y=0.8, s="Ion", c="r", **kw)
    ax.text(y=0.2, s="Elc", c="k", **kw)

    # MMS1 nonthermal pressures
    Pnt_ion = read_data(f"/postprocess/interval_{interval}/ion/P_scalar_nt")
    Pnt_elc = read_data(f"/postprocess/interval_{interval}/elc/P_scalar_nt")
    mu.add_colorbar(ax := axes[7]).remove()
    ax.set_ylabel(f"{Pnt_ion.unit:latex_inline}")
    ax.plot(t_elc, Pnt_elc, "-k", lw=2)
    ax.plot(t_ion, Pnt_ion, "--r", lw=2)
    kw = dict(x=1.02, transform=ax.transAxes, fontsize="small")
    ax.text(y=0.8, s="Ion", c="r", **kw)
    ax.text(y=0.2, s="Elc", c="k", **kw)

    for (i, ax) in enumerate(axes):
        mu.format_datetime_axis(ax)

    fig.suptitle(f"Interval {interval}")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.05)
    fig.savefig(f"tmp/{interval}.png")
    mu.plt.close(fig)
    sys.stdout.write(f"Plotted interval {interval}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    os.makedirs("tmp", exist_ok=True)
    with Pool(8, maxtasksperchild=1) as p:
        for _ in p.uimap(plot, range(1919)):
            pass
