import numpy as np
from lib.utils import read_data
from tvolib import mpl_utils as mu


def plot(interval):
    t_ion = read_data(f"/postprocess/interval_{interval}/ion/t").astype("datetime64[ns]")
    tg_ion = read_data(f"/postprocess/interval_{interval}/ion/tg").astype("datetime64[ns]")
    E_ion = read_data(f"/postprocess/interval_{interval}/ion/f_omni_energy")
    f_ion = read_data(f"/postprocess/interval_{interval}/ion/f_omni")
    N_ion = read_data(f"/postprocess/interval_{interval}/ion/N")
    P_ion = read_data(f"/postprocess/interval_{interval}/ion/P_scalar")
    P_nt_ion = read_data(f"/postprocess/interval_{interval}/ion/P_scalar_nt")

    t_elc = read_data(f"/postprocess/interval_{interval}/elc/t").astype("datetime64[ns]")
    tg_elc = read_data(f"/postprocess/interval_{interval}/elc/tg").astype("datetime64[ns]")
    E_elc = read_data(f"/postprocess/interval_{interval}/elc/f_omni_energy")
    f_elc = read_data(f"/postprocess/interval_{interval}/elc/f_omni")
    N_elc = read_data(f"/postprocess/interval_{interval}/elc/N")
    P_elc = read_data(f"/postprocess/interval_{interval}/elc/P_scalar")
    P_nt_elc = read_data(f"/postprocess/interval_{interval}/elc/P_scalar_nt")

    fig, axes = mu.plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    cax = mu.add_colorbar(ax := axes[0])
    im = ax.pcolormesh(tg_ion, E_ion.value, f_ion.value, cmap="jet", norm=mu.mplc.LogNorm())
    fig.colorbar(im, cax=cax)
    ax.set_ylabel(f"{E_ion.unit:latex_inline}")
    ax.set_ylim(E_ion.value.min(), E_ion.value.max())
    ax.set_yscale("log")

    cax = mu.add_colorbar(ax := axes[1])
    im = ax.pcolormesh(tg_ion, E_elc.value, f_elc.value, cmap="jet", norm=mu.mplc.LogNorm())
    fig.colorbar(im, cax=cax)
    ax.set_ylabel(f"{E_elc.unit:latex_inline}")
    ax.set_ylim(E_elc.value.min(), E_elc.value.max())
    ax.set_yscale("log")

    mu.add_colorbar(ax := axes[2]).remove()
    ax.plot(t_ion, N_ion.value, "-r")
    ax.plot(t_elc, N_elc.value, "-b")
    ax.set_ylabel(f"{N_ion.unit:latex_inline}")

    mu.add_colorbar(ax := axes[3]).remove()
    ax.plot(t_ion, P_ion.value, "-r")
    ax.plot(t_elc, P_elc.value, "-b")
    ax.set_ylabel(f"{P_ion.unit:latex_inline}")

    mu.add_colorbar(ax := axes[4]).remove()
    ax.plot(t_ion, P_nt_ion / P_ion * 100, "-r")
    ax.plot(t_elc, P_nt_elc / P_elc * 100, "-b")
    ax.set_ylabel("%")

    for (i, ax) in enumerate(axes):
        ax.set_xlim(np.datetime64("2017-07-26T07:10"), np.datetime64("2017-07-26T07:50"))
        mu.format_datetime_axis(ax)

    fig.tight_layout(h_pad=0.05)
    fig.savefig("test.png")


if __name__ == "__main__":
    plot(418)
