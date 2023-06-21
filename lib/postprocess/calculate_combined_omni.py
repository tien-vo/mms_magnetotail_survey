from lib.utils import read_dataset, read_num_intervals
import astropy.constants as c
import astropy.units as u
import tvolib as tv
import numpy as np
import h5py as h5
import warnings
import lib


def f_omni_background(E, species="ion", factor=1):
    f_unit = u.Unit("cm-2 s-1 sr-1")
    gaps = dict(
        ion=np.array([28.3, 76.8]) * u.keV,
        elc=np.array([27.5, 65.9]) * u.keV,
    )
    bg_levels = dict(
        ion=np.array([3e4, 4e3]) * f_unit,
        elc=np.array([6e5, 8e2]) * f_unit,
    )

    f = 1e-4 * np.ones(E.shape) * f_unit
    f[E <= gaps[species][0]] = factor * bg_levels[species][0]
    f[E >= gaps[species][1]] = factor * bg_levels[species][1]
    if species == "elc":
        f[(7e-1 * u.keV <= E) & (E <= gaps[species][0])] = factor * 1e5 * f_unit

    return f


def omni_integrate(E, f, species="ion"):
    charge = c.si.e if species == "ion" else -c.si.e
    mass = c.si.m_p if species == "ion" else c.si.m_e
    assert f.unit == "cm-2 s-1 sr-1"

    f[np.isnan(f)] = 0
    N = (4 * np.pi * np.sqrt(mass / 2) * np.trapz(f * u.sr * E ** (-3 / 2), x=E, axis=1)).to(u.Unit("cm-3"))
    P = (4 * np.pi * np.sqrt(mass / 2) * np.trapz(f * u.sr * E ** (-1 / 2), x=E, axis=1)).to(u.Unit("keV cm-3"))

    return N, P


def combine_omni(interval, species="ion", N_ext=5, mask_cutoff=True, bg_remove=False, factor=1):

    t_fpi = read_dataset(f"mms1/{species}-fpi-moms/interval_{interval}/t").astype("datetime64[ns]")
    f_fpi = read_dataset(f"mms1/{species}-fpi-moms/interval_{interval}/f_omni")
    E_fpi = read_dataset(f"mms1/{species}-fpi-moms/interval_{interval}/f_omni_energy")
    Vsc = read_dataset(f"mms1/{species}-fpi-moms/interval_{interval}/Vsc")
    if mask_cutoff:
        f_fpi[(E_fpi < 60 * u.eV) | (E_fpi < np.abs(Vsc)[:, np.newaxis])] = np.nan
    if bg_remove:
        f_sorted = np.take_along_axis(f_fpi, np.argsort(f_fpi, axis=1), axis=1)
        f_fpi = f_fpi - np.nanmean(f_sorted[:, :5], axis=1)[:, np.newaxis]
        f_fpi[f_fpi <= f_omni_background(E_fpi, species=species, factor=factor)] = np.nan

    t_feeps = read_dataset(f"mms1/{species}-feeps/interval_{interval}/t").astype("datetime64[ns]")
    f_feeps = read_dataset(f"mms1/{species}-feeps/interval_{interval}/f_omni")
    E_feeps = np.tile(read_dataset(f"mms1/{species}-feeps/interval_{interval}/f_omni_energy"), (t_fpi.shape[0], 1))
    # NOTE: recalculate the 2/3rd average
    window = 2 / 3 * (19.67 * u.s / tv.utils.sampling_period(t_feeps)).decompose()
    f_feeps = tv.numeric.move_avg(f_feeps, (window, 1), smooth=True)
    # END NOTE
    f_feeps = tv.numeric.interpol(f_feeps, t_feeps, t_fpi, avg=True)

    # Sanity check
    assert f_fpi.unit == f_feeps.unit
    assert E_fpi.unit == E_feeps.unit

    # Extrapolate
    slope = np.log10(f_feeps[:, 0] / f_fpi[:, -1]) / np.log10(E_feeps[:, 0] / E_fpi[:, -1])
    E_ext = np.logspace(np.log10(E_fpi[:, -1].value), np.log10(E_feeps[:, 0].value), N_ext + 2).T[:, 1:-1] * E_fpi.unit
    f_ext = (
        np.power(10, (np.log10(f_fpi[:, -1].value) - np.log10(E_fpi[:, -1].value) * slope)[:, np.newaxis])
        * np.power(E_ext.value, slope[:, np.newaxis])
    ) * f_fpi.unit

    # Concatenate distribution functions and integrate for scalar moments
    E = np.concatenate((E_fpi, E_ext, E_feeps), axis=1)
    f = np.concatenate((f_fpi, f_ext, f_feeps), axis=1)
    t = t_fpi
    tg = np.tile(t, (E.shape[1], 1)).T
    N, P_scalar = omni_integrate(E + Vsc[:, np.newaxis], f, species=species)
    nt_idx = E_fpi.shape[1] + 1
    N_nt, P_scalar_nt = omni_integrate(E[:, nt_idx:] + Vsc[:, np.newaxis], f[:, nt_idx:], species=species)

    with h5.File(f"{lib.postprocess_dir}/interval_{interval}.h5", "a") as h5f:
        if (where := f"/{species}") in h5f:
            del h5f[where]

        h5f.create_dataset(f"{where}/t", data=t.astype("f8"))
        h5f.create_dataset(f"{where}/tg", data=tg.astype("f8"))
        for name, var in dict(
            f_omni=f, f_omni_energy=E, N=N, P_scalar=P_scalar, N_nt=N_nt, P_scalar_nt=P_scalar_nt,
        ).items():
            h5d = h5f.create_dataset(f"{where}/{name}", data=var.value)
            h5d.attrs["unit"] = str(var.unit)

    print(f"Calculated combined omni distribution and scalar moments for interval {interval}")



def run():

    #for interval in range(read_num_intervals()):
    #for interval in range(1):
    for interval in [418,]:
        combine_omni(interval, species="ion", bg_remove=True, factor=0.8)
        combine_omni(interval, species="elc",)


def plot_test(interval):
    t_ion = read_dataset(f"/postprocess/interval_{interval}/ion/t").astype("datetime64[ns]")
    tg_ion = read_dataset(f"/postprocess/interval_{interval}/ion/tg").astype("datetime64[ns]")
    E_ion = read_dataset(f"/postprocess/interval_{interval}/ion/f_omni_energy")
    f_ion = read_dataset(f"/postprocess/interval_{interval}/ion/f_omni")
    N_ion = read_dataset(f"/postprocess/interval_{interval}/ion/N")
    P_ion = read_dataset(f"/postprocess/interval_{interval}/ion/P_scalar")
    P_nt_ion = read_dataset(f"/postprocess/interval_{interval}/ion/P_scalar_nt")

    t_elc = read_dataset(f"/postprocess/interval_{interval}/elc/t").astype("datetime64[ns]")
    tg_elc = read_dataset(f"/postprocess/interval_{interval}/elc/tg").astype("datetime64[ns]")
    E_elc = read_dataset(f"/postprocess/interval_{interval}/elc/f_omni_energy")
    f_elc = read_dataset(f"/postprocess/interval_{interval}/elc/f_omni")
    N_elc = read_dataset(f"/postprocess/interval_{interval}/elc/N")
    P_elc = read_dataset(f"/postprocess/interval_{interval}/elc/P_scalar")
    P_nt_elc = read_dataset(f"/postprocess/interval_{interval}/elc/P_scalar_nt")

    import matplotlib.pyplot as plt
    import mpl_utils as mu

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

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

    #for (i, ax) in enumerate(axes):
    #    ax.set_xlim(np.datetime64("2017-07-26T07:10"), np.datetime64("2017-07-26T07:45"))

    fig.tight_layout(h_pad=0.05)
    fig.savefig("test.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    #run()
    plot_test(418)
