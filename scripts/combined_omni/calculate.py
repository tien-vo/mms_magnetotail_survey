import astropy.units as u
import h5py as h5
import numpy as np
import tvolib as tv
from background import f_omni_background
from integrator import omni_integrate
from pathos.pools import ProcessPool as Pool

import lib
from lib.utils import read_data, read_num_intervals


def combine_omni(
    interval,
    species="ion",
    N_ext=5,
    mask_cutoff=True,
    bg_remove=False,
    factor=1,
):
    t_fpi = read_data(f"mms1/{species}-fpi-moms/interval_{interval}/t").astype(
        "datetime64[ns]"
    )
    f_fpi = read_data(f"mms1/{species}-fpi-moms/interval_{interval}/f_omni")
    E_fpi = read_data(
        f"mms1/{species}-fpi-moms/interval_{interval}/f_omni_energy"
    )
    Vsc = read_data(f"mms1/{species}-fpi-moms/interval_{interval}/Vsc")
    if mask_cutoff:
        f_fpi[
            (E_fpi < 60 * u.eV) | (E_fpi < np.abs(Vsc)[:, np.newaxis])
        ] = np.nan
    if bg_remove:
        f_sorted = np.take_along_axis(f_fpi, np.argsort(f_fpi, axis=1), axis=1)
        f_fpi = f_fpi - np.nanmean(f_sorted[:, :5], axis=1)[:, np.newaxis]
        f_fpi[
            f_fpi <= f_omni_background(E_fpi, species=species, factor=factor)
        ] = np.nan

    t_feeps = read_data(f"mms1/{species}-feeps/interval_{interval}/t").astype(
        "datetime64[ns]"
    )
    f_feeps = read_data(f"mms1/{species}-feeps/interval_{interval}/f_omni")
    E_feeps = np.tile(
        read_data(f"mms1/{species}-feeps/interval_{interval}/f_omni_energy"),
        (t_fpi.shape[0], 1),
    )
    # NOTE: recalculate the 2/3rd average
    window = (
        2 / 3 * (19.67 * u.s / tv.numeric.sampling_period(t_feeps)).decompose()
    )
    f_feeps = tv.numeric.move_avg(f_feeps, (window, 1), window="gauss")
    # END NOTE
    f_feeps = tv.numeric.interpol(f_feeps, t_feeps, t_fpi, window="box")

    # Sanity check
    assert f_fpi.unit == f_feeps.unit
    assert E_fpi.unit == E_feeps.unit

    # Extrapolate
    slope = np.log10(f_feeps[:, 0] / f_fpi[:, -1]) / np.log10(
        E_feeps[:, 0] / E_fpi[:, -1]
    )
    E_ext = (
        np.logspace(
            np.log10(E_fpi[:, -1].value),
            np.log10(E_feeps[:, 0].value),
            N_ext + 2,
        ).T[:, 1:-1]
        * E_fpi.unit
    )
    f_ext = (
        np.power(
            10,
            (
                np.log10(f_fpi[:, -1].value)
                - np.log10(E_fpi[:, -1].value) * slope
            )[:, np.newaxis],
        )
        * np.power(E_ext.value, slope[:, np.newaxis])
    ) * f_fpi.unit

    # Concatenate distribution functions and integrate for scalar moments
    E = np.concatenate((E_fpi, E_ext, E_feeps), axis=1)
    f = np.concatenate((f_fpi, f_ext, f_feeps), axis=1)
    t = t_fpi
    tg = np.tile(t, (E.shape[1], 1)).T
    N, P_scalar = omni_integrate(E + Vsc[:, np.newaxis], f, species=species)
    nt_idx = E_fpi.shape[1] + 1
    N_nt, P_scalar_nt = omni_integrate(
        E[:, nt_idx:] + Vsc[:, np.newaxis], f[:, nt_idx:], species=species
    )

    h5f = h5.File(lib.postprocess_dir / f"interval_{interval}.h5", "a")

    if (where := f"/{species}") in h5f:
        del h5f[where]

    h5f.create_dataset(f"{where}/t", data=t.astype("f8"))
    h5f.create_dataset(f"{where}/tg", data=tg.astype("f8"))
    for name, var in dict(
        f_omni=f,
        f_omni_energy=E,
        N=N,
        P_scalar=P_scalar,
        N_nt=N_nt,
        P_scalar_nt=P_scalar_nt,
    ).items():
        h5d = h5f.create_dataset(f"{where}/{name}", data=var.value)
        h5d.attrs["unit"] = str(var.unit)

    print(
        f"Calculated combined {species} omni distribution and scalar moments for interval {interval}"
    )


if __name__ == "__main__":
    intervals = range(read_num_intervals())
    with Pool() as p:
        for _ in p.uimap(
            lambda i: combine_omni(
                i, species="ion", bg_remove=True, factor=0.8
            ),
            intervals,
        ):
            pass

        for _ in p.uimap(lambda i: combine_omni(i, species="elc"), intervals):
            pass
