import h5py as h5
import numpy as np
import tvolib as tv
import astropy.units as u

from pathos.pools import ProcessPool as Pool

import lib
from lib.utils import read_data, read_num_intervals


def helper(interval):
    t_ion = read_data(f"/postprocess/interval_{interval}/ion/t").astype("datetime64[ns]")
    t_elc = read_data(f"/postprocess/interval_{interval}/elc/t").astype("datetime64[ns]")
    N_ion = read_data(f"/postprocess/interval_{interval}/ion/N")
    N_elc = read_data(f"/postprocess/interval_{interval}/elc/N")
    P_elc = read_data(f"/postprocess/interval_{interval}/elc/P_scalar")
    assert (t_ion == t_elc).all()

    t_fields = read_data(f"/postprocess/interval_{interval}/barycenter/t").astype("datetime64[ns]")
    B = read_data(f"/postprocess/interval_{interval}/barycenter/B_bc")
    R = read_data(f"/postprocess/interval_{interval}/barycenter/R_bc")

    X = tv.numeric.interpol(R[:, 0], t_fields, t_ion).to(u.R_earth)
    Y = tv.numeric.interpol(R[:, 1], t_fields, t_ion).to(u.R_earth)
    Z = tv.numeric.interpol(R[:, 2], t_fields, t_ion).to(u.R_earth)
    Bx = tv.numeric.interpol(B[:, 0], t_fields, t_ion)
    By = tv.numeric.interpol(B[:, 1], t_fields, t_ion)
    Bz = tv.numeric.interpol(B[:, 2], t_fields, t_ion)
    Nsmooth = np.int64(Tsmooth / tv.numeric.sampling_period(t_fields))
    dBx = tv.numeric.interpol(tv.numeric.move_std(B[:, 0], (Nsmooth,)), t_fields, t_ion)
    dBy = tv.numeric.interpol(tv.numeric.move_std(B[:, 1], (Nsmooth,)), t_fields, t_ion)
    dBz = tv.numeric.interpol(tv.numeric.move_std(B[:, 2], (Nsmooth,)), t_fields, t_ion)
    dB = np.sqrt(dBx**2 + dBy**2 + dBz**2)

    mask = N_elc >= 0.05 * u.Unit("cm-3")
    T_elc = P_elc / N_elc

    H = np.histogramdd((X, Y, Z), bins=bins)[0]
    H_N_ion = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=N_ion[mask])[0]
    H_dB = np.histogramdd((X, Y, Z), bins=bins, weights=dB)[0]
    H_T_elc = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=T_elc[mask])[0]
    return (H, H_N_ion, H_dB, H_T_elc)


if __name__ == "__main__":
    Tsmooth = 5 * u.s

    X_bins = np.arange(-30, 10 + 0.5, 0.5) * u.R_earth
    Y_bins = np.arange(-30, 30 + 0.5, 0.5) * u.R_earth
    Z_bins = np.arange(-10, 10 + 0.5, 0.5) * u.R_earth
    bins = (X_bins, Y_bins, Z_bins)
    Xg, Yg, Zg = np.meshgrid(X_bins[:-1], Y_bins[:-1], Z_bins[:-1], indexing="ij")

    H, H_N_ion, H_dB, H_T_elc = 0, 0, 0, 0
    with Pool(8) as p:
        count = 0
        for data in p.uimap(helper, range(N := read_num_intervals())):
            (_H, _H_N_ion, _H_dB, _H_T_elc) = data
            H += _H
            H_N_ion += _H_N_ion
            H_dB += _H_dB
            H_T_elc += _H_T_elc
            count += 1
            print(f"Processed {count}/{N} files.")

    h5f = h5.File(lib.analysis_file, "a")

    if (where := f"/inner_tail") in h5f:
        del h5f[where]

    h5d = h5f.create_dataset(f"{where}/Xg", data=Xg.value)
    h5d.attrs["unit"] = str(Xg.unit)

    h5d = h5f.create_dataset(f"{where}/Yg", data=Yg.value)
    h5d.attrs["unit"] = str(Yg.unit)

    h5d = h5f.create_dataset(f"{where}/Zg", data=Zg.value)
    h5d.attrs["unit"] = str(Zg.unit)

    h5f.create_dataset(f"{where}/H", data=H)

    h5d = h5f.create_dataset(f"{where}/H_N_ion", data=H_N_ion.value)
    h5d.attrs["unit"] = str(H_N_ion.unit)

    h5d = h5f.create_dataset(f"{where}/H_dB", data=H_dB.value)
    h5d.attrs["unit"] = str(H_dB.unit)

    h5d = h5f.create_dataset(f"{where}/H_T_elc", data=H_T_elc.value)
    h5d.attrs["unit"] = str(H_T_elc.unit)
