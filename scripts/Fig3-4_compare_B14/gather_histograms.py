import h5py as h5
import numpy as np
import tvolib as tv
import astropy.units as u
import astropy.constants as c

from pathos.pools import ProcessPool as Pool
from tvolib.models.magnetopause_model import Lin10MagnetopauseModel

import lib
from lib.utils import read_data, read_num_intervals


def helper(interval):
    t_fields = read_data(f"/postprocess/interval_{interval}/barycenter/t").astype("datetime64[ns]")
    B = read_data(f"/postprocess/interval_{interval}/barycenter/B_bc")
    R = read_data(f"/postprocess/interval_{interval}/barycenter/R_bc").to(u.R_earth)
    J = read_data(f"/postprocess/interval_{interval}/barycenter/J_clm")
    J_err = read_data(f"/postprocess/interval_{interval}/barycenter/J_err")

    t_ion = read_data(f"/postprocess/interval_{interval}/ion/t").astype("datetime64[ns]")
    t_elc = read_data(f"/postprocess/interval_{interval}/elc/t").astype("datetime64[ns]")
    P_ion = read_data(f"/postprocess/interval_{interval}/ion/P_scalar")
    N_elc = read_data(f"/postprocess/interval_{interval}/elc/N")
    assert (t_ion == t_elc).all()

    # ---- Calculations
    # FAC
    z_hat = np.zeros(B.shape)
    z_hat[:, 2] = 1
    e_para = B / np.linalg.norm(B, axis=1, keepdims=True)
    e_perp = np.cross(z_hat, B, axis=1)
    e_perp = e_perp / np.linalg.norm(e_perp, axis=1, keepdims=True)
    J_para = np.einsum("...i,...i", J, e_para)
    J_perp = np.einsum("...i,...i", J, e_perp)
    Bmag = np.linalg.norm(B, axis=1)
    B_xy = tv.numeric.interpol(np.linalg.norm(B[:, 0:2], axis=1), t_fields, t_ion)
    beta_fields = (tv.numeric.interpol(P_ion, t_ion, t_fields) / (Bmag ** 2 / 2 / c.si.mu0)).decompose()
    beta_ptcl = (P_ion / (tv.numeric.interpol(Bmag ** 2 / 2 / c.si.mu0, t_fields, t_ion, window="box"))).decompose()

    # Mask
    R_XY = np.sqrt(R[:, 0] ** 2 + R[:, 1] ** 2).value
    T_XY = np.arctan2(R[:, 1], R[:, 0]).to(u.rad).value
    mask_fields = R_XY <= model_f(T_XY)
    mask_ptcl = tv.numeric.interpol(R_XY, t_fields, t_ion) <= tv.numeric.interpol(model_f(T_XY), t_fields, t_ion)

    H_beta_N_elc = np.histogram2d(beta_ptcl[mask_ptcl], N_elc[mask_ptcl], bins=(b_bins, N_bins))[0]
    H_beta_B_xy = np.histogram2d(beta_ptcl[mask_ptcl], B_xy[mask_ptcl], bins=(b_bins, B_bins))[0]
    H_beta_J_err = np.histogram2d(beta_fields[mask_fields], J_err[mask_fields], bins=(b_bins, J_bins))[0]
    H_beta_J_para = np.histogram2d(beta_fields[mask_fields], J_para[mask_fields], bins=(b_bins, J_bins))[0]
    H_beta_J_perp = np.histogram2d(beta_fields[mask_fields], J_perp[mask_fields], bins=(b_bins, J_bins))[0]
    return (interval, (H_beta_N_elc, H_beta_B_xy, H_beta_J_para, H_beta_J_perp, H_beta_J_err))


if __name__ == "__main__":
    model = Lin10MagnetopauseModel(pressure=20, bfield=0, tilt_angle=0)
    model_f = model.shape_F(which="XY", rotation_angle=np.radians(-5))

    N = 200
    b_bins, db = np.linspace(-4, 4, N, retstep=True)
    b_bins = np.power(10, np.append(b_bins, b_bins[-1] + db))
    B_bins, dB = np.linspace(0, 40, N, retstep=True)
    B_bins = np.append(B_bins, B_bins[-1] + dB) * u.nT
    N_bins, dN = np.linspace(np.log10(5e-4), 1, N, retstep=True)
    N_bins = np.power(10, np.append(N_bins, N_bins[-1] + dN)) * u.Unit("cm-3")
    J_bins, dJ = np.linspace(-15, 15, N, retstep=True)
    J_bins = np.append(J_bins, J_bins[-1] + dJ) * u.Unit("nA m-2")

    bg, Ng = np.meshgrid(b_bins[:-1], N_bins[:-1], indexing="ij")
    _, Bg = np.meshgrid(b_bins[:-1], B_bins[:-1], indexing="ij")
    _, Jg = np.meshgrid(b_bins[:-1], J_bins[:-1], indexing="ij")

    H_beta_N_elc, H_beta_B_xy, H_beta_J_para, H_beta_J_perp, H_beta_J_err = 0, 0, 0, 0, 0
    with Pool(8) as p:
        for i, data in p.uimap(helper, range(N_int := read_num_intervals())):
            (_H_beta_N_elc, _H_beta_B_xy, _H_beta_J_para, _H_beta_J_perp, _H_beta_J_err) = data
            H_beta_N_elc += _H_beta_N_elc
            H_beta_B_xy += _H_beta_B_xy
            H_beta_J_para += _H_beta_J_para
            H_beta_J_perp += _H_beta_J_perp
            H_beta_J_err += _H_beta_J_err
            print(f"Processed {i}/{N_int} files.")

    h5f = h5.File(lib.analysis_file, "a")
    if (where := f"/compare_B14") in h5f:
        del h5f[where]

    h5d = h5f.create_dataset(f"{where}/bg", data=bg)

    h5d = h5f.create_dataset(f"{where}/Ng", data=Ng.value)
    h5d.attrs["unit"] = str(Ng.unit)

    h5d = h5f.create_dataset(f"{where}/Bg", data=Bg.value)
    h5d.attrs["unit"] = str(Bg.unit)

    h5d = h5f.create_dataset(f"{where}/Jg", data=Jg.value)
    h5d.attrs["unit"] = str(Jg.unit)

    h5f.create_dataset(f"{where}/H_beta_N_elc", data=H_beta_N_elc)
    h5f.create_dataset(f"{where}/H_beta_B_xy", data=H_beta_B_xy)
    h5f.create_dataset(f"{where}/H_beta_J_para", data=H_beta_J_para)
    h5f.create_dataset(f"{where}/H_beta_J_perp", data=H_beta_J_perp)
    h5f.create_dataset(f"{where}/H_beta_J_err", data=H_beta_J_err)
