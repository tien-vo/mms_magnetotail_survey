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
    t_ion = read_data(f"/postprocess/interval_{interval}/ion/t").astype("datetime64[ns]")
    P_ion = read_data(f"/postprocess/interval_{interval}/ion/P_scalar")
    Vi = read_data(f"/mms1/ion-fpi-moms/interval_{interval}/V_gsm")

    t_fields = read_data(f"/postprocess/interval_{interval}/barycenter/t").astype("datetime64[ns]")
    B = read_data(f"/postprocess/interval_{interval}/barycenter/B_bc")
    E = read_data(f"/postprocess/interval_{interval}/barycenter/E_bc")
    R = read_data(f"/postprocess/interval_{interval}/barycenter/R_bc").to(u.R_earth)
    R = tv.numeric.interpol(R, t_fields, t_ion)
    B = tv.numeric.interpol(B, t_fields, t_ion)
    E = tv.numeric.interpol(E, t_fields, t_ion)

    t_mec = read_data(f"/mms1/mec/interval_{interval}/t").astype("datetime64[ns]")
    dipole_tilt = read_data(f"/mms1/mec/interval_{interval}/dipole_tilt")
    dipole_tilt = tv.numeric.interpol(dipole_tilt, t_mec, t_ion)

    # Derived quantities
    Bxy = np.linalg.norm(B[:, 0:2], axis=1)
    B_mag = np.linalg.norm(B, axis=1)
    beta = (P_ion / (B_mag**2 / 2 / c.si.mu0)).decompose()
    Vi_mag = np.linalg.norm(Vi, axis=1)

    # Mask
    R_XY = np.sqrt(R[:, 0]**2 + R[:, 1]**2).value
    T_XY = np.arctan2(R[:, 1], R[:, 0]).to(u.rad).value
    mask = R_XY <= model_f(T_XY)

    X, Y, Z = R.T
    H = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins)[0]
    H_Z = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=Z[mask])[0]
    H_beta = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=beta[mask])[0]
    H_Bxy = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=Bxy[mask])[0]
    H_Bx = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=B[:, 0][mask])[0]
    H_Bz = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=B[:, 2][mask])[0]
    H_Ez = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=E[:, 2][mask])[0]
    H_Vi = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=Vi_mag[mask])[0]
    H_tilt = np.histogramdd((X[mask], Y[mask], Z[mask]), bins=bins, weights=dipole_tilt[mask])[0]
    return (H, H_Z, H_beta, H_Bx, H_Bz, H_Bxy, H_Ez, H_Vi, H_tilt)

if __name__ == "__main__":
    model = Lin10MagnetopauseModel(pressure=20, bfield=0, tilt_angle=0)
    model_f = model.shape_F(which="XY", rotation_angle=np.radians(-5))

    X_bins = np.arange(-30, 10 + 0.5, 0.5) * u.R_earth
    Y_bins = np.arange(-30, 30 + 0.5, 0.5) * u.R_earth
    Z_bins = np.arange(-10, 10 + 0.5, 0.5) * u.R_earth
    bins = (X_bins, Y_bins, Z_bins)
    Xg, Yg, Zg = np.meshgrid(X_bins[:-1], Y_bins[:-1], Z_bins[:-1], indexing="ij")

    H, H_Z, H_beta, H_Bx, H_Bz, H_Bxy, H_Ez, H_Vi, H_tilt = 0, 0, 0, 0, 0, 0, 0, 0, 0
    with Pool(8) as p:
        count = 0
        for data in p.uimap(helper, range(N_int := read_num_intervals())):
            _H, _H_Z, _H_beta, _H_Bx, _H_Bz, _H_Bxy, _H_Ez, _H_Vi, _H_tilt = data
            H += _H
            H_Z += _H_Z
            H_beta += _H_beta
            H_Bx += _H_Bx
            H_Bz += _H_Bz
            H_Bxy += _H_Bxy
            H_Ez += _H_Ez
            H_Vi += _H_Vi
            H_tilt += _H_tilt
            count += 1
            print(f"Processed {count}/{N_int} files.")

    h5f = h5.File(lib.analysis_file, "a")

    if (where := f"/XYZ_distribution") in h5f:
        del h5f[where]

    h5f.create_dataset(f"{where}/H", data=H)
    h5f.create_dataset(f"{where}/H_beta", data=H_beta)

    for key, data in dict(
        Xg=Xg, Yg=Yg, Zg=Zg, H_Z=H_Z, H_Bx=H_Bx, H_Bz=H_Bz,
        H_Bxy=H_Bxy, H_Ez=H_Ez, H_Vi=H_Vi, H_tilt=H_tilt,
    ).items():
        h5d = h5f.create_dataset(f"{where}/{key}", data=data.value)
        h5d.attrs["unit"] = str(data.unit)
