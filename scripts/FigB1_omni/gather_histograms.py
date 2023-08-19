import h5py as h5
import numpy as np
import tvolib as tv
import astropy.units as u

from pathos.pools import ProcessPool as Pool
from tvolib.models.magnetopause_model import Lin10MagnetopauseModel

import lib
from lib.utils import read_data, read_num_intervals


def helper(interval):
    t_omni = read_data(f"/omni/interval_{interval}/t").astype("datetime64[ns]")
    P = read_data(f"/omni/interval_{interval}/Pp")
    By = read_data(f"/omni/interval_{interval}/By_gsm")
    Bz = read_data(f"/omni/interval_{interval}/Bz_gsm")

    t_fields = read_data(f"/postprocess/interval_{interval}/barycenter/t").astype("datetime64[ns]")
    R = read_data(f"/postprocess/interval_{interval}/barycenter/R_bc").to(u.R_earth)
    R = tv.numeric.interpol(R, t_fields, t_omni)

    # Mask
    R_XY = np.sqrt(R[:, 0]**2 + R[:, 1]**2).value
    T_XY = np.arctan2(R[:, 1], R[:, 0]).to(u.rad).value
    mask = R_XY <= model_f(T_XY)

    H_P_By = np.histogram2d(P, By, bins=(P_bins, B_bins))[0]
    H_P_Bz = np.histogram2d(P, Bz, bins=(P_bins, B_bins))[0]
    return (H_P_By, H_P_Bz)


if __name__ == "__main__":
    model = Lin10MagnetopauseModel(pressure=20, bfield=0, tilt_angle=0)
    model_f = model.shape_F(which="XY", rotation_angle=np.radians(-5))

    P_bins = np.linspace(0, 5, 51) * u.nPa
    B_bins = np.linspace(-10, 10, 51) * u.nT
    Pg, Bg = np.meshgrid(P_bins[:-1], B_bins[:-1], indexing="ij")

    H_P_By, H_P_Bz = 0, 0
    with Pool(8) as p:
        count = 0
        for data in p.uimap(helper, range(N_int := read_num_intervals())):
            (_H_P_By, _H_P_Bz) = data
            H_P_By += _H_P_By
            H_P_Bz += _H_P_Bz
            count += 1
            print(f"Processed {count}/{N_int} files.")

    h5f = h5.File(lib.analysis_file, "a")
    if (where := f"/omni_stats") in h5f:
        del h5f[where]

    h5d = h5f.create_dataset(f"{where}/H_P_By", data=H_P_By)
    h5d = h5f.create_dataset(f"{where}/H_P_Bz", data=H_P_Bz)

    for key, data in dict(Pg=Pg, Bg=Bg).items():
        h5d = h5f.create_dataset(f"{where}/{key}", data=data.value)
        h5d.attrs["unit"] = str(data.unit)
