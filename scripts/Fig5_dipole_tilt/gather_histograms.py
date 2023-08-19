import h5py as h5
import numpy as np
import tvolib as tv
import astropy.units as u

from pathos.pools import ProcessPool as Pool
from tvolib.models.magnetopause_model import Lin10MagnetopauseModel

import lib
from lib.utils import read_data, read_num_intervals


def helper(interval):
    t_mec = read_data(f"/mms1/mec/interval_{interval}/t").astype("datetime64[ns]")
    doy = t_mec.astype("datetime64[D]") - t_mec.astype("datetime64[Y]") + 1
    dipole_tilt = read_data(f"/mms1/mec/interval_{interval}/dipole_tilt")

    t_fields = read_data(f"/postprocess/interval_{interval}/barycenter/t").astype("datetime64[ns]")
    R = read_data(f"/postprocess/interval_{interval}/barycenter/R_bc").to(u.R_earth)
    R = tv.numeric.interpol(R, t_fields, t_mec)

    # Mask
    R_XY = np.sqrt(R[:, 0]**2 + R[:, 1]**2).value
    T_XY = np.arctan2(R[:, 1], R[:, 0]).to(u.rad).value
    mask = R_XY <= model_f(T_XY)

    H_doy_tilt = np.histogram2d(doy[mask].astype("i8"), dipole_tilt[mask], bins=(doy_bins, tilt_bins))[0]
    H_Y_tilt = np.histogram2d(R[:, 1][mask], dipole_tilt[mask], bins=(Y_bins, tilt_bins))[0]
    return H_doy_tilt, H_Y_tilt


if __name__ == "__main__":
    model = Lin10MagnetopauseModel(pressure=20, bfield=0, tilt_angle=0)
    model_f = model.shape_F(which="XY", rotation_angle=np.radians(-5))

    doy_bins = np.arange(100, 301, 2)
    Y_bins = np.arange(-20, 21, 0.5) * u.R_earth
    tilt_bins = np.arange(-20, 41, 1) * u.deg

    doy, tilt_doy = np.meshgrid(doy_bins[:-1], tilt_bins[:-1], indexing="ij")
    Y, tilt_Y = np.meshgrid(Y_bins[:-1], tilt_bins[:-1], indexing="ij")

    H_doy_tilt, H_Y_tilt = 0, 0
    with Pool(8) as p:
        count = 0
        for (_H_doy_tilt, _H_Y_tilt) in p.uimap(helper, range(N_int := read_num_intervals())):
            H_doy_tilt += _H_doy_tilt
            H_Y_tilt += _H_Y_tilt
            count += 1
            print(f"Processed {count}/{N_int} files.")

    h5f = h5.File(lib.analysis_file, "a")
    if (where := f"/dipole_tilt") in h5f:
        del h5f[where]

    h5f.create_dataset(f"{where}/H_doy_tilt", data=H_doy_tilt)
    h5f.create_dataset(f"{where}/H_Y_tilt", data=H_Y_tilt)
    h5f.create_dataset(f"{where}/doy", data=doy)

    h5d = h5f.create_dataset(f"{where}/Y", data=Y.value)
    h5d.attrs["unit"] = str(Y.unit)

    h5d = h5f.create_dataset(f"{where}/tilt_doy", data=tilt_doy.value)
    h5d.attrs["unit"] = str(tilt_doy.unit)

    h5d = h5f.create_dataset(f"{where}/tilt_Y", data=tilt_Y.value)
    h5d.attrs["unit"] = str(tilt_Y.unit)
