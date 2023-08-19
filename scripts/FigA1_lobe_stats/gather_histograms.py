import h5py as h5
import numpy as np
import tvolib as tv
import astropy.units as u

import lib
from lib.utils import read_data


def get_combined_dist(interval, species="ion"):
    t_fpi = read_data(f"mms1/{species}-fpi-moms/interval_{interval}/t").astype(
        "datetime64[ns]"
    )
    f_fpi = read_data(f"mms1/{species}-fpi-moms/interval_{interval}/f_omni")
    W_fpi = read_data(
        f"mms1/{species}-fpi-moms/interval_{interval}/f_omni_energy"
    )
    Vsc = read_data(f"mms1/{species}-fpi-moms/interval_{interval}/Vsc")

    t_feeps = read_data(f"mms1/{species}-feeps/interval_{interval}/t").astype(
        "datetime64[ns]"
    )
    f_feeps = read_data(f"mms1/{species}-feeps/interval_{interval}/f_omni")
    W_feeps = np.tile(
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
    assert W_fpi.unit == W_feeps.unit

    # Concatenate distribution functions and integrate for scalar moments
    W = np.concatenate((W_fpi, W_feeps), axis=1)
    f = np.concatenate((f_fpi, f_feeps), axis=1)
    t = t_fpi
    tg = np.tile(t, (W.shape[1], 1)).T
    return tg, W, f


intervals = {
    "267": np.array(["2017-07-04T04:50", "2017-07-04T05:20"], dtype="datetime64[ns]"),
    "288": np.array(["2017-07-07T02:30", "2017-07-07T03:20"], dtype="datetime64[ns]"),
    "300": np.array(["2017-07-10T07:20", "2017-07-10T09:20"], dtype="datetime64[ns]"),
    "327": np.array(["2017-07-13T09:30", "2017-07-13T10:10"], dtype="datetime64[ns]"),
    "352": np.array(["2017-07-15T15:20", "2017-07-15T16:40"], dtype="datetime64[ns]"),
}

W_bins = np.logspace(-3, 3, 50) * u.keV
f_bins = np.logspace(-1, 9, 80) * u.Unit("cm-2 s-1 sr-1")
Wg, fg = np.meshgrid(W_bins[:-1], f_bins[:-1], indexing="ij")

H_ion, H_elc = 0, 0
for i in intervals.keys():

    t_ion, W_ion, f_ion = get_combined_dist(int(i), species="ion")
    idx = np.where((intervals[i][0] <= t_ion[:, 0]) & (t_ion[:, 0] <= intervals[i][1]))
    W_ion = W_ion[idx, :]
    f_ion = f_ion[idx, :]

    t_elc, W_elc, f_elc = get_combined_dist(int(i), species="elc")
    idx = np.where((intervals[i][0] <= t_elc[:, 0]) & (t_elc[:, 0] <= intervals[i][1]))
    W_elc = W_elc[idx, :]
    f_elc = f_elc[idx, :]

    H_ion += np.histogram2d(W_ion.ravel(), f_ion.ravel(), bins=(W_bins, f_bins))[0]
    H_elc += np.histogram2d(W_elc.ravel(), f_elc.ravel(), bins=(W_bins, f_bins))[0]
    print(f"Processed {i}")

h5f = h5.File(lib.analysis_file, "a")

if (where := f"/lobe_stats") in h5f:
    del h5f[where]

h5f.create_dataset(f"{where}/H_ion", data=H_ion)
h5f.create_dataset(f"{where}/H_elc", data=H_elc)

for key, data in dict(Wg=Wg, fg=fg).items():
    h5d = h5f.create_dataset(f"{where}/{key}", data=data.value)
    h5d.attrs["unit"] = str(data.unit)
