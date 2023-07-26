__all__ = ["feeps"]

import astropy.units as u
import numpy as np
import tempfile
import lib
from pyspedas.mms.feeps.mms_feeps_active_eyes import mms_feeps_active_eyes
from tvolib.numeric import move_avg, sampling_period
from pyspedas.mms import mms_load_feeps, mms_config
from pytplot import get, del_data
from lib.utils import read_trange


energy_channels = dict(
    ion=np.array([
        76.8, 95.4, 114.1, 133.0, 153.7, 177.6, 205.1, 236.7, 273.2, 315.4,
        363.8, 419.7, 484.2, 558.6
    ]) * u.keV,
    elc=np.array([
        51.9, 70.6, 89.4, 107.1, 125.2, 146.5, 171.3, 200.2, 234.0, 273.4,
        319.4, 373.2, 436.0, 509.2
    ]) * u.keV,
)
energy_correction = dict(
    ion={
        "1": 0.0 * u.keV,
        "2": 0.0 * u.keV,
        "3": 0.0 * u.keV,
        "4": 0.0 * u.keV,
    },
    elc={
        "1": 14.0 * u.keV,
        "2": -1.0 * u.keV,
        "3": -3.0 * u.keV,
        "4": -3.0 * u.keV,
    },
)
geometric_factor = dict(
    ion={
        "1": 0.84,
        "2": 1.0,
        "3": 1.0,
        "4": 1.0,
    },
    elc={
        "1": 1.0,
        "2": 1.0,
        "3": 1.0,
        "4": 1.0,
    },
)


def feeps(probe, interval, drate="srvy", species="elc"):

    trange = read_trange(interval, dtype=str)
    dtype = "ion" if species == "ion" else "electron"
    pfx = f"mms{probe}_epd_feeps_{drate}_l2_{dtype}"
    sfx = "clean_sun_removed"
    active_heads = mms_feeps_active_eyes(trange, str(probe), drate, dtype,
                                         "l2")
    insts = [(head, eye) for head in active_heads
             for eye in active_heads[head]]

    # Download FEEPS files
    with tempfile.TemporaryDirectory(dir=lib.tmp_dir) as tmp_dir:
        tempfile.tempdir = tmp_dir
        #mms_config.CONFIG["local_data_dir"] = tmp_dir
        kw = dict(trange=trange,
                  probe=probe,
                  data_rate=drate,
                  datatype=dtype,
                  time_clip=True)
        for _ in range(3):
            try:
                mms_load_feeps(latest_version=True, **kw)
                break
            except OSError:
                mms_load_feeps(major_version=True, **kw)
                break

    # Preallocate
    t, _, _ = get(
        f"{pfx}_{insts[0][0]}_intensity_sensorid_{insts[0][1]}_{sfx}", dt=True)
    energy = energy_channels[species] + energy_correction[species][str(probe)]
    eflux, eflux_err = np.empty(
        (2, len(insts), len(t), len(energy))) * u.Unit("cm-2 s-1 sr-1")
    eflux[:] = np.nan
    eflux_err[:] = np.nan

    # Accumulate data from each instrument
    for i, (head, eye) in enumerate(insts):
        _, nflux, _energy = get(f"{pfx}_{head}_intensity_sensorid_{eye}_{sfx}")
        _, err, _ = get(f"{pfx}_{head}_percent_error_sensorid_{eye}")
        _energy = _energy[1:]
        # Convert number flux to energy flux
        eflux[i] = (nflux[:, 1:] * u.Unit("cm-2 s-1 sr-1 keV-1") *
                    energy[np.newaxis, np.newaxis, :])
        # Kludge for outdated cdf files
        eflux_err[i] = 0 if err is None else eflux[i] * err[:, 1:15] / 100
        # Mask out energy correction
        idx = np.where(np.abs(energy.value - _energy) > (0.1 * energy.value))
        eflux[i, :, idx] = np.nan
        eflux_err[i, :, idx] = np.nan

    # Collapse to omni-directional distribution with condition on error threshold
    cnd = (0 <= (eflux_err / eflux)) & ((eflux_err / eflux) <= 1)
    eflux_omni = (np.nanmean(eflux, axis=0, where=cnd) *
                  geometric_factor[species][str(probe)])

    # 2/3rd spin averages
    window = 2 / 3 * (19.67 * u.s / sampling_period(t)).decompose()
    eflux_omni_avg = move_avg(eflux_omni, (window, 1), window="gauss")

    del_data()
    return dict(t=t.astype("f8"),
                f_omni_energy=energy,
                f_omni=eflux_omni,
                f_omni_avg=eflux_omni_avg)


if __name__ == "__main__":
    from tvolib import mpl_utils as mu
    import matplotlib.pyplot as plt

    i = 418
    ion_data = feeps(1, i, species="ion")
    t_ion, energy_ion = np.meshgrid(ion_data["t"].astype("datetime64[ns]"),
                                    ion_data["f_omni_energy"],
                                    indexing="ij")
    elc_data = feeps(1, i, species="elc")
    t_elc, energy_elc = np.meshgrid(elc_data["t"].astype("datetime64[ns]"),
                                    elc_data["f_omni_energy"],
                                    indexing="ij")

    fig, axes = mu.plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    kw = dict(cmap="jet", norm=mu.mplc.LogNorm())

    cax = mu.add_colorbar(axes[0])
    im = axes[0].pcolormesh(t_ion, energy_ion.value, ion_data["f_omni"].value,
                            **kw)
    fig.colorbar(im, cax=cax)
    axes[0].set_ylim(energy_ion.value.min(), energy_ion.value.max())

    cax = mu.add_colorbar(axes[1])
    im = axes[1].pcolormesh(t_ion, energy_ion.value,
                            ion_data["f_omni_avg"].value, **kw)
    fig.colorbar(im, cax=cax)
    axes[1].set_ylim(energy_ion.value.min(), energy_ion.value.max())

    cax = mu.add_colorbar(axes[2])
    im = axes[2].pcolormesh(t_elc, energy_elc.value, elc_data["f_omni"].value,
                            **kw)
    fig.colorbar(im, cax=cax)
    axes[2].set_ylim(energy_elc.value.min(), energy_elc.value.max())

    cax = mu.add_colorbar(axes[3])
    im = axes[3].pcolormesh(t_elc, energy_elc.value,
                            elc_data["f_omni_avg"].value, **kw)
    fig.colorbar(im, cax=cax)
    axes[3].set_ylim(energy_elc.value.min(), energy_elc.value.max())

    for (i, ax) in enumerate(axes):
        ax.set_ylabel(f"{ion_data['f_omni_energy'].unit:latex_inline}")
        ax.set_yscale("log")
        ax.set_xlim(np.datetime64("2017-07-26T07:28"),
                    np.datetime64("2017-07-26T07:33"))

    fig.tight_layout(h_pad=0.05)
    mu.plt.show()
