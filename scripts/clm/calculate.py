import astropy.constants as c
import astropy.units as u
import h5py as h5
import numpy as np
from pathos.pools import ProcessPool as Pool
from tvolib.numeric import interpol, curlometer

import lib
from lib.utils import read_data, read_num_intervals


def calculate(interval):
    t1 = read_data(f"mms1/fgm/interval_{interval}/t").astype("datetime64[ns]")
    B1 = read_data(f"mms1/fgm/interval_{interval}/B_gsm")
    R1 = read_data(f"mms1/fgm/interval_{interval}/R_gsm")
    t2 = read_data(f"mms2/fgm/interval_{interval}/t").astype("datetime64[ns]")
    B2 = read_data(f"mms2/fgm/interval_{interval}/B_gsm")
    R2 = read_data(f"mms2/fgm/interval_{interval}/R_gsm")
    t3 = read_data(f"mms3/fgm/interval_{interval}/t").astype("datetime64[ns]")
    B3 = read_data(f"mms3/fgm/interval_{interval}/B_gsm")
    R3 = read_data(f"mms3/fgm/interval_{interval}/R_gsm")
    t4 = read_data(f"mms4/fgm/interval_{interval}/t").astype("datetime64[ns]")
    B4 = read_data(f"mms4/fgm/interval_{interval}/B_gsm")
    R4 = read_data(f"mms4/fgm/interval_{interval}/R_gsm")

    B2 = interpol(B2, t2, t1)
    R2 = interpol(R2, t2, t1)
    B3 = interpol(B3, t3, t1)
    R3 = interpol(R3, t3, t1)
    B4 = interpol(B4, t4, t1)
    R4 = interpol(R4, t4, t1)
    B_bc = 0.25 * (B1 + B2 + B3 + B4)
    R_bc = 0.25 * (R1 + R2 + R3 + R4)

    _t1 = read_data(f"mms1/edp/interval_{interval}/t").astype("datetime64[ns]")
    E1 = read_data(f"mms1/edp/interval_{interval}/E_gsm")
    _t2 = read_data(f"mms2/edp/interval_{interval}/t").astype("datetime64[ns]")
    E2 = read_data(f"mms2/edp/interval_{interval}/E_gsm")
    _t3 = read_data(f"mms3/edp/interval_{interval}/t").astype("datetime64[ns]")
    E3 = read_data(f"mms3/edp/interval_{interval}/E_gsm")
    _t4 = read_data(f"mms4/edp/interval_{interval}/t").astype("datetime64[ns]")
    E4 = read_data(f"mms4/edp/interval_{interval}/E_gsm")
    E1 = interpol(E1, _t1, t1)
    E2 = interpol(E2, _t2, t1)
    E3 = interpol(E3, _t3, t1)
    E4 = interpol(E4, _t4, t1)
    E_bc = 0.25 * (E1 + E2 + E3 + E4)

    clm_data = curlometer(B1, B2, B3, B4, R1, R2, R3, R4)
    J_clm = (clm_data["curl_Q"] / c.si.mu0).to(u.Unit("nA m-2"))

    J_para = np.einsum("...i,...i", J_clm, B_bc) / np.linalg.norm(
        B_bc, axis=-1
    )
    E_para = np.einsum("...i,...i", E_bc, B_bc) / np.linalg.norm(B_bc, axis=-1)
    JdE = np.einsum("...i,...i", J_clm, E_bc).to(u.Unit("nW m-3"))
    JdE_para = (J_para * E_para).to(u.Unit("nW m-3"))
    JdE_perp = JdE - JdE_para

    h5f = h5.File(lib.postprocess_dir / f"interval_{interval}.h5", "a")

    if (where := f"/barycenter") in h5f:
        del h5f[where]

    h5f.create_dataset(f"{where}/t", data=t1.astype("f8"))
    for name, var in dict(
        R_bc=R_bc,
        B_bc=B_bc,
        E_bc=E_bc,
        J_clm=J_clm,
        J_para=J_para,
        E_para=E_para,
    ).items():
        h5d = h5f.create_dataset(f"{where}/{name}", data=var.value)
        h5d.attrs["unit"] = str(var.unit)

    print(f"Calculated barycentric quantities for interval {interval}")


if __name__ == "__main__":
    with Pool() as p:
        for _ in p.uimap(calculate, range(1100, read_num_intervals())):
            pass
