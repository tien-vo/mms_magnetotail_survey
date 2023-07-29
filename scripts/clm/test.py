import astropy.constants as c
import astropy.units as u
import numpy as np
import pyspedas as sp
from pyspedas.mms.fgm.mms_curl import mms_curl
from pytplot import get
from tvolib import mpl_utils as mu
from tvolib.numeric import curlometer, interpol

from lib.utils import read_data

i = 418
# trange = np.array(["2017-07-26T07:28:00", "2017-07-26T07:29:30"], dtype="datetime64[ns]")
trange = np.array(
    ["2017-07-26T07:25:00", "2017-07-26T07:35:00"], dtype="datetime64[ns]"
)

t1 = read_data(f"mms1/fgm/interval_{i}/t").astype("datetime64[ns]")
B1 = read_data(f"mms1/fgm/interval_{i}/B_gsm")
R1 = read_data(f"mms1/fgm/interval_{i}/R_gsm")
t2 = read_data(f"mms2/fgm/interval_{i}/t").astype("datetime64[ns]")
B2 = read_data(f"mms2/fgm/interval_{i}/B_gsm")
R2 = read_data(f"mms2/fgm/interval_{i}/R_gsm")
t3 = read_data(f"mms3/fgm/interval_{i}/t").astype("datetime64[ns]")
B3 = read_data(f"mms3/fgm/interval_{i}/B_gsm")
R3 = read_data(f"mms3/fgm/interval_{i}/R_gsm")
t4 = read_data(f"mms4/fgm/interval_{i}/t").astype("datetime64[ns]")
B4 = read_data(f"mms4/fgm/interval_{i}/B_gsm")
R4 = read_data(f"mms4/fgm/interval_{i}/R_gsm")

B2 = interpol(B2, t2, t1)
R2 = interpol(R2, t2, t1)
B3 = interpol(B3, t3, t1)
R3 = interpol(R3, t3, t1)
B4 = interpol(B4, t4, t1)
R4 = interpol(R4, t4, t1)
B_bc = 0.25 * (B1 + B2 + B3 + B4)

_t1 = read_data(f"mms1/edp/interval_{i}/t").astype("datetime64[ns]")
E1 = read_data(f"mms1/edp/interval_{i}/E_gsm")
_t2 = read_data(f"mms2/edp/interval_{i}/t").astype("datetime64[ns]")
E2 = read_data(f"mms2/edp/interval_{i}/E_gsm")
_t3 = read_data(f"mms3/edp/interval_{i}/t").astype("datetime64[ns]")
E3 = read_data(f"mms3/edp/interval_{i}/E_gsm")
_t4 = read_data(f"mms4/edp/interval_{i}/t").astype("datetime64[ns]")
E4 = read_data(f"mms4/edp/interval_{i}/E_gsm")
E1 = interpol(E1, _t1, t1)
E2 = interpol(E2, _t2, t1)
E3 = interpol(E3, _t3, t1)
E4 = interpol(E4, _t4, t1)
E_bc = 0.25 * (E1 + E2 + E3 + E4)

clm_data = curlometer(B1, B2, B3, B4, R1, R2, R3, R4)
J_clm = (clm_data["curl_Q"] / c.si.mu0).to(u.Unit("nA m-2"))

J_para = np.einsum("...i,...i", J_clm, B_bc) / np.linalg.norm(B_bc, axis=-1)
E_para = np.einsum("...i,...i", E_bc, B_bc) / np.linalg.norm(B_bc, axis=-1)
JdE = np.einsum("...i,...i", J_clm, E_bc).to(u.Unit("nW m-3"))
JdE_para = (J_para * E_para).to(u.Unit("nW m-3"))
JdE_perp = JdE - JdE_para


sp.mms.fgm(
    trange=trange.astype(str),
    probe=[1, 2, 3, 4],
    data_rate="srvy",
    time_clip=True,
    get_fgm_ephemeris=True,
)

fields = [f"mms{i}_fgm_b_gsm_srvy_l2" for i in range(1, 5)]
positions = [f"mms{i}_fgm_r_gsm_srvy_l2" for i in range(1, 5)]
mms_curl(fields=fields, positions=positions)

t_ref, J_ref = get("curlB", dt=True, units=True)
J_ref = (J_ref * u.Unit("nT/km") / c.si.mu0).to(u.Unit("nA m-2"))

fig, axes = mu.plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(t1, E_bc[:, 0], "-b")
axes[0].plot(t1, E_bc[:, 1], "-g")
axes[0].plot(t1, E_bc[:, 2], "-r")

axes[1].plot(t1, J_clm[:, 0], "-b")
axes[1].plot(t1, J_clm[:, 1], "-g")
axes[1].plot(t1, J_clm[:, 2], "-r")

axes[2].plot(t1, JdE_perp, "-b")
axes[2].plot(t1, JdE_para, "-r")

for i, ax in enumerate(axes):
    # ax.set_xlim(*trange)
    ax.set_xlim(
        *np.array(
            ["2017-07-26T07:28:00", "2017-07-26T07:29:30"],
            dtype="datetime64[ns]",
        )
    )
    mu.format_datetime_axis(ax)

fig.tight_layout(h_pad=0.05)
fig.savefig("test_event.png")

fig, axes = mu.plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(t1, J_clm[:, 0], "-k", lw=2)
axes[0].plot(t_ref, J_ref[:, 0], "--r", lw=2)

axes[1].plot(t1, J_clm[:, 1], "-k", lw=2)
axes[1].plot(t_ref, J_ref[:, 1], "--r", lw=2)

axes[2].plot(t1, J_clm[:, 2], "-k", lw=2)
axes[2].plot(t_ref, J_ref[:, 2], "--r", lw=2)

for i, ax in enumerate(axes):
    # ax.set_xlim(*trange)
    ax.set_xlim(
        *np.array(
            ["2017-07-26T07:28:00", "2017-07-26T07:29:30"],
            dtype="datetime64[ns]",
        )
    )
    mu.format_datetime_axis(ax)

fig.tight_layout(h_pad=0.05)
fig.savefig("test_compare.png")
