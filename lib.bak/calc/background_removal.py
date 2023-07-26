r""" TO-DO """

def remove_background(N_bg, P_bg, N, V_gse, V_gsm, P_tensor_gse, Ns, Vs_gse, Vs_gsm, Ps_tensor_gse):
    Ns = N - N_bg
    Ns[Ns < 0] = np.nan
    Vs_gsm = (N / Ns)[:, np.newaxis] * V_gsm
    Vs_gse = (N / Ns)[:, np.newaxis] * V_gse
    Ps_tensor_gse = (
        P_tensor_gse - P_bg[:, np.newaxis, np.newaxis] * np.identity(3)[np.newaxis, ...] +
        mass * N[:, np.newaxis, np.newaxis] * np.einsum("...i,...j", V_gse, V_gse) -
        mass * Ns[:, np.newaxis, np.newaxis] * np.einsum("...i,...j", Vs_gse, Vs_gse)
    ).to(u.Unit("keV cm-3"))
    Ps_tensor_gse[Ps_tensor_gse < 0] = np.nan

    return Ns, Vs_gse, Vs_gsm, Ps_tensor_gse
