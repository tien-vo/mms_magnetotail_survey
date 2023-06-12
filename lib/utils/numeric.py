from scipy.interpolate import interp1d
from bottleneck import move_mean
import numpy as np

__all__ = ["interpol", "curlometer"]


def interpol(y, x, xout, kind="linear", avg=False):

    kw = dict(kind=kind, bounds_error=False, fill_value=np.nan)
    kw_avg = dict(window=max(int(np.diff(xout).mean() / np.diff(x).mean()), 1), min_count=1)

    if len(y.shape) == 1:
        func = interp1d(x, move_mean(y, **kw_avg) if avg else y, **kw)
        yout = func(xout)
    elif len(y.shape) == 2:
        yout = np.empty(len(xout), y.shape[1])
        for i in range(y.shape[1]):
            func = interp1d(x, move_mean(y[:, i], **kw_avg) if avg else y[:, i], **kw)
            yout[:, i] = func(xout)
    else:
        raise NotImplementedError

    return yout


def curlometer(Q1, Q2, Q3, Q4, R1, R2, R3, R4):

    dR_12 = R2 - R1
    dR_13 = R3 - R1
    dR_14 = R4 - R1

    k2 = (tmp := np.cross(dR_13, dR_14)) / np.einsum("...i,...i", dR_12, tmp)[:, np.newaxis]
    k3 = (tmp := np.cross(dR_12, dR_14)) / np.einsum("...i,...i", dR_13, tmp)[:, np.newaxis]
    k4 = (tmp := np.cross(dR_12, dR_13)) / np.einsum("...i,...i", dR_14, tmp)[:, np.newaxis]
    k1 = - k2 - k3 - k4

    div = np.sum(k1 * Q1 + k2 * Q2 + k3 * Q3 + k4 * Q4, axis=-1)
    curl = np.cross(k1, Q1) + np.cross(k2, Q2) + np.cross(k3, Q3) + np.cross(k4, Q4)
    return dict(div=div, curl=curl)
