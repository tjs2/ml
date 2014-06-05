import numpy as np
from sklearn.preprocessing import normalize


def normalize_args(X, axis=0):
    mx = np.amax(X, axis=axis)
    mn = np.amin(X, axis=axis)
    rg = mx - mn
    rg[rg == 0] = 1.0
    return mx, mn, rg


def normalize(X, mx=None, mn=None, rg=None, axis=0):
    if mx is None or mn is None or rg is None:
        mx, mn, rg = normalize_args(X, axis=axis)

    return (X - mn) / rg
