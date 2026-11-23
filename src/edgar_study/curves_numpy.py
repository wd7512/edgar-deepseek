"""Numpy evaluation and Jacobians for the tuning curves in `edgar_study.curves`.

Reference implementations for the backend benchmark: every framework fits the same equations, so
differences in the benchmark come from the optimiser and runtime, not the model. The Jacobians are
analytic; `tests/unit/test_curves_numpy.py` checks them against JAX's forward-mode autodiff.
"""

import numpy as np

SIGMA_RANGE = (0.05, 3.0)
EXPONENT_RANGE = (0.5, 8.0)

PARAM_NAMES = {
    "double-gaussian": ("phi", "b", "a1", "a2", "s"),
    "winner": (
        "phi1",
        "phi2",
        "b",
        "a1",
        "a2",
        "s1p",
        "s1m",
        "s2p",
        "s2m",
        "p1",
        "p2",
    ),
}


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _peak(x, phi, amplitude, sigma_plus, sigma_minus, exponent):
    d = _wrap(x - phi)
    sigma = np.where(
        d >= 0,
        np.clip(sigma_plus, *SIGMA_RANGE),
        np.clip(sigma_minus, *SIGMA_RANGE),
    )
    p = np.clip(exponent, *EXPONENT_RANGE)
    absolute = np.abs(d)
    u = absolute / sigma
    q = u**p
    e = np.exp(-0.5 * q)
    value = np.clip(amplitude, 0, None) * e
    return value, d, e, q, u, sigma


def _peak_derivatives(amplitude, d, e, q, u, sigma, p):
    """d(value)/d(phi), /d(amplitude), /d(sigma_plus), /d(sigma_minus), /d(exponent)."""
    gain = np.clip(amplitude, 0, None)
    active = (amplitude > 0).astype(e.dtype)
    absolute = np.abs(d)
    dphi = 0.5 * gain * e * p * np.sign(d) * q / (absolute + 1e-12)
    dsigma = 0.5 * gain * e * q * p / sigma
    ds_plus = np.where(d >= 0, dsigma, 0.0)
    ds_minus = np.where(d < 0, dsigma, 0.0)
    dp = -0.5 * gain * e * q * np.log(u + 1e-12)
    return dphi, e * active, ds_plus, ds_minus, dp


def model(name, x, params):
    """Predicted response, shape (cells, trials), for a curve and (cells, params) parameters."""
    x = np.asarray(x)
    params = np.asarray(params)
    if name == "double-gaussian":
        phi = params[:, 0:1]
        b = params[:, 1:2]
        a1 = params[:, 2:3]
        a2 = params[:, 3:4]
        s = params[:, 4:5]
        first = _peak(x, phi, a1, s, s, 2.0)[0]
        second = _peak(x, phi + np.pi, a2, s, s, 2.0)[0]
        return np.clip(b, 0, None) + first + second
    if name == "winner":
        phi1, phi2 = params[:, 0:1], params[:, 1:2]
        b = params[:, 2:3]
        a1, a2 = params[:, 3:4], params[:, 4:5]
        s1p, s1m, s2p, s2m = params[:, 5:6], params[:, 6:7], params[:, 7:8], params[:, 8:9]
        p1, p2 = params[:, 9:10], params[:, 10:11]
        first = _peak(x, phi1, a1, s1p, s1m, p1)[0]
        second = _peak(x, phi2, a2, s2p, s2m, p2)[0]
        return np.clip(b, 0, None) + first + second
    raise ValueError(f"Unknown curve {name!r}.")


def jacobian(name, x, params):
    """d(model)/d(params), shape (cells, trials, params)."""
    x = np.asarray(x)
    params = np.asarray(params)
    cells, trials = x.shape
    out = np.zeros((cells, trials, params.shape[1]), dtype=np.result_type(x, params))
    if name == "double-gaussian":
        phi = params[:, 0:1]
        a1, a2, s = params[:, 2:3], params[:, 3:4], params[:, 4:5]
        first = _peak(x, phi, a1, s, s, 2.0)
        second = _peak(x, phi + np.pi, a2, s, s, 2.0)
        d1 = _peak_derivatives(a1, first[1], first[2], first[3], first[4], first[5], 2.0)
        d2 = _peak_derivatives(a2, second[1], second[2], second[3], second[4], second[5], 2.0)
        out[..., 0] = d1[0] + d2[0]
        out[..., 1] = 1.0
        out[..., 2] = d1[1]
        out[..., 3] = d2[1]
        out[..., 4] = d1[2] + d1[3] + d2[2] + d2[3]
        return out
    if name == "winner":
        phi1, phi2 = params[:, 0:1], params[:, 1:2]
        a1, a2 = params[:, 3:4], params[:, 4:5]
        s1p, s1m = params[:, 5:6], params[:, 6:7]
        s2p, s2m = params[:, 7:8], params[:, 8:9]
        p1, p2 = params[:, 9:10], params[:, 10:11]
        first = _peak(x, phi1, a1, s1p, s1m, p1)
        second = _peak(x, phi2, a2, s2p, s2m, p2)
        d1 = _peak_derivatives(a1, first[1], first[2], first[3], first[4], first[5], p1)
        d2 = _peak_derivatives(a2, second[1], second[2], second[3], second[4], second[5], p2)
        out[..., 0] = d1[0]
        out[..., 1] = d2[0]
        out[..., 2] = 1.0
        out[..., 3] = d1[1]
        out[..., 4] = d2[1]
        out[..., 5] = d1[2]
        out[..., 6] = d1[3]
        out[..., 7] = d2[2]
        out[..., 8] = d2[3]
        out[..., 9] = d1[4]
        out[..., 10] = d2[4]
        return out
    raise ValueError(f"Unknown curve {name!r}.")
