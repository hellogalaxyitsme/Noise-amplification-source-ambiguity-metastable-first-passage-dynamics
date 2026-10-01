"""Maximum-likelihood fits of the leaky stochastic accumulator to waiting times.

Model (the accumulator of Schurger et al., 2012, in gradient form):

    dx = (I - k x) dt + sqrt(2 D) dW,   x(0) = 0,   commitment at x = 1,
    reflecting wall at ell = -3,        U(x) = k x^2 / 2 - I x.

``I/k > 1`` places the deterministic fixed point beyond the threshold
(drift-dominated regime); ``I/k < 1`` makes commitment noise-driven. A
non-decision shift ``t_nd`` in ``[0, 0.98 min(T))`` is fitted.

The waiting-time density is that of the Markov-chain discretisation
(:mod:`metastable_fpt.timing_law`), so the likelihood carries no sampling error.
The elasticity of a fitted model follows from Lemma 4
(:func:`metastable_fpt.elasticity.weighted_barrier`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .elasticity import weighted_barrier
from .generators import Domain, white_generator
from .timing_law import TimeGrid, spectrum, timing_law

ELL = -3.0
THRESHOLD = 1.0


def density(I: float, k: float, D: float, t: np.ndarray, *, n_x: int = 160, dt0: float = 2e-3) -> np.ndarray:
    """Commitment-time density of the discretised accumulator at times ``t``."""

    dom = Domain(ELL, THRESHOLD, n_x)
    g = white_generator(lambda x: k * np.asarray(x) - I, D=D, x0=0.0, dom=dom)
    t_end = float(np.max(t)) * 1.02 + 0.05
    law = timing_law(g, TimeGrid(dt0=dt0, steps=60, t_end=t_end), spec=spectrum(g))
    logf = np.log(np.maximum(law.f, 1e-300))
    return np.exp(np.interp(t, law.t, logf))


def _unpack(z: np.ndarray, wmin: float) -> tuple[float, float, float, float]:
    I, k, D = np.exp(z[:3])
    shift = wmin * 0.98 / (1.0 + np.exp(-z[3]))
    return float(I), float(k), float(D), float(shift)


def negloglik(z: np.ndarray, w: np.ndarray, n_x: int = 160) -> float:
    """Negative log-likelihood in the unconstrained parametrisation ``(log I, log k, log D, logit shift)``."""

    I, k, D, shift = _unpack(z, float(w.min()))
    if not (1e-3 < I < 1e3 and 1e-4 < k < 1e3 and 1e-4 < D < 1e3):
        return 1e12
    try:
        f = density(I, k, D, w - shift, n_x=n_x)
    except Exception:
        return 1e12
    return float(-np.sum(np.log(np.maximum(f, 1e-300))))


@dataclass
class AccumulatorFit:
    I: float
    k: float
    D: float
    shift: float
    nll: float
    fixed_point_over_threshold: float
    elasticity: float
    mean_model: float
    cv_model: float
    converged: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _elasticity(I: float, k: float, D: float):
    return weighted_barrier(lambda x: k * np.asarray(x) ** 2 / 2 - I * np.asarray(x),
                            lambda x: k * np.asarray(x) - I, D=D, x0=0.0, ell=ELL, b=THRESHOLD,
                            n=20001, check_fd=False)


def fit_accumulator(w: np.ndarray, *, starts: list[np.ndarray] | None = None, n_x: int = 160,
                    max_iter: int = 600) -> AccumulatorFit:
    """Best of several Nelder-Mead runs from fixed starting points."""

    w = np.asarray(w, dtype=float)
    m, v = float(w.mean()), float(w.var())
    if starts is None:
        v0 = 1.0 / m
        D0 = max(v * v0**3 / 2.0, 1e-3)
        starts = [
            np.array([np.log(v0 * 1.2), np.log(0.05), np.log(D0), -2.0]),
            np.array([np.log(1.0), np.log(1.2), np.log(max(D0, 0.05)), -2.0]),
            np.array([np.log(0.5), np.log(0.8), np.log(0.1), 0.0]),
        ]
    best = None
    for z0 in starts:
        res = minimize(negloglik, z0, args=(w, n_x), method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-5})
        if best is None or res.fun < best.fun:
            best = res
    I, k, D, shift = _unpack(best.x, float(w.min()))
    el = _elasticity(I, k, D)
    return AccumulatorFit(I, k, D, shift, float(best.fun), float(I / k) / THRESHOLD, el.elasticity,
                          el.mean + shift, float(np.sqrt(el.var) / (el.mean + shift)), bool(best.success))


def profile_fixed_point(w: np.ndarray, rho_grid, *, n_x: int = 160, max_iter: int = 500,
                        start: AccumulatorFit | None = None) -> list[dict]:
    """Constrained fits with the fixed-point ratio ``rho = I/k`` held at each grid value.

    Returns, for each ``rho``, the minimised negative log-likelihood, the fitted
    parameters and the elasticity of the constrained optimum. ``rho < 1`` is the
    noise-driven regime.
    """

    w = np.asarray(w, dtype=float)
    wmin = float(w.min())
    out = []
    for rho in rho_grid:
        def nll(y, rho=rho):
            k, D = np.exp(y[:2])
            z = np.array([np.log(rho * k), np.log(k), np.log(D), y[2]])
            return negloglik(z, w, n_x)

        starts = []
        if start is not None:
            frac = min(max(start.shift / (0.98 * wmin), 1e-6), 1 - 1e-6)
            starts.append(np.array([np.log(start.k), np.log(start.D), np.log(frac / (1 - frac))]))
        m = float(w.mean())
        starts += [np.array([np.log(0.3 / m), np.log(0.05), 0.0]), np.array([np.log(1.0), np.log(0.1), 0.0]),
                   np.array([np.log(3.0), np.log(0.3), -1.0])]
        best = None
        for y0 in starts:
            res = minimize(nll, y0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-5})
            if best is None or res.fun < best.fun:
                best = res
        k, D = np.exp(best.x[:2])
        I = rho * k
        try:
            el = _elasticity(float(I), float(k), float(D))
            s = el.elasticity
        except Exception:
            s = float("nan")
        out.append({"rho": float(rho), "nll": float(best.fun), "I": float(I), "k": float(k), "D": float(D),
                    "shift": float(wmin * 0.98 / (1 + np.exp(-best.x[2]))), "elasticity": float(s),
                    "noise_to_barrier": float((1 - rho) ** 2 * k / (2 * D)) if rho < 1 else 0.0})
    return out
