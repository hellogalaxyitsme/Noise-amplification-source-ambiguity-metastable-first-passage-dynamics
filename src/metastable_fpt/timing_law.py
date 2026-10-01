"""Commitment-time laws of killed Markov generators and the divergence between them.

For a :class:`~metastable_fpt.generators.KilledGenerator` the commitment time
``T`` is phase-type:

    S(t) = mu0^T exp(t Q) 1,    f(t) = mu0^T exp(t Q) kill,    h(t) = f(t) / S(t).

This module computes ``S``, ``f`` and ``h`` on a deterministic geometric time grid
with the L-stable TR-BDF2 scheme, the leading spectrum of ``-Q`` (escape rate
``lambda_1`` and spectral gap), the first two moments by linear solves, and the
Kullback-Leibler divergence between two timing laws through the hazard identity
(Lemma 1 of the article):

    KL(P_A || P_B) = int_0^inf S_A(t) phi(h_A(t), h_B(t)) dt,
    phi(a, b) = a log(a/b) - a + b.

Beyond the final grid time the hazards are constant to a verified tolerance and
the tail is integrated in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

from .generators import KilledGenerator
from .seeding import DomainError

Array = np.ndarray
_TINY = 1e-300


@dataclass(frozen=True)
class TimeGrid:
    """Geometric stages: ``steps`` steps of ``dt0 * 2**k`` in stage ``k``, up to ``t_end``."""

    dt0: float = 1e-3
    steps: int = 100
    t_end: float = 50.0

    def times(self) -> tuple[Array, Array]:
        ts = [0.0]
        dts = []
        k = 0
        while ts[-1] < self.t_end:
            dt = self.dt0 * 2.0**k
            for _ in range(self.steps):
                ts.append(ts[-1] + dt)
                dts.append(dt)
                if ts[-1] >= self.t_end:
                    break
            k += 1
        return np.asarray(ts), np.asarray(dts)


@dataclass
class Spectrum:
    lam1: float
    lam2: complex
    gap: float
    eigenvalues: Array


@dataclass
class TimingLaw:
    label: str
    t: Array
    S: Array
    f: Array
    h: Array
    spectrum: Spectrum
    mean: float
    var: float
    tail_hazard_rel_err: float
    mass_defect: float
    meta: dict = field(default_factory=dict)

    @property
    def cv(self) -> float:
        return float(np.sqrt(self.var) / self.mean)

    @property
    def lam(self) -> float:
        return self.spectrum.lam1


def principal_rate(gen: KilledGenerator, *, rtol: float = 1e-12, max_iter: int = 500) -> float:
    """``lambda_1`` by inverse iteration on ``(-Q)^{-1}`` with Collatz-Wielandt bounds.

    ``(-Q)^{-1}`` is entrywise positive for an irreducible killed generator, so its
    Perron root ``1/lambda_1`` is bracketed by ``min(w/v)`` and ``max(w/v)`` for any
    positive ``v``; the iteration stops when the bracket closes to ``rtol``. This
    remains reliable when ``lambda_1`` is exponentially small, where shift-invert
    Arnoldi around zero is ill-conditioned.
    """

    lu = spla.splu((-gen.Q).tocsc())
    v = np.ones(gen.n_states)
    for _ in range(max_iter):
        w = lu.solve(v)
        wmax = float(np.max(w))
        if not wmax > 0:
            raise DomainError("(-Q)^{-1} is not positive: the generator is not irreducible and killed")
        # entries below the floating-point floor of the solve carry no information
        ok = (w > 1e-10 * wmax) & (v > 1e-10 * float(np.max(v)))
        ratio = w[ok] / v[ok]
        lo, hi = float(ratio.min()), float(ratio.max())
        v = np.maximum(w, 1e-12 * wmax) / wmax
        if hi / lo - 1.0 < rtol:
            break
    return float(2.0 / (lo + hi))


def spectrum(gen: KilledGenerator, k: int = 6) -> Spectrum:
    """``lambda_1`` (inverse iteration) and the next eigenvalue of ``-Q`` (shift-invert Arnoldi)."""

    lam1 = principal_rate(gen)
    Q = gen.Q.tocsc()
    n = Q.shape[0]
    if n <= 400:
        vals = np.linalg.eigvals(Q.toarray())
    else:
        vals = spla.eigs(Q, k=k, sigma=0.0, which="LM", return_eigenvectors=False, tol=1e-10, maxiter=20000)
    vals = -np.asarray(vals)  # eigenvalues of -Q
    # drop the principal eigenvalue, then take the next by real part
    i1 = int(np.argmin(np.abs(vals - lam1)))
    rest = np.delete(vals, i1)
    rest = rest[np.argsort(rest.real)]
    lam2 = complex(rest[0])
    if lam2.real <= lam1 * (1 + 1e-9):
        raise DomainError("no spectral gap resolved")
    return Spectrum(lam1=lam1, lam2=lam2, gap=float(lam2.real - lam1), eigenvalues=np.sort_complex(vals))


def moments(gen: KilledGenerator) -> tuple[float, float]:
    """``E T`` and ``Var T`` from ``(-Q) m = 1`` and ``(-Q) m2 = 2 m``."""

    A = (-gen.Q).tocsc()
    lu = spla.splu(A)
    m = lu.solve(np.ones(A.shape[0]))
    m2 = lu.solve(2.0 * m)
    mean = float(gen.mu0 @ m)
    second = float(gen.mu0 @ m2)
    return mean, float(second - mean**2)


def mean_only(gen: KilledGenerator) -> float:
    """``E T`` from ``(-Q) m = 1``."""

    A = (-gen.Q).tocsc()
    return float(gen.mu0 @ spla.spsolve(A, np.ones(A.shape[0])))


def default_t_end(spec: Spectrum, margin: float = 32.0) -> float:
    """Time after which the hazard is constant to about ``exp(-margin)`` relative."""

    ratio = max(spec.lam2.real / max(spec.lam1, _TINY), np.e)
    return float((np.log(ratio) + margin) / max(spec.gap, _TINY))


def timing_law(gen: KilledGenerator, grid: TimeGrid, *, spec: Spectrum | None = None) -> TimingLaw:
    """Survival, density and hazard on ``grid`` by TR-BDF2 forward propagation."""

    spec = spec or spectrum(gen)
    mean, var = moments(gen)
    t, dts = grid.times()
    A = gen.Q.T.tocsc()
    I = sp.identity(A.shape[0], format="csc")
    g = 2.0 - np.sqrt(2.0)
    c_bdf = (1.0 - g) / (2.0 - g)
    a1 = 1.0 / (g * (2.0 - g))
    a2 = (1.0 - g) ** 2 / (g * (2.0 - g))
    p = gen.mu0.astype(float).copy()
    S = [p.sum()]
    f = [float(gen.kill @ p)]
    cache: dict[float, tuple] = {}
    for dt in dts:
        if dt not in cache:
            lu_tr = spla.splu((I - (g * dt / 2.0) * A).tocsc())
            rhs_tr = (I + (g * dt / 2.0) * A).tocsr()
            lu_bdf = spla.splu((I - (c_bdf * dt) * A).tocsc())
            cache = {dt: (lu_tr, rhs_tr, lu_bdf)}
        lu_tr, rhs_tr, lu_bdf = cache[dt]
        pstar = lu_tr.solve(rhs_tr @ p)
        p = lu_bdf.solve(a1 * pstar - a2 * p)
        S.append(p.sum())
        f.append(float(gen.kill @ p))
    S = np.asarray(S)
    f = np.asarray(f)
    if np.min(S) < -1e-10:
        raise DomainError("negative survival: time grid too coarse")
    S = np.maximum(S, _TINY)
    f = np.maximum(f, 0.0)
    h = f / S
    # absorbed mass (trapezoid in time) must balance survival
    absorbed = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])
    mass_defect = float(np.max(np.abs(absorbed + S - 1.0)))
    tail_err = float(abs(h[-1] / spec.lam1 - 1.0))
    return TimingLaw(gen.label, t, S, f, h, spec, mean, var, tail_err, mass_defect, dict(gen.meta))


def _phi(a: Array, b: Array) -> Array:
    a = np.maximum(a, 0.0)
    b = np.maximum(b, _TINY)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(a > 0, a * np.log(np.maximum(a, _TINY) / b), 0.0) - a + b
    return np.maximum(val, 0.0)


def _trapz(y: Array, t: Array) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def kl_timing(A: TimingLaw, B: TimingLaw) -> dict[str, float]:
    """``KL(P_A || P_B)`` by the hazard identity, with the density form as a cross-check.

    Returned keys: ``kl`` (hazard form), ``kl_body`` and ``kl_tail`` (its grid and
    closed-form tail parts), ``kl_density_form`` and ``kl_rate_only`` (the
    divergence between two exponential laws with the same escape rates).
    """

    if A.t.shape != B.t.shape or np.max(np.abs(A.t - B.t)) > 1e-12:
        raise DomainError("timing laws must share a time grid")
    t = A.t
    body = _trapz(A.S * _phi(A.h, B.h), t)
    lamA, lamB = A.lam, B.lam
    tail = float(A.S[-1] * _phi(np.array(lamA), np.array(lamB)) / lamA)
    # density form: int f_A log(f_A / f_B)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(A.f > 0, A.f * np.log(np.maximum(A.f, _TINY) / np.maximum(B.f, _TINY)), 0.0)
    body_d = _trapz(dens, t)
    fa, fb = A.f[-1], B.f[-1]
    tail_d = float((fa / lamA) * (np.log(max(fa, _TINY) / max(fb, _TINY)) - (lamA - lamB) / lamA))
    return {
        "kl": body + tail,
        "kl_body": body,
        "kl_tail": tail,
        "kl_density_form": body_d + tail_d,
        "kl_rate_only": float(np.log(lamA / lamB) + lamB / lamA - 1.0),
    }


def hazard_relaxation_time(law: TimingLaw, *, K: float, gamma: float) -> float:
    """Smallest grid time ``t0`` with ``|h(u)/lambda - 1| <= K exp(-gamma (u - t0))`` for all ``u >= t0``."""

    eps = np.abs(law.h / law.lam - 1.0)
    logg = np.log(np.maximum(eps, _TINY)) + gamma * law.t
    suffix = np.maximum.accumulate(logg[::-1])[::-1]
    ok = suffix <= np.log(K) + gamma * law.t
    if not ok[-1]:
        raise DomainError("hazard has not relaxed by the end of the grid")
    # first index from which the condition holds for every later time
    bad = np.nonzero(~ok)[0]
    i0 = 0 if bad.size == 0 else int(bad[-1] + 1)
    return float(law.t[i0])


def binary_kl(p: float, q: float) -> float:
    p = min(max(p, _TINY), 1 - 1e-16)
    q = min(max(q, _TINY), 1 - 1e-16)
    return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


def decomposition_at(A: TimingLaw, B: TimingLaw, t0: float) -> dict[str, float]:
    """Chain-rule split of ``KL(P_A || P_B)`` at ``t0`` (Theorem 2, Eq. 2 of the article).

    ``KL = kl(F_A(t0) || F_B(t0)) + F_A(t0) KL_early + S_A(t0) KL_late``.
    """

    t = A.t
    i0 = int(np.searchsorted(t, t0))
    i0 = min(max(i0, 1), t.size - 2)
    t0 = float(t[i0])
    FA, FB = 1.0 - A.S[i0], 1.0 - B.S[i0]
    # early part: conditional densities f/F on [0, t0]
    with np.errstate(divide="ignore", invalid="ignore"):
        fe_a = A.f[: i0 + 1] / FA
        fe_b = B.f[: i0 + 1] / FB
        dens = np.where(fe_a > 0, fe_a * np.log(np.maximum(fe_a, _TINY) / np.maximum(fe_b, _TINY)), 0.0)
    kl_early = _trapz(dens, t[: i0 + 1])
    # late part: residual life after t0, hazards h(t0 + u)
    SA_rel = A.S[i0:] / A.S[i0]
    late_body = _trapz(SA_rel * _phi(A.h[i0:], B.h[i0:]), t[i0:])
    late_tail = float(SA_rel[-1] * _phi(np.array(A.lam), np.array(B.lam)) / A.lam)
    kl_late = late_body + late_tail
    total = binary_kl(FA, FB) + FA * kl_early + A.S[i0] * kl_late
    return {
        "t0": t0,
        "F_A": float(FA),
        "F_B": float(FB),
        "binary_kl": binary_kl(FA, FB),
        "kl_early_conditional": float(kl_early),
        "kl_late_conditional": float(kl_late),
        "kl_reassembled": float(total),
    }


def match_rate(
    make_white,
    target_lam: float,
    *,
    D_lo: float,
    D_hi: float,
    rtol: float = 1e-10,
) -> tuple[float, KilledGenerator]:
    """Escape-equivalent intensity ``D_esc``: the white model's escape rate equals ``target_lam``.

    ``make_white(D)`` returns a white-noise generator. Brent's method on ``log D``
    within ``[D_lo, D_hi]``; the bracket is checked at its endpoints.
    """

    def f(logD: float) -> float:
        return np.log(principal_rate(make_white(float(np.exp(logD))))) - np.log(target_lam)

    lo, hi = np.log(D_lo), np.log(D_hi)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise DomainError("rate-matching bracket does not contain the target")
    logD = brentq(f, lo, hi, xtol=rtol, rtol=rtol)
    D = float(np.exp(logD))
    return D, make_white(D)


def match_mean(
    make_white,
    target_mean: float,
    *,
    D_lo: float,
    D_hi: float,
    rtol: float = 1e-10,
) -> tuple[float, KilledGenerator]:
    """Intensity ``D`` at which the white model's mean commitment time equals ``target_mean``."""

    def f(logD: float) -> float:
        return np.log(mean_only(make_white(float(np.exp(logD))))) - np.log(target_mean)

    lo, hi = np.log(D_lo), np.log(D_hi)
    if f(lo) * f(hi) > 0:
        raise DomainError("mean-matching bracket does not contain the target")
    logD = brentq(f, lo, hi, xtol=rtol, rtol=rtol)
    D = float(np.exp(logD))
    return D, make_white(D)
