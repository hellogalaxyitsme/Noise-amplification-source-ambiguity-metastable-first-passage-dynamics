"""Weighted-barrier identities and the variability-amplification inequality.

For the one-dimensional gradient diffusion ``dX = -U'(X) dt + sqrt(2D) dW`` on
``[ell, b]``, reflecting at ``ell``, absorbing at ``b`` and started at ``x0``:

* mean:        ``m = int_{x0}^b int_ell^y D^-1 exp(Phi(y,z)/D) dz dy``, with
  ``Phi(y, z) = U(y) - U(z)``;
* elasticity:  ``s := -d log m / d log D = 1 + <Phi>_w / D``          (Lemma 4);
* variance:    ``Var T = 2 m <tau(z, y)>_w``, ``tau(z, y) = m(z) - m(y)``  (Lemma 5);

where ``w`` is the probability density ``D^-1 exp(Phi/D) / m`` on the triangle
``{x0 < y < b, ell < z < y}``. With ``L = sup |U'|``, ``tau_L = D / L^2`` and
``A = e^2 CV^2 m / (2 tau_L)``,

    s <= 3 + log A  (A >= e),      s <= 3 + A/e  (A < e)          (Theorem 6).

Lemma and theorem numbers refer to the article. All integrals are evaluated with
nested one-dimensional cumulative integrals in a locally shifted form, which
avoids overflow at small ``D``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .seeding import DomainError

Array = np.ndarray


def _cumtrapz(y: Array, x: Array) -> Array:
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _trapz(y: Array, x: Array) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


@dataclass
class ElasticityResult:
    mean: float
    var: float
    cv: float
    elasticity: float
    mean_barrier_over_D: float
    lipschitz: float
    tau_L: float
    bound: float
    bound_log_argument: float
    elasticity_finite_difference: float | None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def weighted_barrier(
    U: Callable[[Array], Array],
    dU: Callable[[Array], Array],
    *,
    D: float,
    x0: float,
    ell: float,
    b: float,
    n: int = 20001,
    check_fd: bool = True,
) -> ElasticityResult:
    """Mean, variance, elasticity ``s`` and the Theorem 6 bound by quadrature on ``n`` nodes.

    With ``check_fd=True`` the elasticity is also computed by a central finite
    difference of ``log m`` in ``log D`` (relative step ``1e-4``).
    """

    if not (ell <= x0 < b) or D <= 0:
        raise DomainError("need ell <= x0 < b and D > 0")
    z = np.linspace(ell, b, n)
    h = float(z[1] - z[0])
    Uz = np.asarray(U(z), dtype=float)
    Uz = Uz - Uz[np.searchsorted(z, x0)]          # reduces cancellation in the Phi moment
    step = np.exp(np.diff(Uz) / D)                # e^{(U_{i+1} - U_i)/D}: only local differences

    def left_integral(q: Array) -> Array:
        """``int_ell^y e^{(U(y) - U(z))/D} q(z) dz`` at every node, by a stable recursion."""

        out = np.zeros_like(Uz)
        for i in range(n - 1):
            e = step[i]
            out[i + 1] = e * out[i] + 0.5 * h * (e * q[i] + q[i + 1])
        return out

    ones = np.ones_like(Uz)
    G = left_integral(ones)                       # D g(y)
    g = G / D                                     # -m'(y)
    m_of = _trapz(g, z) - _cumtrapz(g, z)         # m(z) for every node
    sel = z >= x0
    mean = _trapz(g[sel], z[sel])
    # <Phi>: int D^-1 int e^{Phi/D} (U(y) - U(z)) dz dy = int D^-1 [U(y) G(y) - H(y)] dy
    H = left_integral(Uz)
    phi_num = _trapz(((Uz * G - H) / D)[sel], z[sel])
    mean_phi = phi_num / mean
    s = 1.0 + mean_phi / D
    # variance: 2 int D^-1 int e^{Phi/D} (m(z) - m(y)) dz dy
    K = left_integral(m_of)
    var = 2.0 * _trapz(((K - m_of * G) / D)[sel], z[sel])
    cv = float(np.sqrt(max(var, 0.0)) / mean) if np.isfinite(mean) and mean > 0 else float("nan")
    L = float(np.max(np.abs(dU(z))))
    tau_L = D / L**2 if L > 0 else np.inf
    log_mean = np.log(mean)
    log_A = 2.0 + 2.0 * np.log(cv) + log_mean - np.log(2.0 * tau_L) if cv > 0 else -np.inf
    bound = 3.0 + max(1.0, float(log_A)) if log_A >= 1.0 else 3.0 + float(np.exp(log_A - 1.0))
    fd = None
    if check_fd:
        eps = 1e-4
        up = weighted_barrier(U, dU, D=D * (1 + eps), x0=x0, ell=ell, b=b, n=n, check_fd=False)
        dn = weighted_barrier(U, dU, D=D * (1 - eps), x0=x0, ell=ell, b=b, n=n, check_fd=False)
        log_up = np.log(up.mean)
        log_dn = np.log(dn.mean)
        fd = float(-(log_up - log_dn) / (np.log(1 + eps) - np.log(1 - eps)))
    return ElasticityResult(
        mean=float(mean),
        var=float(var),
        cv=cv,
        elasticity=float(s),
        mean_barrier_over_D=float(mean_phi / D),
        lipschitz=L,
        tau_L=float(tau_L),
        bound=float(bound),
        bound_log_argument=float(log_A),
        elasticity_finite_difference=fd,
    )


def elasticity_bound(*, cv: float | Array, mean: float | Array, tau_L: float | Array) -> Array:
    """Theorem 6 upper bound on the elasticity from an observed ``CV`` and mean ``m``."""

    cv = np.asarray(cv, dtype=float)
    mean = np.asarray(mean, dtype=float)
    tau_L = np.asarray(tau_L, dtype=float)
    log_A = 2.0 + 2.0 * np.log(cv) + np.log(mean) - np.log(2.0 * tau_L)
    return np.where(log_A >= 1.0, 3.0 + log_A, 3.0 + np.exp(log_A - 1.0))


def elasticity_bound_stages(*, cv: float, mean: float, tau_L: float, n_stages: int) -> float:
    """Theorem 6 for ``k`` independent stages: the single-stage bound plus ``log k``."""

    if n_stages < 1:
        raise DomainError("n_stages must be >= 1")
    return float(elasticity_bound(cv=cv, mean=mean, tau_L=tau_L)) + float(np.log(n_stages))
