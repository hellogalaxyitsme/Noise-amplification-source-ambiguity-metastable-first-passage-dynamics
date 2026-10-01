"""Transition-path duration laws for killed Markov generators.

A transition path is the final segment of a commitment trajectory, from its last
exit of the well region ``A`` (here ``{x <= x_A}``) to the commitment threshold.
Its law follows from discrete transition-path theory in the quasi-stationary
setting:

* committor ``q(j) = P_j(commit before entering A)``, solving ``(Q q)(j) + kill(j) = 0``
  off ``A`` with ``q = 0`` on ``A``;
* quasi-stationary law ``nu`` (left Perron vector of ``Q``);
* reactive entrance law ``rho(j) propto sum_{i in A} nu(i) Q(i, j) q(j)`` for ``j`` outside ``A``;
* reactive dynamics given by the Doob transform of ``Q`` with ``q`` on the
  complement of ``A``: ``Q^q(j, k) = Q(j, k) q(k) / q(j)``, ``kill^q(j) = kill(j) / q(j)``,
  with the diagonal unchanged.

The transition-path duration is the commitment time of the returned
:class:`~metastable_fpt.generators.KilledGenerator`, so
:func:`metastable_fpt.timing_law.timing_law` gives its law.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .generators import KilledGenerator
from .seeding import DomainError

Array = np.ndarray


def committor(gen: KilledGenerator, in_A: Array) -> Array:
    Q = gen.Q.tocsr()
    out = np.nonzero(~in_A)[0]
    Qoo = Q[out][:, out].tocsc()
    q = np.zeros(gen.n_states)
    q[out] = spla.spsolve(Qoo, -gen.kill[out])
    if np.any(q[out] <= 0) or np.any(q[out] > 1 + 1e-9):
        raise DomainError("committor outside (0, 1]")
    return np.clip(q, 0.0, 1.0)


def quasi_stationary(gen: KilledGenerator) -> Array:
    vals, vecs = spla.eigs(gen.Q.T.tocsc(), k=1, sigma=0.0, which="LM", tol=1e-12, maxiter=20000)
    v = np.real(vecs[:, 0])
    v = v * np.sign(v[np.argmax(np.abs(v))])
    v = np.maximum(v, 0.0)
    return v / v.sum()


def transition_path_generator(gen: KilledGenerator, in_A: Array, *, label: str | None = None) -> KilledGenerator:
    """Killed generator whose commitment time is the transition-path duration."""

    in_A = np.asarray(in_A, dtype=bool)
    if in_A.shape != (gen.n_states,) or not in_A.any() or in_A.all():
        raise DomainError("in_A must be a proper non-empty subset of states")
    q = committor(gen, in_A)
    nu = quasi_stationary(gen)
    Q = gen.Q.tocsr()
    out = np.nonzero(~in_A)[0]
    A_idx = np.nonzero(in_A)[0]
    flux = (Q[A_idx][:, out].T @ nu[A_idx]) * q[out]
    if flux.sum() <= 0:
        raise DomainError("no reactive flux out of A")
    rho = flux / flux.sum()
    Qoo = Q[out][:, out].tocoo()
    qo = q[out]
    off = Qoo.row != Qoo.col
    vals = Qoo.data.copy()
    vals[off] = vals[off] * qo[Qoo.col[off]] / qo[Qoo.row[off]]
    Qh = sp.coo_matrix((vals, (Qoo.row, Qoo.col)), shape=Qoo.shape).tocsr()
    kill_h = gen.kill[out] / qo
    # the original diagonal is kept; rows sum to zero with kill_h by the committor equation
    tp = KilledGenerator(Qh, kill_h, rho, label or f"{gen.label}:transition-path", dict(gen.meta, reactive=True))
    tp.validate(tol=1e-7)
    return tp
