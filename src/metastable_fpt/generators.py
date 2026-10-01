"""Killed Markov generators for commitment processes driven by different noises.

Every model is represented as a finite continuous-time Markov chain with killing
(a Markov-chain approximation in the sense of Kushner and Dupuis). The commitment
time ``T`` of the chain is a phase-type random variable whose law follows from
linear algebra on the generator, so two noise mechanisms can be compared through
their timing laws without sampling error.

Conventions:

* ``Q`` holds transient-to-transient rates in the row convention: ``Q[i, j]`` is
  the rate ``i -> j`` for ``i != j`` and ``Q[i, i] = -(sum_j Q[i, j] + kill[i])``,
  hence ``Q @ 1 = -kill``.
* ``kill[i]`` is the rate at which state ``i`` jumps to the absorbing (committed)
  state.
* ``mu0`` is the initial law on transient states.

Mechanisms, all sharing the drift ``b(x) = -U'(x)``, a reflecting lower boundary
``ell`` and an absorbing threshold ``bnd``:

* white noise: ``dx = b(x) dt + sqrt(2 D(x)) dW`` (``D`` may depend on ``x``);
* OU forcing: ``dx = (b(x) + eta) dt + sqrt(2 D_C) dW_1`` with
  ``d eta = -eta/tau dt + sqrt(2 D_Q)/tau dW_2``; the stationary variance of
  ``eta`` is ``D_Q/tau`` and its long-time contribution to the diffusion is ``D_Q``;
* telegraph forcing: ``dx = (b(x) + a sigma_t) dt + sqrt(2 D_C) dW`` with
  ``sigma in {-1, +1}`` switching at rate ``r`` and ``a = sqrt(2 D_Q r)``, so that
  the long-time contribution ``a^2/(2r)`` equals ``D_Q``;
* jumps: white noise ``D_C`` plus symmetric jumps ``+-A`` at total rate ``nu``
  with ``nu A^2 / 2 = D_Q``.

Spatial operators use central differences where they are monotone (non-negative
off-diagonal rates) and first-order upwinding otherwise. Reflecting boundaries use
the ghost-node Neumann row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .seeding import DomainError

Array = np.ndarray


@dataclass(frozen=True)
class KilledGenerator:
    """A finite killed Markov generator plus its initial law."""

    Q: sp.csr_matrix
    kill: Array
    mu0: Array
    label: str
    meta: dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return int(self.Q.shape[0])

    def validate(self, tol: float = 1e-9) -> None:
        Q = self.Q.tocsr()
        n = Q.shape[0]
        if Q.shape != (n, n) or self.kill.shape != (n,) or self.mu0.shape != (n,):
            raise DomainError("inconsistent generator shapes")
        off = Q - sp.diags(Q.diagonal())
        if off.nnz and off.data.min() < -tol:
            raise DomainError("negative off-diagonal rate")
        if np.any(self.kill < -tol):
            raise DomainError("negative killing rate")
        row = np.asarray(Q.sum(axis=1)).ravel() + self.kill
        scale = max(1.0, float(np.max(np.abs(Q.diagonal()))))
        if np.max(np.abs(row)) > tol * scale:
            raise DomainError("rows of Q plus kill must sum to zero")
        if abs(self.mu0.sum() - 1.0) > 1e-9 or np.any(self.mu0 < -1e-15):
            raise DomainError("mu0 must be a probability vector")
        if not np.any(self.kill > 0):
            raise DomainError("no absorbing transition: the commitment time is infinite")


def axis_rates(drift: Array, diff: Array, h: float) -> tuple[Array, Array]:
    """Up and down rates on a uniform axis: central where monotone, upwind otherwise."""

    drift = np.asarray(drift, dtype=float)
    diff = np.asarray(diff, dtype=float)
    up = diff / h**2 + drift / (2.0 * h)
    down = diff / h**2 - drift / (2.0 * h)
    bad = (up < 0) | (down < 0)
    if np.any(bad):
        up = np.where(bad, diff / h**2 + np.maximum(drift, 0.0) / h, up)
        down = np.where(bad, diff / h**2 + np.maximum(-drift, 0.0) / h, down)
    return up, down


def _point_mass(grid: Array, x0: float) -> Array:
    """Linear split of a point mass between the two neighbouring nodes."""

    w = np.zeros(grid.size)
    if x0 <= grid[0]:
        w[0] = 1.0
        return w
    if x0 >= grid[-1]:
        w[-1] = 1.0
        return w
    k = int(np.searchsorted(grid, x0) - 1)
    frac = (x0 - grid[k]) / (grid[k + 1] - grid[k])
    w[k] = 1.0 - frac
    w[k + 1] = frac
    return w


class _Builder:
    """Accumulates COO triplets and killing rates."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.rows: list[Array] = []
        self.cols: list[Array] = []
        self.vals: list[Array] = []
        self.kill = np.zeros(n)
        self.out = np.zeros(n)

    def add(self, src: Array, dst: Array, rate: Array) -> None:
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        rate = np.broadcast_to(np.asarray(rate, dtype=float), src.shape).copy()
        keep = rate > 0
        src, dst, rate = src[keep], dst[keep], rate[keep]
        self.rows.append(src)
        self.cols.append(dst)
        self.vals.append(rate)
        np.add.at(self.out, src, rate)

    def add_kill(self, src: Array, rate: Array) -> None:
        src = np.asarray(src, dtype=np.int64)
        rate = np.broadcast_to(np.asarray(rate, dtype=float), src.shape)
        np.add.at(self.kill, src, rate)

    def build(self) -> tuple[sp.csr_matrix, Array]:
        rows = np.concatenate(self.rows + [np.arange(self.n)])
        cols = np.concatenate(self.cols + [np.arange(self.n)])
        vals = np.concatenate(self.vals + [-(self.out + self.kill)])
        Q = sp.coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsr()
        Q.sum_duplicates()
        return Q, self.kill.copy()


@dataclass(frozen=True)
class Domain:
    """Commitment interval ``[ell, bnd]`` with ``n`` transient x-nodes."""

    ell: float = -2.0
    bnd: float = 0.0
    n: int = 400

    def __post_init__(self) -> None:
        if not self.bnd > self.ell or self.n < 10:
            raise DomainError("need bnd > ell and at least 10 nodes")

    @property
    def h(self) -> float:
        return (self.bnd - self.ell) / self.n

    @property
    def x(self) -> Array:
        """Transient nodes ``ell + i h``, ``i = 0..n-1``; node ``n`` is absorbing."""

        return self.ell + self.h * np.arange(self.n)


def _add_x_transport(
    B: _Builder,
    idx: Callable[[Array], Array],
    dom: Domain,
    drift: Array,
    diff: float | Array,
) -> None:
    """x-direction moves for every transient x node; ``idx(i)`` maps x-index to state."""

    n, h = dom.n, dom.h
    i = np.arange(n)
    diff_arr = np.broadcast_to(np.asarray(diff, dtype=float), (n,))
    up, down = axis_rates(drift, diff_arr, h)
    # reflecting ghost row at i = 0: u_{-1} = u_1
    up = up.copy()
    down = down.copy()
    up[0] = 2.0 * diff_arr[0] / h**2 if diff_arr[0] > 0 else max(drift[0], 0.0) / h
    down[0] = 0.0
    inner = i[:-1]
    B.add(idx(inner), idx(inner + 1), up[:-1])
    B.add(idx(i[1:]), idx(i[1:] - 1), down[1:])
    B.add_kill(idx(np.array([n - 1])), up[-1:])


def white_generator(
    grad_U: Callable[[Array], Array],
    *,
    D: float | Callable[[Array], Array],
    x0: float,
    dom: Domain = Domain(),
    label: str = "white",
) -> KilledGenerator:
    """``dx = -U'(x) dt + sqrt(2 D(x)) dW`` on ``[ell, bnd]``."""

    x = dom.x
    drift = -np.asarray(grad_U(x), dtype=float)
    Dx = np.asarray(D(x), dtype=float) if callable(D) else np.full(dom.n, float(D))
    if np.any(Dx <= 0):
        raise DomainError("white-noise diffusion must be positive")
    B = _Builder(dom.n)
    _add_x_transport(B, lambda i: i, dom, drift, Dx)
    Q, kill = B.build()
    gen = KilledGenerator(
        Q, kill, _point_mass(x, x0), label,
        {"model": "white", "D": D if not callable(D) else "state-dependent", "n": dom.n, "h": dom.h},
    )
    gen.validate()
    return gen


def _gaussian_axis(sd: float, m: int, width: float) -> tuple[Array, float, Array]:
    grid = np.linspace(-width * sd, width * sd, m)
    k = grid[1] - grid[0]
    w = np.exp(-0.5 * (grid / sd) ** 2)
    return grid, k, w / w.sum()


def ou_generator(
    grad_U: Callable[[Array], Array],
    *,
    D_C: float,
    D_Q: float,
    tau: float,
    x0: float,
    dom: Domain = Domain(),
    m_eta: int = 61,
    width: float = 6.0,
    label: str = "ou",
) -> KilledGenerator:
    """White ``D_C`` plus an OU forcing with long-time diffusion ``D_Q`` and correlation time ``tau``.

    ``eta`` is discretised on ``m_eta`` equally spaced nodes over ``+-width``
    stationary standard deviations (``sqrt(D_Q/tau)``) with reflecting ends. The
    forcing starts in its stationary law, independent of ``x``.
    """

    if D_Q <= 0 or tau <= 0 or D_C < 0:
        raise DomainError("need D_Q > 0, tau > 0 and D_C >= 0")
    sd = np.sqrt(D_Q / tau)
    eta, k, w_eta = _gaussian_axis(sd, m_eta, width)
    n = dom.n
    N = n * m_eta
    B = _Builder(N)
    base = -np.asarray(grad_U(dom.x), dtype=float)
    for j in range(m_eta):
        _add_x_transport(B, lambda i, j=j: i * m_eta + j, dom, base + eta[j], D_C)
    # eta moves: drift -eta/tau, diffusion coefficient D_Q/tau^2, reflecting ends
    d_eta = D_Q / tau**2
    up, down = axis_rates(-eta / tau, np.full(m_eta, d_eta), k)
    up = up.copy()
    down = down.copy()
    up[0], down[0] = 2 * d_eta / k**2, 0.0
    up[-1], down[-1] = 0.0, 2 * d_eta / k**2
    i_all = np.arange(n)
    for j in range(m_eta):
        src = i_all * m_eta + j
        if j + 1 < m_eta:
            B.add(src, src + 1, up[j])
        if j - 1 >= 0:
            B.add(src, src - 1, down[j])
    Q, kill = B.build()
    mu0 = np.outer(_point_mass(dom.x, x0), w_eta).ravel()
    gen = KilledGenerator(
        Q, kill, mu0, label,
        {"model": "ou", "D_C": D_C, "D_Q": D_Q, "tau": tau, "n": n, "m_eta": m_eta, "width_sd": width},
    )
    gen.validate()
    return gen


def telegraph_generator(
    grad_U: Callable[[Array], Array],
    *,
    D_C: float,
    D_Q: float,
    rate: float,
    x0: float,
    dom: Domain = Domain(),
    label: str = "telegraph",
) -> KilledGenerator:
    """White ``D_C`` plus a telegraph forcing ``a sigma`` with ``a = sqrt(2 D_Q rate)``.

    The initial sign is ``+1`` or ``-1`` with probability 1/2, independent of ``x``.
    """

    if D_Q <= 0 or rate <= 0 or D_C <= 0:
        raise DomainError("need D_Q > 0, rate > 0 and D_C > 0")
    a = float(np.sqrt(2.0 * D_Q * rate))
    n = dom.n
    B = _Builder(2 * n)
    base = -np.asarray(grad_U(dom.x), dtype=float)
    for s, sign in enumerate((-1.0, 1.0)):
        _add_x_transport(B, lambda i, s=s: i * 2 + s, dom, base + sign * a, D_C)
    i_all = np.arange(n)
    B.add(i_all * 2, i_all * 2 + 1, rate)
    B.add(i_all * 2 + 1, i_all * 2, rate)
    Q, kill = B.build()
    mu0 = np.outer(_point_mass(dom.x, x0), [0.5, 0.5]).ravel()
    gen = KilledGenerator(
        Q, kill, mu0, label, {"model": "telegraph", "D_C": D_C, "D_Q": D_Q, "rate": rate, "a": a, "n": n}
    )
    gen.validate()
    return gen


def jump_generator(
    grad_U: Callable[[Array], Array],
    *,
    D_C: float,
    D_Q: float,
    jump_nodes: int,
    x0: float,
    dom: Domain = Domain(),
    label: str = "jump",
) -> KilledGenerator:
    """White ``D_C`` plus symmetric jumps of ``+-A`` (``A = jump_nodes * h``) with ``nu A^2/2 = D_Q``.

    A jump that would cross the threshold commits; a jump below ``ell`` is mirrored.
    """

    if D_Q <= 0 or D_C <= 0 or jump_nodes < 1:
        raise DomainError("need D_Q > 0, D_C > 0 and jump_nodes >= 1")
    n, h = dom.n, dom.h
    A = jump_nodes * h
    nu = 2.0 * D_Q / A**2
    B = _Builder(n)
    base = -np.asarray(grad_U(dom.x), dtype=float)
    _add_x_transport(B, lambda i: i, dom, base, D_C)
    i = np.arange(n)
    upj = i + jump_nodes
    B.add(i[upj < n], upj[upj < n], nu / 2)
    B.add_kill(i[upj >= n], nu / 2)
    dn = np.abs(i - jump_nodes)  # mirror at ell
    B.add(i, dn, np.where(dn == i, 0.0, nu / 2))
    Q, kill = B.build()
    gen = KilledGenerator(
        Q, kill, _point_mass(dom.x, x0), label,
        {"model": "jump", "D_C": D_C, "D_Q": D_Q, "A": A, "nu": nu, "n": n},
    )
    gen.validate()
    return gen


def composite_generator(
    grad_U: Callable[[Array], Array],
    *,
    D_C: float,
    x0: float,
    ou: tuple[float, float] | None = None,
    telegraph: tuple[float, float] | None = None,
    dom: Domain = Domain(),
    m_eta: int = 41,
    width: float = 6.0,
    label: str = "composite",
) -> KilledGenerator:
    """White ``D_C`` plus optional OU ``(D_Q, tau)`` and telegraph ``(D_Q, rate)`` forcings.

    The state is ``(x, eta, sigma)``; both forcings start stationary and independent.
    """

    if D_C <= 0:
        raise DomainError("composite generator needs D_C > 0")
    n = dom.n
    if ou is not None:
        DQo, tau = ou
        eta, k, w_eta = _gaussian_axis(np.sqrt(DQo / tau), m_eta, width)
    else:
        eta, k, w_eta, DQo, tau = np.zeros(1), 1.0, np.ones(1), 0.0, 1.0
    if telegraph is not None:
        DQt, rate = telegraph
        a = float(np.sqrt(2.0 * DQt * rate))
        sig, w_sig = np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    else:
        a, rate, DQt = 0.0, 0.0, 0.0
        sig, w_sig = np.zeros(1), np.ones(1)
    me, ms = eta.size, sig.size
    M = me * ms

    def idx(i, j, s):
        return (np.asarray(i) * me + j) * ms + s

    B = _Builder(n * M)
    base = -np.asarray(grad_U(dom.x), dtype=float)
    for j in range(me):
        for s in range(ms):
            _add_x_transport(B, lambda i, j=j, s=s: idx(i, j, s), dom, base + eta[j] + a * sig[s], D_C)
    i_all = np.arange(n)
    if me > 1:
        d_eta = DQo / tau**2
        up, down = axis_rates(-eta / tau, np.full(me, d_eta), k)
        up, down = up.copy(), down.copy()
        up[0], down[0] = 2 * d_eta / k**2, 0.0
        up[-1], down[-1] = 0.0, 2 * d_eta / k**2
        for j in range(me):
            for s in range(ms):
                src = idx(i_all, j, s)
                if j + 1 < me:
                    B.add(src, idx(i_all, j + 1, s), up[j])
                if j >= 1:
                    B.add(src, idx(i_all, j - 1, s), down[j])
    if ms > 1:
        for j in range(me):
            B.add(idx(i_all, j, 0), idx(i_all, j, 1), rate)
            B.add(idx(i_all, j, 1), idx(i_all, j, 0), rate)
    Q, kill = B.build()
    mu0 = np.einsum("i,j,s->ijs", _point_mass(dom.x, x0), w_eta, w_sig).ravel()
    gen = KilledGenerator(
        Q, kill, mu0, label,
        {"model": "composite", "D_C": D_C, "ou": ou, "telegraph": telegraph, "n": n, "m_eta": me},
    )
    gen.validate()
    return gen


def x_index_of_states(gen: KilledGenerator, dom: Domain) -> Array:
    """x-node index of every state (states are ordered x-major in every builder)."""

    per = gen.n_states // dom.n
    return np.repeat(np.arange(dom.n), per)


def quartic_gradient(a: float = 1.0, b: float = 1.0, c: float = 0.0) -> Callable[[Array], Array]:
    """``U'(x)`` for the normal form ``U(x) = a x^4/4 - b x^2/2 - c x``."""

    return lambda x: a * np.asarray(x) ** 3 - b * np.asarray(x) - c
