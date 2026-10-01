"""Lemma 4, Lemma 5 and Theorem 6 against finite differences, chain moments and closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from metastable_fpt.elasticity import elasticity_bound, elasticity_bound_stages, weighted_barrier
from metastable_fpt.generators import Domain, quartic_gradient, white_generator
from metastable_fpt.timing_law import moments

GU = quartic_gradient()
U = lambda x: np.asarray(x) ** 4 / 4 - np.asarray(x) ** 2 / 2  # noqa: E731


@pytest.mark.parametrize("D", [0.2, 0.05, 0.025])
def test_elasticity_identity_equals_finite_difference(D):
    r = weighted_barrier(U, GU, D=D, x0=-1.0, ell=-2.0, b=0.0, n=40001)
    assert abs(r.elasticity - r.elasticity_finite_difference) < 1e-5 * max(1.0, r.elasticity)
    assert r.elasticity <= r.bound


def test_variance_identity_equals_chain_moments():
    D = 0.07
    r = weighted_barrier(U, GU, D=D, x0=-1.0, ell=-2.0, b=0.0, n=40001, check_fd=False)
    g = white_generator(GU, D=D, x0=-1.0, dom=Domain(-2.0, 0.0, 800))
    m, v = moments(g)
    assert abs(r.cv - np.sqrt(v) / m) / r.cv < 1e-2


def test_flat_potential_closed_forms():
    # U = 0: the mean scales as 1/D, so s = 1; reflected Brownian motion from the wall
    # to distance 1 has m = 1/(2D) and CV^2 = 2/3
    r = weighted_barrier(lambda x: 0 * np.asarray(x), lambda x: 0 * np.asarray(x), D=0.3, x0=0.0, ell=0.0, b=1.0,
                         n=20001, check_fd=False)
    assert r.elasticity == pytest.approx(1.0, abs=1e-9)
    assert r.mean == pytest.approx(1 / (2 * 0.3), rel=1e-6)
    assert r.cv == pytest.approx(np.sqrt(2 / 3), rel=1e-5)


def test_bound_holds_on_random_potentials():
    rng = np.random.default_rng(7)
    kk = np.arange(1, 5)
    for _ in range(40):
        a = rng.normal(size=4)
        tilt = rng.uniform(-1.5, 1.5)
        Uf = lambda x, a=a, t=tilt: t * np.asarray(x) + np.sum(a[:, None] * np.sin(np.outer(kk, x)), axis=0)  # noqa: E731
        dUf = lambda x, a=a, t=tilt: t + np.sum(kk[:, None] * a[:, None] * np.cos(np.outer(kk, x)), axis=0)  # noqa: E731
        D = float(10 ** rng.uniform(-1.2, 0.3))
        r = weighted_barrier(Uf, dUf, D=D, x0=float(rng.uniform(-2, 1.5)), ell=-2.0, b=2.0, n=10001, check_fd=False)
        if np.isfinite(r.elasticity):
            assert r.elasticity <= r.bound + 1e-9


def test_bound_is_decreasing_in_tau_L_and_stage_term_is_log_k():
    b = elasticity_bound(cv=0.5, mean=2.0, tau_L=np.array([1e-4, 1e-3, 1e-2]))
    assert np.all(np.diff(b) < 0)
    assert elasticity_bound_stages(cv=0.5, mean=2.0, tau_L=1e-3, n_stages=2) == pytest.approx(
        float(elasticity_bound(cv=0.5, mean=2.0, tau_L=1e-3)) + np.log(2))
