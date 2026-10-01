"""Killed generators, timing laws, divergences and transition paths.

Each test compares against an independent reference (quadrature, a dense
eigenvalue solver, a closed form or an algebraic identity).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from metastable_fpt.elasticity import weighted_barrier
from metastable_fpt.generators import (
    Domain,
    composite_generator,
    jump_generator,
    ou_generator,
    quartic_gradient,
    telegraph_generator,
    white_generator,
    x_index_of_states,
)
from metastable_fpt.seeding import DomainError
from metastable_fpt.timing_law import (
    TimeGrid,
    binary_kl,
    decomposition_at,
    default_t_end,
    hazard_relaxation_time,
    kl_timing,
    match_mean,
    match_rate,
    mean_only,
    principal_rate,
    spectrum,
    timing_law,
)
from metastable_fpt.transition_paths import committor, transition_path_generator

GU = quartic_gradient()
U = lambda x: np.asarray(x) ** 4 / 4 - np.asarray(x) ** 2 / 2  # noqa: E731
DOM = Domain(-2.0, 0.0, 120)


def _grid(*specs):
    return TimeGrid(dt0=2e-3, steps=80, t_end=max(default_t_end(s) for s in specs))


@pytest.mark.parametrize(
    "make",
    [
        lambda: white_generator(GU, D=0.1, x0=-1.0, dom=DOM),
        lambda: white_generator(GU, D=lambda x: 0.1 * (1 + 0.3 * (np.asarray(x) + 1)), x0=-1.0, dom=DOM),
        lambda: ou_generator(GU, D_C=0.05, D_Q=0.05, tau=0.5, x0=-1.0, dom=DOM, m_eta=21),
        lambda: telegraph_generator(GU, D_C=0.05, D_Q=0.05, rate=1.0, x0=-1.0, dom=DOM),
        lambda: jump_generator(GU, D_C=0.05, D_Q=0.05, jump_nodes=5, x0=-1.0, dom=DOM),
        lambda: composite_generator(GU, D_C=0.04, x0=-1.0, ou=(0.004, 20.0), telegraph=(0.004, 0.5), dom=DOM, m_eta=15),
    ],
)
def test_generators_are_valid_killed_chains(make):
    g = make()
    g.validate()
    row = np.asarray(g.Q.sum(axis=1)).ravel() + g.kill
    assert np.max(np.abs(row)) < 1e-8 * np.max(np.abs(g.Q.diagonal()))
    assert x_index_of_states(g, DOM).size == g.n_states


def test_white_chain_mean_matches_quadrature():
    D = 0.1
    ref = weighted_barrier(U, GU, D=D, x0=-1.0, ell=-2.0, b=0.0, n=40001, check_fd=False).mean
    g = white_generator(GU, D=D, x0=-1.0, dom=Domain(-2.0, 0.0, 800))
    assert abs(mean_only(g) - ref) / ref < 2e-3


def test_composite_without_forcing_equals_white():
    a = composite_generator(GU, D_C=0.1, x0=-1.0, dom=DOM)
    b = white_generator(GU, D=0.1, x0=-1.0, dom=DOM)
    assert abs(mean_only(a) - mean_only(b)) < 1e-9 * mean_only(b)


def test_negative_rate_is_rejected():
    g = white_generator(GU, D=0.1, x0=-1.0, dom=DOM)
    bad = type(g)(g.Q + sp.csr_matrix(([-1.0], ([0], [1])), shape=g.Q.shape), g.kill, g.mu0, "bad")
    with pytest.raises(DomainError):
        bad.validate()


def test_principal_rate_matches_dense_eigenvalue():
    g = white_generator(GU, D=0.08, x0=-1.0, dom=Domain(-2.0, 0.0, 60))
    ev = np.linalg.eigvals(-g.Q.toarray())
    assert abs(principal_rate(g) - ev.real.min()) / ev.real.min() < 1e-9


def test_timing_law_mass_balance_and_mean():
    g = white_generator(GU, D=0.1, x0=-1.0, dom=DOM)
    s = spectrum(g)
    law = timing_law(g, _grid(s), spec=s)
    assert law.mass_defect < 1e-5
    assert law.tail_hazard_rel_err < 1e-8
    # mean from the survival curve (with the constant-hazard tail) equals the linear-solve mean
    mean_S = np.sum(0.5 * (law.S[1:] + law.S[:-1]) * np.diff(law.t)) + law.S[-1] / law.lam
    assert abs(mean_S - law.mean) / law.mean < 1e-4


def test_hazard_and_density_forms_of_kl_agree_and_vanish_on_equal_laws():
    a = white_generator(GU, D=0.1, x0=-1.0, dom=DOM)
    b = white_generator(GU, D=0.11, x0=-1.0, dom=DOM)
    sa, sb = spectrum(a), spectrum(b)
    grid = _grid(sa, sb)
    A, B = timing_law(a, grid, spec=sa), timing_law(b, grid, spec=sb)
    k = kl_timing(A, B)
    assert k["kl"] > 0
    assert abs(k["kl"] - k["kl_density_form"]) / k["kl"] < 1e-3
    assert kl_timing(A, A)["kl"] < 1e-12


def test_decomposition_reassembles_and_late_bound_holds():
    D = 0.05
    mech = ou_generator(GU, D_C=D / 2, D_Q=D / 2, tau=0.5, x0=-1.0, dom=DOM, m_eta=21)
    lam = spectrum(mech).lam1
    mk = lambda DD: white_generator(GU, D=DD, x0=-1.0, dom=DOM)  # noqa: E731
    D_esc, w = match_rate(mk, lam, D_lo=D / 2.5, D_hi=2.5 * D)
    assert abs(principal_rate(w) / lam - 1) < 1e-8
    sm, sw = spectrum(mech), spectrum(w)
    grid = _grid(sm, sw)
    A, B = timing_law(mech, grid, spec=sm), timing_law(w, grid, spec=sw)
    kl = kl_timing(A, B)["kl"]
    gamma = 0.9 * min(sm.gap, sw.gap)
    t0 = max(hazard_relaxation_time(A, K=0.5, gamma=gamma), hazard_relaxation_time(B, K=0.5, gamma=gamma))
    dec = decomposition_at(A, B, t0)
    assert abs(dec["kl_reassembled"] - kl) / kl < 1e-2
    assert dec["kl_late_conditional"] <= 2 * 0.25 * lam / (0.5 * gamma) * (1 + 1e-6)
    # escape matching removes almost all of the divergence left by effective-diffusion matching
    kl_eff = kl_timing(A, timing_law(mk(D), grid, spec=spectrum(mk(D))))["kl"]
    assert kl < 0.01 * kl_eff


def test_match_mean_hits_target():
    mech = telegraph_generator(GU, D_C=0.05, D_Q=0.05, rate=1.0, x0=-1.0, dom=DOM)
    target = mean_only(mech)
    D, w = match_mean(lambda DD: white_generator(GU, D=DD, x0=-1.0, dom=DOM), target, D_lo=0.03, D_hi=0.3)
    assert abs(mean_only(w) / target - 1) < 1e-8


def test_binary_kl():
    assert binary_kl(0.3, 0.3) == pytest.approx(0.0, abs=1e-15)
    assert binary_kl(0.1, 0.2) > 0


def test_transition_path_generator_is_valid_and_short():
    g = white_generator(GU, D=0.05, x0=-1.0, dom=DOM)
    in_A = x_index_of_states(g, DOM) <= np.searchsorted(DOM.x, -1.0)
    q = committor(g, in_A)
    assert np.all(q[in_A] == 0) and np.all((q[~in_A] > 0) & (q[~in_A] <= 1))
    assert np.all(np.diff(q[~in_A]) > 0)  # the one-dimensional committor increases towards the threshold
    tp = transition_path_generator(g, in_A)
    assert 0 < mean_only(tp) < 0.05 * mean_only(g)
