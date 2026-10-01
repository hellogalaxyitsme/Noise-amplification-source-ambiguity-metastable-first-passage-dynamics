"""Forcing samplers, the Euler-Maruyama crossing simulation and censoring-aware statistics."""

from __future__ import annotations

import numpy as np
import pytest

from metastable_fpt.forcing import (
    CompoundPoissonStream,
    OUStream,
    TelegraphStream,
    analytic_integrated_variance,
    analytic_lag1_state_correlation,
    analytic_telegraph_increment_correlation,
    analytic_telegraph_increment_covariance,
    analytic_telegraph_increment_variance,
    exact_flip_probability,
)
from metastable_fpt.seeding import DomainError, make_generator
from metastable_fpt.simulation import simulate_crossing
from metastable_fpt.survival_stats import (
    ObservedLaw,
    common_horizon,
    equivalence_verdict,
    observed_law_from_times,
    power_positive_control,
    rmst_difference,
)


def test_streams_are_reproducible_and_distinct():
    a = make_generator(11, "x").standard_normal(5)
    b = make_generator(11, "x").standard_normal(5)
    c = make_generator(11, "y").standard_normal(5)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    with pytest.raises(DomainError):
        make_generator(0, "x")


def test_telegraph_state_correlation_and_flip_probability():
    rate, dt, n = 0.5, 0.05, 20000
    assert exact_flip_probability(rate, dt) == pytest.approx((1 - np.exp(-2 * rate * dt)) / 2)
    assert analytic_lag1_state_correlation(rate, dt) == pytest.approx(np.exp(-2 * rate * dt))
    stream = TelegraphStream(D=0.05, rate=rate, dt=dt, n_paths=n, rng=make_generator(5, "tele"))
    rng = make_generator(6, "tele_steps")
    states = [stream.state.copy()]
    for _ in range(50):
        stream.next_increments(rng)
        states.append(stream.state.copy())
    x = np.array(states)
    lag1 = float(np.mean([np.corrcoef(x[k], x[k + 1])[0, 1] for k in range(x.shape[0] - 1)]))
    assert lag1 == pytest.approx(np.exp(-2 * rate * dt), abs=0.01)


def test_telegraph_integrated_variance_matches_finite_horizon_law():
    D, dt = 0.05, 0.05
    for rate, n_steps, n_paths in ((20.0, 40, 20000), (0.5, 400, 20000)):
        stream = TelegraphStream(D=D, rate=rate, dt=dt, n_paths=n_paths, rng=make_generator(11, f"tele{rate}"))
        rng = make_generator(12, f"tele_steps{rate}")
        total = np.zeros(n_paths)
        for _ in range(n_steps):
            total += stream.next_increments(rng)
        target = analytic_integrated_variance(D, 1.0 / (2.0 * rate), n_steps * dt)
        se = target * np.sqrt(2.0 / (n_paths - 1))
        assert abs(float(np.var(total, ddof=1)) - target) < 4.0 * se


def test_telegraph_step_integrals_match_their_covariance():
    D, rate, dt = 0.05, 0.5, 0.5
    n_paths, n_steps = 40000, 30
    target_var = analytic_telegraph_increment_variance(D, rate, dt)
    target_cov = analytic_telegraph_increment_covariance(D, rate, dt)
    target_corr = analytic_telegraph_increment_correlation(D, rate, dt)
    stream = TelegraphStream(D=D, rate=rate, dt=dt, n_paths=n_paths, rng=make_generator(21, "ti"))
    rng = make_generator(22, "ti_steps")
    prev = stream.next_increments(rng)
    var_acc, cov_acc = prev * prev, np.zeros(n_paths)
    for _ in range(n_steps - 1):
        cur = stream.next_increments(rng)
        var_acc += cur * cur
        cov_acc += cur * prev
        prev = cur
    measured_var = float(var_acc.mean() / n_steps)
    measured_cov = float(cov_acc.mean() / (n_steps - 1))
    assert abs(measured_var - target_var) < 4.0 * target_var * np.sqrt(2.0 / (n_paths * n_steps))
    assert abs(measured_cov - target_cov) < 6.0 * abs(target_cov) * np.sqrt(2.0 / (n_paths * (n_steps - 1)))
    assert measured_cov / measured_var == pytest.approx(target_corr, abs=0.02)


def test_ou_integrated_variance_and_state_correlation():
    D, tau, dt = 0.05, 2.0, 0.1
    n_paths, n_steps = 20000, 300
    stream = OUStream(D=D, tau=tau, dt=dt, n_paths=n_paths, rng=make_generator(3, "ou"))
    rng = make_generator(4, "ou_steps")
    total = np.zeros(n_paths)
    states = []
    for _ in range(n_steps):
        total += stream.next_increments(rng)
        states.append(stream.state.copy())
    target = analytic_integrated_variance(D, tau, n_steps * dt)
    assert abs(float(np.var(total, ddof=1)) - target) < 4.0 * target * np.sqrt(2.0 / (n_paths - 1))
    x = np.array(states)
    assert float(np.corrcoef(x[:-1].ravel(), x[1:].ravel())[0, 1]) == pytest.approx(np.exp(-dt / tau), abs=0.02)


def test_compound_poisson_long_time_diffusion():
    D, dt, rate = 0.02, 0.01, 5.0
    n_paths, n_steps = 40000, 200
    stream = CompoundPoissonStream(D=D, rate=rate, dt=dt, n_paths=n_paths)
    rng = make_generator(9, "jump")
    total = np.zeros(n_paths)
    for _ in range(n_steps):
        total += stream.next_increments(rng)
    measured = float(np.var(total, ddof=1)) / (2.0 * n_steps * dt)
    assert abs(measured - D) < 4.0 * D * np.sqrt(2.0 / (n_paths - 1))


def test_crossing_time_of_drifted_brownian_motion():
    # dx = v dt + sqrt(2D) dW from 0 to theta: E T = theta / v, Var T = 2 D theta / v^3
    v, D, theta, n = 1.0, 0.02, 1.0, 20000
    res = simulate_crossing(drift=lambda x: np.full_like(x, v), threshold=theta, x0=0.0, dt=1e-3,
                            n_steps=5000, n_paths=n, rng=make_generator(1, "bm"), white_D=D)
    assert res.n_events == n
    sd = np.sqrt(2 * D * theta / v**3)
    # discrete monitoring of the threshold delays crossings by O(sqrt(D dt) / v)
    monitoring_bias = 0.6 * np.sqrt(2 * D * 1e-3) / v
    assert abs(res.times.mean() - theta / v) < 4 * sd / np.sqrt(n) + monitoring_bias


def test_restricted_mean_equals_sample_mean_under_administrative_censoring():
    law = ObservedLaw(np.array([1.0, 2.0, 5.0, 5.0]), np.array([True, True, False, False]), horizon=5.0)
    assert law.administratively_censored
    assert law.restricted_mean() == pytest.approx(3.25)
    assert law.restricted_mean() == pytest.approx(law.restricted_mean_reference())
    early = ObservedLaw(np.array([1.0, 2.0, 3.0]), np.array([True, False, True]), horizon=5.0)
    assert not early.administratively_censored
    assert early.restricted_mean() > float(np.mean(early.time))


def test_equivalence_verdict_classes():
    assert equivalence_verdict(0.01, 0.02, 0.05)["verdict"] == "equivalent_within_margin"
    assert equivalence_verdict(0.20, 0.30, 0.05)["verdict"] == "differs_beyond_margin"
    assert equivalence_verdict(0.02, 0.40, 0.05)["verdict"] == "differs_within_margin_band"
    assert equivalence_verdict(-0.06, 0.03, 0.05)["verdict"] == "indeterminate"
    with pytest.raises(DomainError):
        equivalence_verdict(0.0, 1.0, 0.0)


def test_rmst_difference_uses_the_common_horizon():
    a = observed_law_from_times(np.array([1.0, 2.0, np.inf]), horizon=10.0)
    b = observed_law_from_times(np.array([1.5, 2.5]), horizon=4.0)
    assert common_horizon(a, b) == 4.0
    res = rmst_difference(a, b)
    assert res["horizon"] == 4.0
    assert res["method"] == "normal_ci_sample_means"
    assert res["ci_low"] <= res["difference"] <= res["ci_high"]


def test_positive_control_detects_a_known_effect():
    rng = np.random.default_rng(7)
    base = observed_law_from_times(rng.exponential(400.0, size=400), horizon=2000.0)
    control = power_positive_control(base, rng=rng, shift_fraction=0.3, n_reps=20)
    assert control["status"] == "calibrated"
    assert control["power_detected_fraction"] >= 0.8
    assert control["false_positive_fraction"] <= 0.2
