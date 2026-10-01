"""Censoring-aware comparison of simulated first-passage laws.

Observations are stored as the observed law ``(min(T, H), event)``. The restricted
mean survival time (RMST) up to the horizon ``H`` is computed exactly from the
Kaplan-Meier step function. When every censored observation sits at the
horizon (administrative censoring, the case for all simulations in this
package), the RMST equals the sample mean of ``min(T, H)`` and the difference of
two RMSTs has a closed-form normal confidence interval; otherwise a
Kaplan-Meier bootstrap is used.

Comparisons against a practical margin report statistical difference from zero
and equivalence within the margin as separate fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from .seeding import DomainError


@dataclass(frozen=True)
class ObservedLaw:
    """Censored first-passage observations ``(min(T, H), event)``."""

    time: np.ndarray
    event: np.ndarray
    horizon: float

    def __post_init__(self) -> None:
        t = np.asarray(self.time, dtype=float)
        e = np.asarray(self.event, dtype=bool)
        if t.shape != e.shape:
            raise DomainError("time and event must have the same shape")
        if t.size == 0:
            raise DomainError("empty observation set")
        if not np.all(np.isfinite(t)):
            raise DomainError("times must be finite (use the horizon for censored trials)")
        if np.any(t < 0) or np.any(t > self.horizon + 1e-12 * max(1.0, self.horizon)):
            raise DomainError("times must lie in [0, horizon]")
        object.__setattr__(self, "time", t)
        object.__setattr__(self, "event", e)

    @property
    def n(self) -> int:
        return int(self.time.size)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    @property
    def censor_fraction(self) -> float:
        return float(1.0 - self.event.mean())

    @property
    def administratively_censored(self) -> bool:
        """True when every censored observation sits at the horizon."""

        censored = ~self.event
        if not censored.any():
            return True
        return bool(np.all(np.isclose(self.time[censored], self.horizon, atol=1e-12)))

    def truncate(self, horizon: float) -> "ObservedLaw":
        """Administratively censor at a shorter horizon."""

        if horizon <= 0:
            raise DomainError("horizon must be positive")
        if horizon > self.horizon + 1e-12 * max(1.0, self.horizon):
            raise DomainError("cannot extend a law beyond its follow-up horizon")
        within = self.time <= horizon
        return ObservedLaw(
            np.where(within, self.time, horizon),
            np.where(within, self.event, False),
            float(horizon),
        )

    def kaplan_meier(self, grid: np.ndarray) -> np.ndarray:
        """Kaplan-Meier survival estimate evaluated on ``grid``."""

        grid = np.asarray(grid, dtype=float)
        times, values = self._km_steps()
        if times.size == 0:
            return np.ones_like(grid)
        idx = np.searchsorted(times, grid, side="right") - 1
        return np.where(idx >= 0, values[np.clip(idx, 0, values.size - 1)], 1.0)

    def _km_steps(self) -> tuple[np.ndarray, np.ndarray]:
        """Right-continuous Kaplan-Meier steps ``(time, S(time))`` with ties handled jointly."""

        order = np.argsort(self.time, kind="mergesort")
        t = self.time[order]
        e = self.event[order]
        n = t.size
        at_risk = n
        survival = 1.0
        out_t: list[float] = []
        out_s: list[float] = []
        i = 0
        while i < n:
            j = i
            deaths = 0
            censored = 0
            while j < n and t[j] == t[i]:
                if e[j]:
                    deaths += 1
                else:
                    censored += 1
                j += 1
            if deaths > 0 and at_risk > 0:
                survival *= 1.0 - deaths / at_risk
            # events are removed before censorings at the same time
            at_risk -= deaths + censored
            out_t.append(float(t[i]))
            out_s.append(survival)
            i = j
        return np.array(out_t), np.array(out_s)

    def restricted_mean(self) -> float:
        """``E[min(T, horizon)]`` from the Kaplan-Meier step function, integrated exactly."""

        order = np.argsort(self.time, kind="mergesort")
        t = self.time[order]
        e = self.event[order]
        n = t.size
        at_risk = n
        survival = 1.0
        total = 0.0
        previous = 0.0
        i = 0
        while i < n:
            j = i
            deaths = 0
            censored = 0
            while j < n and t[j] == t[i]:
                if e[j]:
                    deaths += 1
                else:
                    censored += 1
                j += 1
            step_time = float(t[i])
            if step_time > previous:
                total += survival * (step_time - previous)
            if deaths > 0 and at_risk > 0:
                survival *= 1.0 - deaths / at_risk
            at_risk -= deaths + censored
            previous = step_time
            i = j
        if self.horizon > previous:
            total += survival * (self.horizon - previous)
        return float(total)

    def restricted_mean_reference(self) -> float:
        """Sample mean of ``min(T, H)``; equal to the RMST under administrative censoring."""

        if not self.administratively_censored:
            raise DomainError("the sample mean of min(T, H) equals the RMST only under administrative censoring")
        return float(np.mean(self.time))

    def median(self) -> float:
        """Kaplan-Meier median, or ``nan`` if the survival never reaches 1/2."""

        times, values = self._km_steps()
        below = np.nonzero(values <= 0.5)[0]
        if below.size == 0:
            return float("nan")
        return float(times[below[0]])


def common_horizon(*laws: ObservedLaw) -> float:
    """The smallest follow-up horizon among ``laws``."""

    if not laws:
        raise DomainError("no laws supplied")
    return float(min(law.horizon for law in laws))


def observed_law_from_times(times: np.ndarray, horizon: float) -> ObservedLaw:
    """Observed law from event times, with ``inf`` (or any time beyond ``horizon``) censored at ``horizon``."""

    times = np.asarray(times, dtype=float)
    event = np.isfinite(times)
    observed = np.minimum(np.where(event, times, horizon), horizon)
    return ObservedLaw(observed, event & (times <= horizon), horizon)


def _normal_quantile(p: float) -> float:
    """Standard normal quantile by bisection."""

    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + erf(mid / sqrt(2.0)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rmst_difference(
    a: ObservedLaw,
    b: ObservedLaw,
    *,
    rng: np.random.Generator | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> dict[str, float | str]:
    """``RMST(a) - RMST(b)`` on the common horizon with a ``1 - alpha`` confidence interval.

    Under administrative censoring of both laws the interval is the normal
    interval for a difference of sample means; otherwise it is a Kaplan-Meier
    bootstrap with ``n_boot`` resamples (which requires ``rng``).
    """

    horizon = common_horizon(a, b)
    a_t, b_t = a.truncate(horizon), b.truncate(horizon)
    observed_diff = float(a_t.restricted_mean() - b_t.restricted_mean())
    if a_t.administratively_censored and b_t.administratively_censored:
        va = float(np.var(a_t.time, ddof=1)) / a_t.n
        vb = float(np.var(b_t.time, ddof=1)) / b_t.n
        se = float(np.sqrt(va + vb))
        z = float(_normal_quantile(1 - alpha / 2))
        return {
            "horizon": horizon,
            "method": "normal_ci_sample_means",
            "rmst_a": a_t.restricted_mean_reference(),
            "rmst_b": b_t.restricted_mean_reference(),
            "difference": observed_diff,
            "se": se,
            "ci_low": observed_diff - z * se,
            "ci_high": observed_diff + z * se,
        }
    if rng is None:
        raise DomainError("a generator is required for the Kaplan-Meier bootstrap")
    sa = np.empty(int(n_boot))
    sb = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        pick = rng.integers(0, a_t.n, size=a_t.n)
        sa[i] = ObservedLaw(a_t.time[pick], a_t.event[pick], horizon).restricted_mean()
        pick = rng.integers(0, b_t.n, size=b_t.n)
        sb[i] = ObservedLaw(b_t.time[pick], b_t.event[pick], horizon).restricted_mean()
    diff = sa - sb
    return {
        "horizon": horizon,
        "method": "kaplan_meier_bootstrap",
        "rmst_a": a_t.restricted_mean(),
        "rmst_b": b_t.restricted_mean(),
        "difference": observed_diff,
        "se": float(diff.std(ddof=1)),
        "ci_low": float(np.quantile(diff, alpha / 2)),
        "ci_high": float(np.quantile(diff, 1 - alpha / 2)),
    }


def equivalence_verdict(ci_low: float, ci_high: float, margin: float) -> dict[str, object]:
    """Classify a confidence interval against a practical margin.

    ``differs_from_zero``: the interval excludes zero.
    ``equivalent_within_margin``: the whole interval lies inside ``(-margin, margin)``.
    ``exceeds_margin``: the interval lies entirely outside ``[-margin, margin]``.
    ``verdict``: one of ``equivalent_within_margin``, ``differs_beyond_margin``,
    ``differs_within_margin_band`` or ``indeterminate``.
    """

    if margin <= 0:
        raise DomainError("margin must be positive")
    if not (ci_low <= ci_high):
        raise DomainError("ci_low must not exceed ci_high")
    differs = bool(ci_low > 0.0 or ci_high < 0.0)
    within = bool(ci_low > -margin and ci_high < margin)
    beyond = bool(ci_low > margin or ci_high < -margin)
    if within:
        verdict = "equivalent_within_margin"
    elif beyond:
        verdict = "differs_beyond_margin"
    elif differs:
        verdict = "differs_within_margin_band"
    else:
        verdict = "indeterminate"
    return {
        "differs_from_zero": differs,
        "equivalent_within_margin": within,
        "exceeds_margin": beyond,
        "verdict": verdict,
    }


def power_positive_control(
    law: ObservedLaw,
    *,
    rng: np.random.Generator,
    shift_fraction: float = 0.2,
    n_boot: int = 500,
    n_reps: int = 40,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Detection rate and false-positive rate of the comparison on a known effect.

    Both arms are drawn from an exponential law with the same Kaplan-Meier median as
    ``law``, censored at the same horizon and with the same sample size; in the
    alternative arm the scale is multiplied by ``1 + shift_fraction``. The
    comparison of :func:`rmst_difference` is applied to ``n_reps`` alternative and
    ``n_reps`` null pairs.
    """

    if not 0 < shift_fraction < 1:
        raise DomainError("shift_fraction must be in (0, 1)")
    if n_reps <= 0:
        raise DomainError("n_reps must be positive")
    horizon = float(law.horizon)
    median = law.median()
    if not np.isfinite(median) or median <= 0:
        return {
            "status": "not_calibrated",
            "reason": "the observed law never reaches a finite Kaplan-Meier median",
            "horizon": horizon,
            "shift_fraction": shift_fraction,
        }
    scale = median / float(np.log(2.0))
    detected = 0
    false_positive = 0
    representative: dict[str, float | str] = {}
    for rep in range(int(n_reps)):
        a = _exponential_observed_law(rng, law.n, scale, horizon)
        b = _exponential_observed_law(rng, law.n, scale * (1.0 + shift_fraction), horizon)
        test = rmst_difference(a, b, rng=rng, n_boot=n_boot, alpha=alpha)
        verdict = equivalence_verdict(float(test["ci_low"]), float(test["ci_high"]), median * 1e-9)
        if verdict["differs_from_zero"]:
            detected += 1
        if rep == 0:
            representative = {
                "difference": float(test["difference"]),
                "ci_low": float(test["ci_low"]),
                "ci_high": float(test["ci_high"]),
                "method": str(test["method"]),
            }
        null_a = _exponential_observed_law(rng, law.n, scale, horizon)
        null_b = _exponential_observed_law(rng, law.n, scale, horizon)
        null_test = rmst_difference(null_a, null_b, rng=rng, n_boot=n_boot, alpha=alpha)
        null_verdict = equivalence_verdict(
            float(null_test["ci_low"]), float(null_test["ci_high"]), median * 1e-9
        )
        if null_verdict["differs_from_zero"]:
            false_positive += 1
    return {
        "status": "calibrated",
        "reference_family": "exponential with the observed Kaplan-Meier median, censored at the observed horizon",
        "horizon": horizon,
        "median": median,
        "shift_fraction": shift_fraction,
        "n_reps": int(n_reps),
        "n_per_arm": law.n,
        "power_detected_fraction": detected / float(n_reps),
        "false_positive_fraction": false_positive / float(n_reps),
        "alpha": alpha,
        "representative_alternative": representative,
    }


def _exponential_observed_law(rng: np.random.Generator, n: int, scale: float, horizon: float) -> ObservedLaw:
    """``min(Exp(scale), horizon)`` with administrative censoring."""

    draws = rng.exponential(scale=scale, size=int(n))
    event = draws <= horizon
    return ObservedLaw(np.where(event, draws, horizon), event, horizon)
