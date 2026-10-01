"""Euler-Maruyama first-passage simulation with additive streaming forcings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .seeding import DomainError, check_generator


@dataclass(frozen=True)
class CrossingResult:
    """First-passage times (``inf`` for trials still active at the horizon)."""

    times: np.ndarray
    n_steps_run: int

    @property
    def n_events(self) -> int:
        return int(np.isfinite(self.times).sum())

    @property
    def n_paths(self) -> int:
        return int(self.times.size)


def simulate_crossing(
    *,
    drift: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    x0: float | np.ndarray,
    dt: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    white_D: float = 0.0,
    forcings: Sequence[tuple[str, object]] | None = None,
    rng_by_forcing: dict[str, np.random.Generator] | None = None,
    lower: float | None = None,
) -> CrossingResult:
    """Upward crossing times of ``threshold`` for ``dx = drift(x) dt + sqrt(2 white_D) dW + sum of forcings``.

    ``forcings`` is a sequence of ``(name, sampler)`` pairs; each sampler exposes
    ``next_increments(rng)`` and receives the generator ``rng_by_forcing[name]``
    (``rng`` if absent). ``rng`` drives the white-noise increments. A path that
    falls below ``lower`` is reflected. The crossing time inside a step is found by
    linear interpolation of the Euler step.
    """

    check_generator(rng)
    if dt <= 0 or n_steps <= 0 or n_paths <= 0:
        raise DomainError("dt, n_steps and n_paths must be positive")
    if white_D < 0:
        raise DomainError("white_D must be non-negative")
    if np.isscalar(x0):
        x = np.full(n_paths, float(x0))
    else:
        x = np.asarray(x0, dtype=float).copy()
        if x.shape != (n_paths,):
            raise DomainError(f"x0 must be scalar or shape ({n_paths},)")
    times = np.full(n_paths, np.inf)
    active = np.ones(n_paths, dtype=bool)
    rng_by_forcing = dict(rng_by_forcing or {})
    white_scale = float(np.sqrt(2.0 * white_D))
    for k in range(1, n_steps + 1):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        step = drift(x) * dt
        if white_D > 0:
            step = step + white_scale * np.sqrt(dt) * rng.standard_normal(n_paths)
        for name, sampler in forcings or ():
            step = step + np.asarray(sampler.next_increments(rng_by_forcing.get(name, rng)), dtype=float)
        x = x + step
        if lower is not None:
            x = np.where(x < lower, 2.0 * lower - x, x)
        crossed = active & (x >= threshold)
        if crossed.any():
            local = np.nonzero(crossed)[0]
            previous = x[local] - step[local]
            denom = step[local]
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(denom != 0, (threshold - previous) / denom, 0.0)
            frac = np.clip(np.where(np.isfinite(frac), frac, 0.0), 0.0, 1.0)
            times[local] = (k - 1) * dt + frac * dt
            active[local] = False
    return CrossingResult(times=times, n_steps_run=n_steps)
