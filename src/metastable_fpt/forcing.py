"""Streaming samplers for colored, telegraph and jump forcings.

Convention: for a forcing ``eta`` the contribution to the commitment diffusion is
``D = int_0^inf C_eta(s) ds``, half the integrated variance per unit time. White
noise ``sqrt(2D) dW`` contributes ``D``; an OU forcing with stationary variance
``D/tau`` contributes ``D``; a telegraph forcing of amplitude ``a`` switching at
rate ``r`` contributes ``a^2/(2r)``, so ``a = sqrt(2 D r)``; compound-Poisson jumps
with rate ``lambda`` and amplitude variance ``sigma_A^2`` contribute
``lambda sigma_A^2 / 2``.

Each sampler advances its state one time step at a time and returns the exact
integral of the forcing over the step for all paths, so a simulation never holds
an ``(n_paths, n_steps)`` array.
"""

from __future__ import annotations

import numpy as np

from .seeding import DomainError, check_generator


def analytic_integrated_variance(D: float, tau: float, horizon: float) -> float:
    """``Var[int_0^H eta dt]`` for a stationary forcing with autocovariance ``(D/tau) e^{-|s|/tau}``.

    ``Var = 2 D tau (H/tau - 1 + e^{-H/tau})``. This covers the OU forcing and the
    telegraph forcing with ``tau = 1/(2r)``.
    """

    if D <= 0 or tau <= 0 or horizon < 0:
        raise DomainError("D and tau must be positive and horizon non-negative")
    x = horizon / tau
    return float(2.0 * D * tau * (x - 1.0 + np.exp(-x)))


def analytic_lag1_state_correlation(rate: float, dt: float) -> float:
    """Correlation of the telegraph state at two times ``dt`` apart: ``exp(-2 rate dt)``."""

    if rate <= 0 or dt <= 0:
        raise DomainError("rate and dt must be positive")
    return float(np.exp(-2.0 * rate * dt))


def analytic_telegraph_increment_variance(D: float, rate: float, dt: float) -> float:
    """``Var[I_h]`` of the telegraph integral over one step ``h``.

    With ``a^2 = 2 D r`` and ``lambda = 2 r``:
    ``Var[I_h] = 2 a^2 [h/lambda - (1 - e^{-lambda h}) / lambda^2]``.
    """

    if D <= 0 or rate <= 0 or dt <= 0:
        raise DomainError("D, rate and dt must be positive")
    a2 = 2.0 * D * rate
    lam = 2.0 * rate
    return float(2.0 * a2 * (dt / lam - (1.0 - np.exp(-lam * dt)) / lam**2))


def analytic_telegraph_increment_covariance(D: float, rate: float, dt: float) -> float:
    """``Cov[I_k, I_{k+1}] = a^2 (1 - e^{-lambda h})^2 / lambda^2`` for adjacent step integrals."""

    if D <= 0 or rate <= 0 or dt <= 0:
        raise DomainError("D, rate and dt must be positive")
    a2 = 2.0 * D * rate
    lam = 2.0 * rate
    return float(a2 * (1.0 - np.exp(-lam * dt)) ** 2 / lam**2)


def analytic_telegraph_increment_correlation(D: float, rate: float, dt: float) -> float:
    """Correlation of adjacent telegraph step integrals."""

    var = analytic_telegraph_increment_variance(D, rate, dt)
    if var <= 0:
        raise DomainError("increment variance must be positive")
    return float(analytic_telegraph_increment_covariance(D, rate, dt) / var)


def exact_flip_probability(rate: float, dt: float) -> float:
    """Probability that the telegraph state differs after ``dt``: ``(1 - e^{-2 rate dt}) / 2``."""

    if rate <= 0 or dt <= 0:
        raise DomainError("rate and dt must be positive")
    return float((1.0 - np.exp(-2.0 * rate * dt)) / 2.0)


class OUStream:
    """OU forcing with long-time diffusion ``D`` and correlation time ``tau``.

    The pair ``(eta, int eta dt)`` is sampled exactly from its bivariate Gaussian
    transition over each step. The initial state is drawn from the stationary law
    ``N(0, D/tau)`` with the generator passed at construction.
    """

    def __init__(self, *, D: float, tau: float, dt: float, n_paths: int, rng: np.random.Generator) -> None:
        if D <= 0 or tau <= 0 or dt <= 0 or n_paths <= 0:
            raise DomainError("D, tau, dt and n_paths must be positive")
        check_generator(rng)
        self.D, self.tau, self.dt, self.n = float(D), float(tau), float(dt), int(n_paths)
        self.theta = 1.0 / self.tau
        self.a = float(np.exp(-self.theta * self.dt))
        sigma2 = 2.0 * self.D * self.theta**2
        var_eta = sigma2 * (1.0 - self.a**2) / (2.0 * self.theta)
        var_j = (sigma2 / self.theta**2) * (
            self.dt - 2.0 * (1.0 - self.a) / self.theta + (1.0 - self.a**2) / (2.0 * self.theta)
        )
        cov_ej = (sigma2 / (2.0 * self.theta**2)) * (1.0 - self.a) ** 2
        cov = np.array([[max(var_eta, 0.0), cov_ej], [cov_ej, max(var_j, 0.0)]])
        eig_min = float(np.min(np.linalg.eigvalsh(cov)))
        if eig_min < -1e-12 * max(1.0, float(np.max(np.abs(cov)))):
            raise DomainError("OU increment covariance is not positive semidefinite")
        self.chol = np.linalg.cholesky(cov + np.eye(2) * max(0.0, -eig_min))
        self.state = rng.standard_normal(self.n) * np.sqrt(self.D / self.tau)
        self.mean_factor = (1.0 - self.a) / self.theta

    def next_increments(self, rng: np.random.Generator) -> np.ndarray:
        """Integral of the forcing over the next step; advances the state."""

        check_generator(rng)
        draws = rng.standard_normal((self.n, 2)) @ self.chol.T
        delta_eta = (self.a - 1.0) * self.state + draws[:, 0]
        integral = self.mean_factor * self.state + draws[:, 1]
        self.state = self.state + delta_eta
        return integral


class TelegraphStream:
    """Telegraph forcing ``+-sqrt(2 D rate)`` switching at rate ``rate``.

    The switch times inside each step are sampled and the piecewise-constant path
    is integrated exactly; the end-of-step state is the start state flipped once
    per sampled switch. The initial sign is ``+1`` or ``-1`` with probability 1/2.
    """

    def __init__(self, *, D: float, rate: float, dt: float, n_paths: int, rng: np.random.Generator) -> None:
        if D <= 0 or rate <= 0 or dt <= 0 or n_paths <= 0:
            raise DomainError("D, rate, dt and n_paths must be positive")
        check_generator(rng)
        self.D, self.rate, self.dt, self.n = float(D), float(rate), float(dt), int(n_paths)
        self.amplitude = float(np.sqrt(2.0 * D * rate))
        self.state = rng.choice([-1.0, 1.0], size=self.n) * self.amplitude
        self.mean_switches = self.rate * self.dt

    def next_increments(self, rng: np.random.Generator) -> np.ndarray:
        """Integral of the forcing over the next step; advances the state.

        For a path with ``c`` switches at times ``t_1 < ... < t_c`` inside the step
        and start value ``s`` the integral is
        ``s t_1 + sum_{j=2..c} (-1)^{j-1} s (t_j - t_{j-1}) + (-1)^c s (dt - t_c)``.
        """

        check_generator(rng)
        counts = rng.poisson(self.mean_switches, size=self.n)
        integral = np.zeros(self.n)
        total_switches = int(counts.sum())
        if total_switches == 0:
            integral[:] = self.state * self.dt
        else:
            idx = np.repeat(np.arange(self.n), counts)
            times = rng.random(total_switches) * self.dt
            order = np.lexsort((times, idx))
            idx_sorted = idx[order]
            times_sorted = times[order]
            # local switch index within each path (0-based)
            offsets = np.concatenate([[0], np.cumsum(counts)])[:-1]
            local = np.arange(total_switches) - np.repeat(offsets, counts)
            starts = np.where(local == 0, 0.0, np.concatenate([[0.0], times_sorted[:-1]]))
            pre_lengths = times_sorted - starts
            sign_pre = self.state[idx_sorted] * ((-1.0) ** local)
            integral = np.bincount(idx_sorted, weights=sign_pre * pre_lengths, minlength=self.n)
            # paths without a switch in this step keep their state over the whole step
            no_switch = counts == 0
            integral[no_switch] = self.state[no_switch] * self.dt
            # segment from the last switch of each path to the end of the step
            last_time = np.zeros(self.n)
            np.maximum.at(last_time, idx_sorted, times_sorted)
            switched = counts > 0
            sign_after = self.state * ((-1.0) ** counts)
            integral[switched] += sign_after[switched] * (self.dt - last_time[switched])
        self.state = self.state * (((-1.0) ** counts))
        return integral


class CompoundPoissonStream:
    """Zero-mean Gaussian-amplitude jumps at rate ``rate`` with ``rate sigma_A^2 = 2 D``.

    ``D`` is the long-time diffusion contributed by the jumps; each step returns the
    sum of the jumps that occur in it.
    """

    def __init__(self, *, D: float, rate: float, dt: float, n_paths: int) -> None:
        if D <= 0 or rate <= 0 or dt <= 0 or n_paths <= 0:
            raise DomainError("D, rate, dt and n_paths must be positive")
        self.D, self.rate, self.dt, self.n = float(D), float(rate), float(dt), int(n_paths)
        self.sigma_a = float(np.sqrt(2.0 * D / rate))
        self.mean_jumps = self.rate * self.dt

    def next_increments(self, rng: np.random.Generator) -> np.ndarray:
        check_generator(rng)
        counts = rng.poisson(self.mean_jumps, size=self.n)
        total = int(counts.sum())
        out = np.zeros(self.n)
        if total:
            amps = rng.standard_normal(total) * self.sigma_a
            idx = np.repeat(np.arange(self.n), counts)
            np.add.at(out, idx, amps)
        return out
