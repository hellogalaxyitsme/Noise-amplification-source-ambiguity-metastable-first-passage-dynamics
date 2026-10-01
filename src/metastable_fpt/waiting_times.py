"""Clock-start-to-press waiting times from the Libet-clock dataset of Jeay-Bizot et al. (2026).

Task: a dot circles a clock dial once every 2.56 s. Participants wait one full
revolution and then press a button whenever they wish, avoiding pre-planning.
Blocks of trials alternate between two report conditions.

BrainVision marker codes (``.vmrk``):

* ``S 24``: trial start (fixation);
* ``S 25``: clock start, where the waiting interval begins;
* ``S 20`` / ``S 30``: button press in the two report conditions;
* ``S 15``: aborted trial.

A waiting time is recorded only when the marker immediately after ``S 25`` is a
press. ``T`` is the clock-start-to-press time and ``W = T - 2.56 s`` the free
waiting time after the instructed minimum. Only the marker and header files are
read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .seeding import DomainError

CLOCK_PERIOD_S = 2.56
PRESS_CODES = {"S 20": "W-block", "S 30": "M-block"}


@dataclass
class SubjectWaits:
    subject: str
    sfreq_hz: float
    T: np.ndarray            # clock start to press (s)
    condition: np.ndarray    # report condition of each trial
    n_aborted: int
    n_clock_starts: int

    @property
    def W(self) -> np.ndarray:
        return self.T - CLOCK_PERIOD_S


def read_sfreq(vhdr: Path) -> float:
    """Sampling frequency (Hz) from the ``SamplingInterval`` entry (microseconds) of a header file."""

    text = Path(vhdr).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"SamplingInterval\s*=\s*([0-9.]+)", text)
    if not m:
        raise DomainError(f"{vhdr}: no SamplingInterval")
    return 1e6 / float(m.group(1))


def parse_vmrk(vmrk: Path) -> list[tuple[str, int]]:
    """``(code, sample)`` for every stimulus marker, in file order."""

    events = []
    for line in Path(vmrk).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"Mk\d+=([^,]*),([^,]*),(\d+)", line)
        if m and m.group(1) == "Stimulus":
            events.append((m.group(2).strip(), int(m.group(3))))
    return events


def subject_waits(vhdr: Path) -> SubjectWaits:
    vhdr = Path(vhdr)
    vmrk = vhdr.with_suffix(".vmrk")
    sf = read_sfreq(vhdr)
    ev = parse_vmrk(vmrk)
    T, cond = [], []
    for (c, s), (c2, s2) in zip(ev[:-1], ev[1:]):
        if c == "S 25" and c2 in PRESS_CODES:
            T.append((s2 - s) / sf)
            cond.append(PRESS_CODES[c2])
    if not T:
        raise DomainError(f"{vhdr.name}: no clock-start to press pairs")
    return SubjectWaits(
        subject=vhdr.stem.split("_")[0],
        sfreq_hz=sf,
        T=np.asarray(T),
        condition=np.asarray(cond),
        n_aborted=sum(1 for c, _ in ev if c == "S 15"),
        n_clock_starts=sum(1 for c, _ in ev if c == "S 25"),
    )


def discover(root: str | Path) -> list[Path]:
    """Header files ``P*_RespRP.vhdr`` under ``root``, sorted by name."""

    files = sorted(Path(root).glob("P*_RespRP.vhdr"))
    if not files:
        raise DomainError(f"no P*_RespRP.vhdr under {root}")
    return files


def bootstrap_mean_cv(x: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict[str, list[float]]:
    """Percentile 95 % intervals of the mean and the coefficient of variation (trial resampling)."""

    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    xb = x[idx]
    m = xb.mean(axis=1)
    cv = xb.std(axis=1, ddof=1) / m
    return {"mean_ci": np.quantile(m, [0.025, 0.975]).tolist(), "cv_ci": np.quantile(cv, [0.025, 0.975]).tolist()}


def life_table_hazard(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Discrete hazard per bin: events in the bin / (number at risk at its start x bin width).

    Bins with fewer than five trials at risk are ``nan``.
    """

    x = np.asarray(x, dtype=float)
    out = np.full(edges.size - 1, np.nan)
    for i in range(edges.size - 1):
        at_risk = np.sum(x >= edges[i])
        if at_risk >= 5:
            ev = np.sum((x >= edges[i]) & (x < edges[i + 1]))
            out[i] = ev / (at_risk * (edges[i + 1] - edges[i]))
    return out
