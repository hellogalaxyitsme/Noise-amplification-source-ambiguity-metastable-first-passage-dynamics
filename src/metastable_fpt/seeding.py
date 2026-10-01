"""Input validation and reproducible random-number streams."""

from __future__ import annotations

import hashlib

import numpy as np


class DomainError(ValueError):
    """Raised when an input lies outside the documented domain of a function."""


def stream_key(name: str) -> int:
    """Stable non-negative integer derived from a stream label (first 6 bytes of SHA-256)."""

    if not isinstance(name, str) or not name:
        raise DomainError("seed stream name must be a non-empty string")
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:6], "big")


def make_generator(master_seed: int, stream: str) -> np.random.Generator:
    """Independent generator for a labelled stream under a master seed.

    The generator is ``default_rng(SeedSequence(master_seed, spawn_key=(stream_key(stream),)))``,
    so every labelled stream is reproducible and statistically independent of the
    others. No global random state is used anywhere in the package.
    """

    if not isinstance(master_seed, (int, np.integer)) or master_seed <= 0:
        raise DomainError("master_seed must be a positive integer")
    ss = np.random.SeedSequence(entropy=int(master_seed), spawn_key=(stream_key(stream),))
    return np.random.default_rng(ss)


def check_generator(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise DomainError("expected a numpy.random.Generator")
    return rng
