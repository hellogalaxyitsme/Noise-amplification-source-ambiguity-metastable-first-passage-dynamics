"""Writing analysis results as JSON."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy


def jsonable(value: Any) -> Any:
    """Recursively convert numpy containers and scalars to plain Python objects."""

    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def environment() -> dict[str, str]:
    from . import __version__

    return {
        "metastable_fpt": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
