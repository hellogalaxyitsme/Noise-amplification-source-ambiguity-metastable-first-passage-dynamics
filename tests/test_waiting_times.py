"""Marker parsing, life-table hazard and the checksum manifest."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from metastable_fpt.data_download import read_manifest, verify_markers
from metastable_fpt.waiting_times import CLOCK_PERIOD_S, life_table_hazard, subject_waits


def _write_fixture(tmp_path, markers):
    (tmp_path / "P99_RespRP.vhdr").write_text("SamplingInterval=400.0\n", encoding="utf-8")
    lines = ["Brain Vision Data Exchange Marker File, Version 1.0", "[Marker Infos]", "Mk1=New Segment,,1,1,0"]
    for i, (code, sample) in enumerate(markers, start=2):
        lines.append(f"Mk{i}=Stimulus,{code},{sample},1,0")
    (tmp_path / "P99_RespRP.vmrk").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / "P99_RespRP.vhdr"


def test_waiting_time_parser(tmp_path):
    fs = 2500
    markers = [("S 24", 1000), ("S 25", 3500), ("S 20", 3500 + 4 * fs),        # T = 4 s
               ("S 24", 30000), ("S 15", 31000),                                  # aborted trial
               ("S 24", 40000), ("S 25", 42500), ("S 30", 42500 + int(2.8 * fs))]  # T = 2.8 s
    sw = subject_waits(_write_fixture(tmp_path, markers))
    assert sw.sfreq_hz == pytest.approx(2500.0)
    assert sw.T.tolist() == pytest.approx([4.0, 2.8])
    assert sw.condition.tolist() == ["W-block", "M-block"]
    assert sw.n_aborted == 1
    assert sw.n_clock_starts == 2
    assert sw.W.tolist() == pytest.approx([4.0 - CLOCK_PERIOD_S, 2.8 - CLOCK_PERIOD_S])


def test_life_table_hazard_of_exponential_is_constant():
    rng = np.random.default_rng(3)
    x = rng.exponential(2.0, size=200000)
    h = life_table_hazard(x, np.arange(0, 6.01, 0.5))
    # rate 0.5 and bin width 0.5: the discrete hazard is (1 - e^{-0.25}) / 0.5 in every bin
    assert np.nanmax(np.abs(h - (1 - np.exp(-0.25)) / 0.5)) < 0.025


def test_manifest_verification(tmp_path):
    (tmp_path / "a.vmrk").write_bytes(b"abc")
    manifest = tmp_path / "m.sha256"
    manifest.write_text(f"{hashlib.sha256(b'abc').hexdigest()}  a.vmrk\n{'0' * 64}  b.vmrk\n", encoding="utf-8")
    assert set(read_manifest(manifest)) == {"a.vmrk", "b.vmrk"}
    assert verify_markers(tmp_path, manifest) == ["b.vmrk: missing"]
