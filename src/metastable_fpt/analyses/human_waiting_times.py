"""Self-paced waiting times and the bound on noise amplification (Section VI of the article).

Uses only the header and marker files of the Libet-clock dataset (see
:mod:`metastable_fpt.waiting_times`). Per participant:

* mean, standard deviation and coefficient of variation of the clock-start-to-press
  time ``T`` and of the free waiting time ``W = T - 2.56 s`` (``W > 0``), with
  bootstrap 95 % intervals, and the life-table hazard of ``T``;
* the Theorem 6 bound on the elasticity ``s`` over a grid of the unobservable
  drift time ``tau_L``, from the point estimates and, conservatively, from the
  upper bootstrap limits of the CV and the mean;
* a maximum-likelihood fit of the leaky accumulator to ``T`` and a profile
  likelihood over the fixed-point ratio ``rho = I/k``; a value of ``rho`` is
  compatible with the data when its profile negative log-likelihood is within
  1.92 of the minimum.

Group summaries combine these over participants.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..accumulator import fit_accumulator, profile_fixed_point
from ..elasticity import elasticity_bound
from ..seeding import make_generator
from ..waiting_times import CLOCK_PERIOD_S, bootstrap_mean_cv, discover, life_table_hazard, subject_waits

NAME = "human_waiting_times"
DESCRIPTION = "Waiting-time variability, the elasticity bound and accumulator fits for 17 participants"


def parameters(quick: bool = False) -> dict:
    return {
        "master_seed": 20260930,
        "n_boot_stats": 500 if quick else 2000,
        "n_x_fit": 120 if quick else 160,
        "min_trials": 30,
        "hazard_bin_s": 0.5,
        "hazard_max_s": 20.0,
        "relative_change_r": 0.01,
        "profile_threshold": 1.92,
        "tau_L_grid_s": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "tau_L_reference_s": [1e-3, 1e-2],
        "rho_grid": [0.5, 0.8, 1.0, 1.5, 2.5] if quick else [0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.1, 1.3, 1.6, 2.0, 3.0],
    }


def _fit_job(args):
    subject, T, n_x, rho_grid, threshold = args
    fit = fit_accumulator(T, n_x=n_x)
    prof = profile_fixed_point(T, rho_grid, n_x=n_x, start=fit)
    best = min([fit.nll] + [r["nll"] for r in prof])
    for r in prof:
        r["delta_nll"] = r["nll"] - best
        r["compatible_95"] = bool(r["delta_nll"] <= threshold)
    return subject, fit.as_dict(), prof


def run(p: dict, *, data_dir: str | Path | None = None, workers: int | None = None, **_: object) -> dict:
    if data_dir is None:
        raise ValueError("data_dir (directory with P*_RespRP.vhdr/.vmrk) is required")
    files = discover(data_dir)
    rng = make_generator(p["master_seed"], "bootstrap")
    tau_grid = np.asarray(p["tau_L_grid_s"])
    subjects = []
    edges = np.arange(0.0, p["hazard_max_s"] + 1e-9, p["hazard_bin_s"])
    for f in files:
        sw = subject_waits(f)
        if sw.T.size < p["min_trials"]:
            continue
        T = sw.T
        Wp = sw.W[sw.W > 0]
        row = {
            "subject": sw.subject,
            "sfreq_hz": sw.sfreq_hz,
            "n_trials": int(T.size),
            "n_aborted": sw.n_aborted,
            "n_clock_starts": sw.n_clock_starts,
            "n_W_nonpositive": int(np.sum(sw.W <= 0)),
            "T": {"mean": float(T.mean()), "sd": float(T.std(ddof=1)), "cv": float(T.std(ddof=1) / T.mean()),
                  "min": float(T.min()), "max": float(T.max()), **bootstrap_mean_cv(T, rng, int(p["n_boot_stats"]))},
            "W": {"n": int(Wp.size), "mean": float(Wp.mean()), "sd": float(Wp.std(ddof=1)),
                  "cv": float(Wp.std(ddof=1) / Wp.mean()), **bootstrap_mean_cv(Wp, rng, int(p["n_boot_stats"]))},
            "by_condition": {
                c: {"n": int(np.sum(sw.condition == c)), "mean_T": float(T[sw.condition == c].mean()),
                    "cv_T": float(T[sw.condition == c].std(ddof=1) / T[sw.condition == c].mean())}
                for c in np.unique(sw.condition)
            },
            "hazard_T": life_table_hazard(T, edges).tolist(),
        }
        for key in ("T", "W"):
            st = row[key]
            st["bound_by_tau_L"] = elasticity_bound(cv=st["cv"], mean=st["mean"], tau_L=tau_grid).tolist()
            st["bound_upper_by_tau_L"] = elasticity_bound(cv=st["cv_ci"][1], mean=st["mean_ci"][1],
                                                          tau_L=tau_grid).tolist()
        subjects.append((row, T))

    jobs = [(r["subject"], T, int(p["n_x_fit"]), p["rho_grid"], p["profile_threshold"]) for r, T in subjects]
    n_workers = workers or min(len(jobs), os.cpu_count() or 1)
    fits = {}
    with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as ex:
        for subject, fit, prof in ex.map(_fit_job, jobs):
            comp = [r for r in prof if r["compatible_95"] and np.isfinite(r["elasticity"])]
            fits[subject] = {
                "fit": fit,
                "profile": prof,
                "max_elasticity_compatible_95": max([fit["elasticity"]] + [r["elasticity"] for r in comp]),
                "min_rho_compatible_95": min([fit["fixed_point_over_threshold"]] + [r["rho"] for r in comp]),
                "noise_driven_compatible_95": bool(any(r["rho"] < 1 for r in comp)),
            }
    rows = [r for r, _ in subjects]
    for r in rows:
        r["accumulator"] = fits.get(r["subject"])

    ref = p["tau_L_reference_s"]
    iref = [int(np.argmin(np.abs(tau_grid - x))) for x in ref]
    cvT = np.array([r["T"]["cv"] for r in rows])
    cvW = np.array([r["W"]["cv"] for r in rows])
    bT = np.array([r["T"]["bound_upper_by_tau_L"] for r in rows])
    bW = np.array([r["W"]["bound_upper_by_tau_L"] for r in rows])
    s_fit = np.array([r["accumulator"]["fit"]["elasticity"] for r in rows])
    fp = np.array([r["accumulator"]["fit"]["fixed_point_over_threshold"] for r in rows])
    s_comp = np.array([r["accumulator"]["max_elasticity_compatible_95"] for r in rows])
    rr = p["relative_change_r"]
    group = {
        "n_subjects": len(rows),
        "n_trials_total": int(sum(r["n_trials"] for r in rows)),
        "n_clock_starts_total": int(sum(r["n_clock_starts"] for r in rows)),
        "n_aborted_total": int(sum(r["n_aborted"] for r in rows)),
        "cv_T_median_min_max": [float(np.median(cvT)), float(cvT.min()), float(cvT.max())],
        "cv_W_median_min_max": [float(np.median(cvW)), float(cvW.min()), float(cvW.max())],
        "mean_T_median_s": float(np.median([r["T"]["mean"] for r in rows])),
        "mean_W_median_s": float(np.median([r["W"]["mean"] for r in rows])),
        "subjects_with_cv_T_upper_ci_below_1": int(sum(r["T"]["cv_ci"][1] < 1 for r in rows)),
        "subjects_with_cv_W_upper_ci_below_1": int(sum(r["W"]["cv_ci"][1] < 1 for r in rows)),
        "bound_upper_max_over_subjects_by_tau_L": {
            "tau_L_s": tau_grid.tolist(),
            "T_one_stage": bT.max(axis=0).tolist(),
            "T_two_stage_plus_log2": (bT.max(axis=0) + np.log(2)).tolist(),
            "W_one_stage": bW.max(axis=0).tolist(),
        },
        "bound_at_reference_tau_L": {
            f"{ref[j]:g}s": {"T_median": float(np.median(bT[:, i])), "T_max": float(bT[:, i].max()),
                             "W_max": float(bW[:, i].max()), "T_two_stage_max": float(bT[:, i].max() + np.log(2))}
            for j, i in enumerate(iref)
        },
        "fitted_elasticity_median_min_max": [float(np.median(s_fit)), float(s_fit.min()), float(s_fit.max())],
        "max_elasticity_compatible_95_median_min_max": [
            float(np.median(s_comp)), float(s_comp.min()), float(s_comp.max())],
        "subjects_noise_driven_regime_compatible_95": int(sum(r["accumulator"]["noise_driven_compatible_95"] for r in rows)),
        "fitted_fixed_point_ratio_median_min_max": [float(np.median(fp)), float(fp.min()), float(fp.max())],
        "subjects_drift_dominated_I_over_k_gt_1": int(np.sum(fp > 1)),
        "max_relative_timing_change": {
            "relative_change_r": rr,
            "two_stage_bound_at_smallest_tau_L": float(rr * (bT.max(axis=0)[0] + np.log(2))),
            "fitted_models": float(rr * s_fit.max()),
            "profile_compatible": float(rr * s_comp.max()),
        },
    }
    return {
        "data_dir": str(data_dir),
        "clock_period_s": CLOCK_PERIOD_S,
        "subjects": rows,
        "group": group,
    }
