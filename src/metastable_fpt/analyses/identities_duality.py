"""Weighted-barrier identities, the variability-amplification bound and the duality (Sections IV-V).

Four blocks:

1. Identities. On ``U(x) = x^4/4 - x^2/2``, the elasticity of Lemma 4 is compared
   with a central finite difference of the mean, and the CV of Lemma 5 with the
   moments of an independent Markov-chain discretisation.
2. Random potentials. Theorem 6 is evaluated on seeded random potentials
   ``U(x) = c x + sum_{k=1}^5 a_k sin(k f x)`` on ``[-2, 2]`` with random noise
   levels and starting points; the slack ``bound - s`` is reported.
3. Stages. A metastable stage followed by an independent drift-diffusion stage
   whose law does not depend on ``D``; Theorem 6 with ``+ log 2`` is evaluated.
4. Duality. For each mechanism and its escape-matched white twin across barrier
   heights: the per-trial divergence of the waiting times, the per-trial
   divergence of the transition-path durations, and the ratio of the former to
   ``s e^{-s}`` with ``s`` the elasticity of the twin (Theorem 7).
"""

from __future__ import annotations

import numpy as np

from ..elasticity import elasticity_bound, weighted_barrier
from ..generators import (
    Domain,
    ou_generator,
    quartic_gradient,
    telegraph_generator,
    white_generator,
    x_index_of_states,
)
from ..seeding import make_generator
from ..timing_law import TimeGrid, default_t_end, kl_timing, match_rate, moments, spectrum, timing_law
from ..transition_paths import transition_path_generator

NAME = "identities_duality"
DESCRIPTION = "Weighted-barrier identities, the elasticity bound on random potentials, and transition-path information"
BARRIER = 0.25


def parameters(quick: bool = False) -> dict:
    return {
        "master_seed": 20260930,
        "ell": -2.0,
        "bnd": 0.0,
        "x0": -1.0,
        "x_A": -1.0,
        "n_x": 100 if quick else 150,
        "n_x_chain": 400 if quick else 800,
        "n_quad": 20001 if quick else 40001,
        "m_eta": 31 if quick else 41,
        "q_fraction": 0.5,
        "n_random": 100 if quick else 1000,
        "random_n_quad": 20001,
        "t_end_margin": 32.0,
        "tp_margin": 24.0,
        "D_identity_grid": [0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02],
        "barrier_over_D": [2.0, 4.0, 6.0] if quick else [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "mechanisms": ([["ou", 0.5], ["telegraph", 1.0]] if quick
                       else [["ou", 0.1], ["ou", 0.5], ["telegraph", 1.0], ["telegraph", 0.25]]),
        "two_stage": {"D": [0.1, 0.05, 0.03], "v": [0.02, 0.005, 0.001], "theta": 1.0},
    }


def _U(x):
    x = np.asarray(x)
    return x**4 / 4 - x**2 / 2


def _identities(p: dict) -> dict:
    gU = quartic_gradient()
    rows = []
    for D in p["D_identity_grid"]:
        r = weighted_barrier(_U, gU, D=D, x0=p["x0"], ell=p["ell"], b=p["bnd"], n=int(p["n_quad"]))
        g = white_generator(gU, D=D, x0=p["x0"], dom=Domain(p["ell"], p["bnd"], int(p["n_x_chain"])))
        m_c, v_c = moments(g)
        rows.append({
            "D": D,
            "barrier_over_D": BARRIER / D,
            "mean_identity": r.mean,
            "mean_chain": m_c,
            "mean_rel_diff": abs(r.mean - m_c) / m_c,
            "cv_identity": r.cv,
            "cv_chain": float(np.sqrt(v_c) / m_c),
            "cv_rel_diff": abs(r.cv - np.sqrt(v_c) / m_c) / (np.sqrt(v_c) / m_c),
            "s_identity": r.elasticity,
            "s_finite_difference": r.elasticity_finite_difference,
            "s_abs_diff": abs(r.elasticity - r.elasticity_finite_difference),
            "mean_barrier_over_D": r.mean_barrier_over_D,
            "elasticity_bound": r.bound,
            "bound_slack": r.bound - r.elasticity,
            "tau_L": r.tau_L,
        })
    return {"rows": rows,
            "max_s_abs_diff": max(r["s_abs_diff"] for r in rows),
            "max_cv_rel_diff": max(r["cv_rel_diff"] for r in rows),
            "max_mean_rel_diff": max(r["mean_rel_diff"] for r in rows),
            "bound_holds": all(r["s_identity"] <= r["elasticity_bound"] for r in rows)}


def _random_potentials(p: dict) -> dict:
    rng = make_generator(p["master_seed"], "potentials")
    n = int(p["n_random"])
    res = []
    kk = np.arange(1, 6)
    for _ in range(n):
        amp = rng.normal(size=5) * rng.uniform(0.2, 2.0)
        freq = rng.uniform(0.6, 2.0)
        tilt = rng.uniform(-2.0, 2.0)
        U = lambda x, a=amp, f=freq, t=tilt: t * np.asarray(x) + np.sum(a[:, None] * np.sin(np.outer(kk, x) * f), axis=0)  # noqa: E731
        dU = lambda x, a=amp, f=freq, t=tilt: t + np.sum(f * kk[:, None] * a[:, None] * np.cos(np.outer(kk, x) * f), axis=0)  # noqa: E731
        D = float(10 ** rng.uniform(-1.3, 0.5))
        x0 = float(rng.uniform(-2.0, 1.8))
        r = weighted_barrier(U, dU, D=D, x0=x0, ell=-2.0, b=2.0, n=int(p["random_n_quad"]), check_fd=False)
        if not (np.isfinite(r.mean) and np.isfinite(r.elasticity) and np.isfinite(r.bound)):
            continue
        res.append((r.elasticity, r.bound, r.cv, r.bound_log_argument))
    arr = np.array(res)
    slack = arr[:, 1] - arr[:, 0]
    return {
        "n_cases": int(arr.shape[0]),
        "n_violations": int(np.sum(slack < -1e-9)),
        "min_slack": float(slack.min()),
        "median_slack": float(np.median(slack)),
        "slack_quantiles_5_50_95": np.quantile(slack, [0.05, 0.5, 0.95]).tolist(),
        "max_s": float(arr[:, 0].max()),
        "cv_range": [float(arr[:, 2].min()), float(arr[:, 2].max())],
        "elasticity_bound_pairs": arr[:, :2].tolist(),
    }


def _two_stage(p: dict) -> dict:
    """Stage 1: escape in the quartic potential at ``D``; stage 2: drift ``v`` to ``theta`` (mean ``theta/v``)."""

    gU = quartic_gradient()
    ts = p["two_stage"]
    rows = []
    for D in ts["D"]:
        stage1 = weighted_barrier(_U, gU, D=D, x0=p["x0"], ell=p["ell"], b=p["bnd"], n=int(p["n_quad"]))
        for v in ts["v"]:
            th = ts["theta"]
            m2, var2 = th / v, th * 2 * D / v**3
            m = stage1.mean + m2
            var = stage1.var + var2
            cv = float(np.sqrt(var) / m)
            s_T = stage1.elasticity * stage1.mean / m
            bound = float(elasticity_bound(cv=cv, mean=m, tau_L=stage1.tau_L)) + np.log(2.0)
            rows.append({"D": D, "v": v, "rho_1": stage1.mean / m, "s_T": s_T, "cv": cv, "bound_plus_log2": bound,
                         "holds": bool(s_T <= bound)})
    return {"rows": rows, "all_hold": all(r["holds"] for r in rows)}


def _duality(p: dict) -> dict:
    gU = quartic_gradient()
    dom = Domain(p["ell"], p["bnd"], int(p["n_x"]))
    iA = int(np.searchsorted(dom.x, p["x_A"]))
    frac = p["q_fraction"]
    rows = []
    for kind, par in p["mechanisms"]:
        for beta in p["barrier_over_D"]:
            D = BARRIER / beta
            if kind == "ou":
                g = ou_generator(gU, D_C=(1 - frac) * D, D_Q=frac * D, tau=par, x0=p["x0"], dom=dom, m_eta=int(p["m_eta"]))
                name = f"OU tau={par:g}"
            else:
                g = telegraph_generator(gU, D_C=(1 - frac) * D, D_Q=frac * D, rate=par, x0=p["x0"], dom=dom)
                name = f"telegraph r={par:g}"
            mk = lambda DD: white_generator(gU, D=DD, x0=p["x0"], dom=dom)  # noqa: E731
            sg = spectrum(g)
            D_esc, w = match_rate(mk, sg.lam1, D_lo=D / 2.5, D_hi=D * 2.5)
            sw = spectrum(w)
            grid = TimeGrid(dt0=2e-3, steps=80, t_end=max(default_t_end(sg, p["t_end_margin"]),
                                                          default_t_end(sw, p["t_end_margin"])))
            kl_t = kl_timing(timing_law(g, grid, spec=sg), timing_law(w, grid, spec=sw))
            tg = transition_path_generator(g, x_index_of_states(g, dom) <= iA)
            tw = transition_path_generator(w, x_index_of_states(w, dom) <= iA)
            stg, stw = spectrum(tg), spectrum(tw)
            tgrid = TimeGrid(dt0=1e-4, steps=100, t_end=max(default_t_end(stg, p["tp_margin"]),
                                                             default_t_end(stw, p["tp_margin"])))
            Lg, Lw = timing_law(tg, tgrid, spec=stg), timing_law(tw, tgrid, spec=stw)
            kl_tp = kl_timing(Lg, Lw)
            el = weighted_barrier(_U, gU, D=D_esc, x0=p["x0"], ell=p["ell"], b=p["bnd"], n=int(p["n_quad"]),
                                  check_fd=False)
            rows.append({
                "mechanism": name, "barrier_over_D": beta, "D_esc": D_esc, "elasticity_s": el.elasticity,
                "kl_waiting_time": kl_t["kl"], "kl_waiting_time_density_form": kl_t["kl_density_form"],
                "kl_transition_path_duration": kl_tp["kl"], "kl_transition_path_density_form": kl_tp["kl_density_form"],
                "tp_mean_mechanism": Lg.mean, "tp_mean_white": Lw.mean, "tp_cv_mechanism": Lg.cv, "tp_cv_white": Lw.cv,
                "ratio_tp_over_waiting_time": kl_tp["kl"] / kl_t["kl"],
                "s_exp_minus_s": el.elasticity * np.exp(-el.elasticity),
                "kl_waiting_time_over_s_exp_minus_s": kl_t["kl"] / (el.elasticity * np.exp(-el.elasticity)),
            })
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["mechanism"], []).append(r)
    summ = {}
    for k, rs in by.items():
        rs = sorted(rs, key=lambda r: r["barrier_over_D"])
        tp = np.array([r["kl_transition_path_duration"] for r in rs])
        tm = np.array([r["kl_waiting_time"] for r in rs])
        c = np.array([r["kl_waiting_time_over_s_exp_minus_s"] for r in rs])
        summ[k] = {
            "kl_tp_min": float(tp.min()), "kl_tp_max": float(tp.max()),
            "kl_waiting_time_first_last": [float(tm[0]), float(tm[-1])],
            "ratio_tp_over_waiting_time_at_max_barrier": float(tp[-1] / tm[-1]),
            "duality_constant_range": [float(c.min()), float(c.max())],
        }
    return {"rows": rows, "summary": summ}


def run(p: dict, **_: object) -> dict:
    return {
        "identities": _identities(p),
        "random_potentials": _random_potentials(p),
        "two_stage": _two_stage(p),
        "duality": _duality(p),
    }
