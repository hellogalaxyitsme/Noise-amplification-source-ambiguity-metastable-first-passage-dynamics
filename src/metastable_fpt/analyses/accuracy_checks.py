"""Numerical accuracy at the smallest reported divergences (Supplemental Material, Table S3).

At the mechanism and barrier with the smallest escape-matched divergence
(telegraph forcing, ``r = 0.25``, ``Delta U / D = 8``) the divergence between the
mechanism and its escape-matched white twin is recomputed

* on refined spatial grids (``n_x = 100, 150, 200, 300``),
* with refined time propagation (initial step and steps per stage scaled by 1/2 and 1/4),

each time with the hazard form checked against the density form and with the
transition-path-duration divergence. For OU forcing (``tau = 0.1``,
``Delta U / D = 8``) the auxiliary forcing grid is refined (``m_eta = 41, 61, 81``).

All quantities are deterministic; no random numbers are used.
"""

from __future__ import annotations

import numpy as np

from ..generators import Domain, quartic_gradient, white_generator, x_index_of_states
from ..timing_law import TimeGrid, default_t_end, kl_timing, match_rate, spectrum, timing_law
from ..transition_paths import transition_path_generator
from . import timing_equivalence as te

NAME = "accuracy_checks"
DESCRIPTION = "Spatial, time-propagation and forcing-grid refinement at the smallest escape-matched divergences"
BARRIER = 0.25


def parameters(quick: bool = False) -> dict:
    return {
        "x_A": -1.0,
        "t_end_margin": 32.0,
        "tp_margin": 24.0,
        "point": {"mechanism": ["telegraph", 0.25], "barrier_over_D": 8.0},
        "ou_point": {"mechanism": ["ou", 0.1], "barrier_over_D": 8.0},
        "n_x": [100, 150] if quick else [100, 150, 200, 300],
        "time_refinement": [1, 2] if quick else [1, 2, 4],
        "m_eta": [31, 41] if quick else [41, 61, 81],
        "n_x_reference": 150,
        "m_eta_reference": 41,
        "dt0": 2e-3,
        "steps_per_stage": 80,
        "tp_dt0": 1e-4,
        "tp_steps_per_stage": 100,
    }


def _comparison(p: dict, kind: str, par: float, beta: float, *, n_x: int, m_eta: int, refine: int) -> dict:
    gU = quartic_gradient()
    base = te.parameters()
    base["m_eta"] = m_eta
    dom = Domain(base["ell"], base["bnd"], int(n_x))
    D = BARRIER / beta
    _, gen = te._mechanism(kind, par, D=D, frac=base["q_fraction"], p=base, dom=dom, gU=gU)
    mk = lambda DD: white_generator(gU, D=DD, x0=base["x0"], dom=dom)  # noqa: E731
    sg = spectrum(gen)
    D_esc, w = match_rate(mk, sg.lam1, D_lo=D / 2.5, D_hi=D * 2.5)
    sw = spectrum(w)
    grid = TimeGrid(dt0=p["dt0"] / refine, steps=int(p["steps_per_stage"]) * refine,
                    t_end=max(default_t_end(sg, p["t_end_margin"]), default_t_end(sw, p["t_end_margin"])))
    kl_t = kl_timing(timing_law(gen, grid, spec=sg), timing_law(w, grid, spec=sw))
    iA = int(np.searchsorted(dom.x, p["x_A"]))
    tg = transition_path_generator(gen, x_index_of_states(gen, dom) <= iA)
    tw = transition_path_generator(w, x_index_of_states(w, dom) <= iA)
    stg, stw = spectrum(tg), spectrum(tw)
    tgrid = TimeGrid(dt0=p["tp_dt0"] / refine, steps=int(p["tp_steps_per_stage"]) * refine,
                     t_end=max(default_t_end(stg, p["tp_margin"]), default_t_end(stw, p["tp_margin"])))
    kl_tp = kl_timing(timing_law(tg, tgrid, spec=stg), timing_law(tw, tgrid, spec=stw))
    return {
        "n_x": int(n_x), "m_eta": int(m_eta), "time_refinement": int(refine),
        "lambda": sg.lam1, "D_esc_over_D_eff": D_esc / D,
        "kl_waiting_time": kl_t["kl"], "kl_waiting_time_density_form": kl_t["kl_density_form"],
        "hazard_vs_density_abs": abs(kl_t["kl"] - kl_t["kl_density_form"]),
        "hazard_vs_density_rel": abs(kl_t["kl"] - kl_t["kl_density_form"]) / max(kl_t["kl"], 1e-300),
        "kl_transition_path_duration": kl_tp["kl"],
        "ratio_tp_over_waiting_time": kl_tp["kl"] / kl_t["kl"],
    }


def _spread(rows: list[dict], key: str) -> float:
    v = np.array([r[key] for r in rows])
    return float((v.max() - v.min()) / v[-1])


def run(p: dict, **_: object) -> dict:
    kind, par = p["point"]["mechanism"]
    beta = p["point"]["barrier_over_D"]
    spatial = [_comparison(p, kind, par, beta, n_x=n, m_eta=p["m_eta_reference"], refine=1) for n in p["n_x"]]
    time = [_comparison(p, kind, par, beta, n_x=p["n_x_reference"], m_eta=p["m_eta_reference"], refine=k)
            for k in p["time_refinement"]]
    okind, opar = p["ou_point"]["mechanism"]
    obeta = p["ou_point"]["barrier_over_D"]
    forcing = [_comparison(p, okind, opar, obeta, n_x=p["n_x_reference"], m_eta=m, refine=1) for m in p["m_eta"]]
    return {
        "point": p["point"],
        "ou_point": p["ou_point"],
        "spatial_refinement": spatial,
        "time_refinement": time,
        "forcing_grid_refinement": forcing,
        "relative_spread": {
            "spatial_kl_waiting_time": _spread(spatial, "kl_waiting_time"),
            "time_kl_waiting_time": _spread(time, "kl_waiting_time"),
            "forcing_kl_waiting_time": _spread(forcing, "kl_waiting_time"),
            "spatial_kl_transition_path": _spread(spatial, "kl_transition_path_duration"),
            "forcing_kl_transition_path": _spread(forcing, "kl_transition_path_duration"),
        },
        "max_hazard_vs_density_rel": max(r["hazard_vs_density_rel"] for r in spatial + time + forcing),
        "max_hazard_vs_density_abs": max(r["hazard_vs_density_abs"] for r in spatial + time + forcing),
    }
