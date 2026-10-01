"""Timing equivalence of noise mechanisms under escape-rate matching (Section III of the article).

For each noise mechanism and barrier height ``Delta U / D`` the commitment-time
law of the discretised chain is compared with three white-noise models that share
its drift:

* ``D_eff``: white noise at the mechanism's long-time (effective) diffusion;
* ``D_esc``: white noise with the same escape rate ``lambda_1`` (Corollary 3);
* ``D_mean``: white noise with the same mean commitment time.

Reported per comparison: the per-trial Kullback-Leibler divergence (hazard form
of Lemma 1, checked against the density form), its rate-only part, the
chain-rule decomposition of Theorem 2 at the hazard-relaxation time with the
late-part bound of Eq. (3), and the number of trials at which the summed error
probabilities of any test can first fall below ``1 - eta`` (Corollary 3).

A second block evaluates the matched-control design: a candidate made of white
noise plus OU and telegraph forcing, four controls with the same effective
diffusion, and white controls matched in escape rate and in mean. A third block
refines the spatial grid at one representative point.

All quantities are deterministic; no random numbers are used.
"""

from __future__ import annotations

import numpy as np

from ..generators import (
    Domain,
    composite_generator,
    jump_generator,
    ou_generator,
    quartic_gradient,
    telegraph_generator,
    white_generator,
)
from ..timing_law import (
    TimeGrid,
    decomposition_at,
    default_t_end,
    hazard_relaxation_time,
    kl_timing,
    match_mean,
    match_rate,
    mean_only,
    spectrum,
    timing_law,
)

NAME = "timing_equivalence"
DESCRIPTION = "Divergence between commitment-time laws under effective-diffusion, escape-rate and mean matching"
BARRIER = 0.25  # Delta U of U(x) = x^4/4 - x^2/2 between x0 = -1 and the barrier top at 0


def parameters(quick: bool = False) -> dict:
    return {
        "ell": -2.0,
        "bnd": 0.0,
        "x0": -1.0,
        "n_x": 100 if quick else 150,
        "m_eta": 31 if quick else 41,
        "q_fraction": 0.5,
        "K": 0.5,
        "gamma_safety": 0.9,
        "t_end_margin": 32.0,
        "dt0": 2e-3,
        "steps_per_stage": 80,
        "eta": 0.5,
        "jump_size": 0.1,
        "state_slope": 0.5,
        "barrier_over_D": [2.0, 4.0, 6.0] if quick else [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "mechanisms": (
            [["ou", 0.5], ["telegraph", 1.0], ["jump", 0.0], ["state", 0.0]]
            if quick
            else [["ou", 0.1], ["ou", 0.5], ["ou", 2.0], ["telegraph", 4.0], ["telegraph", 1.0],
                  ["telegraph", 0.25], ["jump", 0.0], ["state", 0.0]]
        ),
        "grid_convergence_n_x": [80, 120] if quick else [100, 150, 200, 300],
        "grid_convergence_point": {"mechanism": ["ou", 0.5], "barrier_over_D": 5.0},
        "matched_control_design": {"D_C": 0.04, "D_Q": 0.004, "tau": 20.0, "telegraph_rate": 0.5,
                                   "rmst_horizon": 1500.0},
    }


def _mechanism(kind: str, par: float, *, D: float, frac: float, p: dict, dom: Domain, gU):
    x0 = p["x0"]
    DQ, DC = frac * D, (1 - frac) * D
    if kind == "ou":
        return f"OU tau={par:g}", ou_generator(gU, D_C=DC, D_Q=DQ, tau=par, x0=x0, dom=dom, m_eta=int(p["m_eta"]))
    if kind == "telegraph":
        return f"telegraph r={par:g}", telegraph_generator(gU, D_C=DC, D_Q=DQ, rate=par, x0=x0, dom=dom)
    if kind == "jump":
        nodes = max(1, int(round(p["jump_size"] / dom.h)))
        return f"jump A={nodes * dom.h:.3f}", jump_generator(gU, D_C=DC, D_Q=DQ, jump_nodes=nodes, x0=x0, dom=dom)
    if kind == "state":
        slope = p["state_slope"]
        return (
            f"state-dependent k={slope:g}",
            white_generator(gU, D=lambda x: D * (1.0 + slope * (np.asarray(x) - x0)), x0=x0, dom=dom),
        )
    raise ValueError(kind)


def _compare(gen, whites: dict, p: dict) -> tuple[dict, dict]:
    """Timing laws on one common time grid and all divergences against the white models."""

    specs = {"mech": spectrum(gen)}
    for k, g in whites.items():
        specs[k] = spectrum(g)
    t_end = max(default_t_end(s, p["t_end_margin"]) for s in specs.values())
    grid = TimeGrid(dt0=p["dt0"], steps=int(p["steps_per_stage"]), t_end=t_end)
    laws = {"mech": timing_law(gen, grid, spec=specs["mech"])}
    for k, g in whites.items():
        laws[k] = timing_law(g, grid, spec=specs[k])
    A = laws["mech"]
    out = {
        "lambda": A.lam,
        "lambda2": [A.spectrum.lam2.real, A.spectrum.lam2.imag],
        "gap": A.spectrum.gap,
        "lambda_over_gap": A.lam / A.spectrum.gap,
        "mean": A.mean,
        "cv": A.cv,
        "mass_defect": A.mass_defect,
        "tail_hazard_rel_err": A.tail_hazard_rel_err,
        "t_end": t_end,
        "n_time_points": int(A.t.size),
        "against": {},
    }
    for k in whites:
        B = laws[k]
        kl = kl_timing(A, B)
        gamma = p["gamma_safety"] * min(A.spectrum.gap, B.spectrum.gap)
        try:
            t0 = max(
                hazard_relaxation_time(A, K=p["K"], gamma=gamma),
                hazard_relaxation_time(B, K=p["K"], gamma=gamma),
            )
            dec = decomposition_at(A, B, t0)
            lam = A.lam
            if abs(A.lam - B.lam) <= 1e-8 * lam:
                late_bound = 2 * p["K"] ** 2 * lam / ((1 - p["K"]) * gamma)
            else:
                lA, lB, K = A.lam, B.lam, p["K"]
                late_bound = min(
                    (1 + d) * (lA - lB) ** 2 / ((1 - K) ** 2 * lA * lB)
                    + (1 + 1 / d) * (lA + lB) ** 2 * K**2 / (2 * gamma * lB * (1 - K))
                    for d in np.logspace(-3, 3, 61)
                )
            dec["late_bound"] = float(late_bound)
            dec["late_bound_holds"] = bool(dec["kl_late_conditional"] <= late_bound * (1 + 1e-6))
            dec["reassembly_rel_err"] = float(abs(dec["kl_reassembled"] - kl["kl"]) / max(kl["kl"], 1e-300))
        except Exception as exc:  # reported in the output
            dec = {"error": repr(exc)}
        out["against"][k] = {
            **kl,
            "white_D": float(whites[k].meta["D"]),
            "white_lambda": B.lam,
            "white_mean": B.mean,
            "white_cv": B.cv,
            "rate_share": float(min(kl["kl_rate_only"] / kl["kl"], 10.0)) if kl["kl"] > 0 else None,
            "hazard_vs_density_rel": float(abs(kl["kl"] - kl["kl_density_form"]) / max(kl["kl"], 1e-300)),
            "hazard_vs_density_abs": float(abs(kl["kl"] - kl["kl_density_form"])),
            "n_trials_error_sum_below_1_minus_eta": float(2 * p["eta"] ** 2 / kl["kl"]) if kl["kl"] > 0 else float("inf"),
            "decomposition": dec,
        }
    return out, laws


def _rmst(law, horizon: float) -> float:
    """Restricted mean of ``min(T, horizon)`` from the survival curve (constant hazard beyond the grid)."""

    sel = law.t <= horizon
    t, S = law.t[sel], law.S[sel]
    if t[-1] < horizon:
        extra = S[-1] * (1 - np.exp(-law.lam * (horizon - t[-1]))) / law.lam
    else:
        extra = 0.0
    return float(np.sum(0.5 * (S[1:] + S[:-1]) * np.diff(t)) + extra)


def _median(law) -> float:
    idx = np.nonzero(law.S <= 0.5)[0]
    if idx.size == 0:
        return float(law.t[-1] + np.log(2 * law.S[-1]) / law.lam)
    i = int(idx[0])
    t0, t1, s0, s1 = law.t[i - 1], law.t[i], law.S[i - 1], law.S[i]
    return float(t0 + (s0 - 0.5) * (t1 - t0) / (s0 - s1))


def _matched_control_design(p: dict, gU, dom: Domain) -> dict:
    e = p["matched_control_design"]
    DC, DQ, tau, r, H = e["D_C"], e["D_Q"], e["tau"], e["telegraph_rate"], e["rmst_horizon"]
    x0 = p["x0"]
    D_total = DC + 2 * DQ
    cand = composite_generator(gU, D_C=DC, x0=x0, ou=(DQ, tau), telegraph=(DQ, r), dom=dom,
                               m_eta=int(p["m_eta"]), label="candidate")
    mk = lambda D: white_generator(gU, D=D, x0=x0, dom=dom)  # noqa: E731
    lam = spectrum(cand).lam1
    D_esc, w_esc = match_rate(mk, lam, D_lo=D_total / 3, D_hi=D_total * 2.5)
    D_mean, w_mean = match_mean(mk, mean_only(cand), D_lo=D_total / 3, D_hi=D_total * 2.5)
    A_jump = float(np.sqrt(2 * DQ / r))
    controls = {
        "white_D_eff": mk(D_total),
        "colored": composite_generator(gU, D_C=DC + DQ, x0=x0, ou=(DQ, tau), dom=dom, m_eta=int(p["m_eta"])),
        "telegraph": composite_generator(gU, D_C=DC + DQ, x0=x0, telegraph=(DQ, r), dom=dom),
        "compound_poisson": jump_generator(gU, D_C=DC + DQ, D_Q=DQ, jump_nodes=max(1, int(round(A_jump / dom.h))),
                                           x0=x0, dom=dom),
        "white_D_esc": w_esc,
        "white_D_mean": w_mean,
    }
    for g in controls.values():
        if "D" not in g.meta:
            g.meta["D"] = float("nan")
    res, laws = _compare(cand, controls, p)
    A = laws["mech"]
    table = {}
    for k in controls:
        B = laws[k]
        table[k] = {
            "rmst": _rmst(B, H),
            "rmst_difference_candidate_minus_control": _rmst(A, H) - _rmst(B, H),
            "median": _median(B),
            "kl_candidate_vs_control": res["against"][k]["kl"],
            "n_trials_error_sum_below_1_minus_eta": res["against"][k]["n_trials_error_sum_below_1_minus_eta"],
            "lambda": B.lam,
        }
    return {
        "candidate": {"rmst": _rmst(A, H), "median": _median(A), "lambda": A.lam, "mean": A.mean, "cv": A.cv},
        "D_eff": D_total,
        "D_esc": D_esc,
        "D_mean": D_mean,
        "rmst_horizon": H,
        "controls": table,
    }


def run(p: dict, **_: object) -> dict:
    gU = quartic_gradient()
    dom = Domain(p["ell"], p["bnd"], int(p["n_x"]))
    frac = p["q_fraction"]
    rows = []
    mk = lambda D: white_generator(gU, D=D, x0=p["x0"], dom=dom)  # noqa: E731
    for kind, par in p["mechanisms"]:
        for beta in p["barrier_over_D"]:
            D = BARRIER / beta
            name, gen = _mechanism(kind, par, D=D, frac=frac, p=p, dom=dom, gU=gU)
            lam = spectrum(gen).lam1
            D_esc, w_esc = match_rate(mk, lam, D_lo=D / 2.5, D_hi=D * 2.5)
            D_mean, w_mean = match_mean(mk, mean_only(gen), D_lo=D / 2.5, D_hi=D * 2.5)
            res, _ = _compare(gen, {"D_eff": mk(D), "D_esc": w_esc, "D_mean": w_mean}, p)
            res.update({"mechanism": name, "kind": kind, "par": par, "barrier_over_D": beta, "D_eff": D,
                        "D_esc": D_esc, "D_mean": D_mean, "D_esc_over_D_eff": D_esc / D})
            rows.append(res)

    # spatial grid refinement at one representative point
    conv = []
    kind, par = p["grid_convergence_point"]["mechanism"]
    D = BARRIER / p["grid_convergence_point"]["barrier_over_D"]
    for n in p["grid_convergence_n_x"]:
        dom_n = Domain(p["ell"], p["bnd"], int(n))
        mk2 = lambda DD, dom_n=dom_n: white_generator(gU, D=DD, x0=p["x0"], dom=dom_n)  # noqa: E731
        _, g = _mechanism(kind, par, D=D, frac=frac, p=p, dom=dom_n, gU=gU)
        lam = spectrum(g).lam1
        D_esc, w_esc = match_rate(mk2, lam, D_lo=D / 2.5, D_hi=D * 2.5)
        r, _ = _compare(g, {"D_eff": mk2(D), "D_esc": w_esc}, p)
        conv.append({"n_x": int(n), "lambda": r["lambda"], "D_esc_over_D_eff": D_esc / D,
                     "kl_D_eff": r["against"]["D_eff"]["kl"], "kl_D_esc": r["against"]["D_esc"]["kl"]})

    design = _matched_control_design(p, gU, dom)

    by_mech: dict[str, list] = {}
    for r in rows:
        by_mech.setdefault(r["mechanism"], []).append(r)
    summary = {}
    for name, rs in by_mech.items():
        rs = sorted(rs, key=lambda r: r["barrier_over_D"])
        kl_esc = np.array([r["against"]["D_esc"]["kl"] for r in rs])
        kl_eff = np.array([r["against"]["D_eff"]["kl"] for r in rs])
        betas = np.array([r["barrier_over_D"] for r in rs])
        summary[name] = {
            "barrier_over_D": betas.tolist(),
            "kl_D_esc_by_barrier": kl_esc.tolist(),
            "kl_D_eff_by_barrier": kl_eff.tolist(),
            "kl_D_mean_by_barrier": [r["against"]["D_mean"]["kl"] for r in rs],
            "rate_share_D_eff": [r["against"]["D_eff"]["rate_share"] for r in rs],
            "D_esc_over_D_eff": [r["D_esc_over_D_eff"] for r in rs],
            "lambda_over_gap": [r["lambda_over_gap"] for r in rs],
            "log_slope_kl_esc_vs_barrier": float(np.polyfit(betas, np.log(kl_esc), 1)[0]),
            "late_bound_holds_everywhere": all(
                r["against"]["D_esc"]["decomposition"].get("late_bound_holds", False) for r in rs),
        }
    checks = {
        "max_hazard_vs_density_rel_err_for_kl_above_1e-6": max(
            a["hazard_vs_density_rel"] for r in rows for a in r["against"].values() if a["kl"] >= 1e-6),
        "max_hazard_vs_density_abs_err": max(a["hazard_vs_density_abs"] for r in rows for a in r["against"].values()),
        "max_reassembly_rel_err": float(np.nanmax([
            a["decomposition"].get("reassembly_rel_err", np.nan) for r in rows for a in r["against"].values()])),
        "max_mass_defect": max(r["mass_defect"] for r in rows),
        "max_tail_hazard_rel_err": max(r["tail_hazard_rel_err"] for r in rows),
    }
    return {
        "rows": rows,
        "summary_by_mechanism": summary,
        "grid_convergence": conv,
        "matched_control_design": design,
        "numerical_checks": checks,
    }
