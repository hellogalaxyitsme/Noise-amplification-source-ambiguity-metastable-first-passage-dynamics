"""Monte Carlo confirmation of escape-rate equivalence (Fig. 2 and Section S2 of the article).

The candidate of :mod:`mc_effective_diffusion_controls` is simulated again with
new random streams, at two Euler-Maruyama time steps, together with:

* ``white_D_eff``: white noise at the candidate's effective diffusion;
* ``white_D_esc``: white noise at the escape-equivalent intensity, computed from
  the discretised chain (never from simulation output);
* ``white_D_mean``: white noise at the intensity that matches the candidate's mean
  commitment time, also from the chain;
* ``candidate_independent_run``: the candidate with independent streams.

Each control is compared with the candidate by the RMST difference up to the
horizon, with a normal 95 % interval, against a margin of 5 % of the candidate's
Kaplan-Meier median at that time step. A positive control reports the detection
rate of a 20 % scale shift and the false-positive rate of the same comparison.
"""

from __future__ import annotations

import numpy as np

from ..forcing import OUStream, TelegraphStream
from ..generators import Domain, composite_generator, quartic_gradient, white_generator
from ..seeding import make_generator
from ..simulation import simulate_crossing
from ..survival_stats import equivalence_verdict, observed_law_from_times, power_positive_control, rmst_difference
from ..timing_law import match_mean, match_rate, mean_only, spectrum

NAME = "mc_escape_matched_controls"
DESCRIPTION = "Euler-Maruyama comparison of the candidate with escape-matched and mean-matched white controls"

# Random-stream label of each arm; the full label of a noise source is
# "<time-step label>_<arm label>_<source>". The labels fix the random numbers
# drawn under the master seed and are kept for reproducibility.
ARM_LABELS = {
    "candidate": "candidate",
    "candidate_independent_run": "twin",
    "white_D_eff": "c1_deff",
    "white_D_esc": "c1_desc",
    "white_D_mean": "c1_dmean",
}


def parameters(quick: bool = False) -> dict:
    return {
        "master_seed": 20260930,
        "ell": -2.0,
        "bnd": 0.0,
        "x0": -1.0,
        "D_C": 0.04,
        "D_Q": 0.004,
        "tau": 20.0,
        "telegraph_rate": 0.5,
        "n_paths": 1500 if quick else 6000,
        "horizon": 900.0 if quick else 1500.0,
        "dt_values": [0.02] if quick else [0.02, 0.005],
        "equivalence_margin_fraction": 0.05,
        "power_control_reps": 20 if quick else 40,
        "power_control_shift": 0.2,
        "n_x_chain": 150,
        "m_eta_chain": 41,
        "arm_labels": ARM_LABELS,
    }


def _arm(p: dict, label: str, *, dt: float, n_steps: int, white_D: float, with_forcing: bool, step_label: str):
    seed, n = p["master_seed"], int(p["n_paths"])
    forcings, rngs = [], {}
    if with_forcing:
        forcings = [
            ("ou", OUStream(D=p["D_Q"], tau=p["tau"], dt=dt, n_paths=n,
                            rng=make_generator(seed, f"{step_label}_{label}_ou_init"))),
            ("tele", TelegraphStream(D=p["D_Q"], rate=p["telegraph_rate"], dt=dt, n_paths=n,
                                     rng=make_generator(seed, f"{step_label}_{label}_tele_init"))),
        ]
        rngs = {"ou": make_generator(seed, f"{step_label}_{label}_ou"),
                "tele": make_generator(seed, f"{step_label}_{label}_tele")}
    gU = quartic_gradient()
    return simulate_crossing(
        drift=lambda x: -gU(x), threshold=p["bnd"], x0=p["x0"], dt=dt, n_steps=n_steps, n_paths=n,
        rng=make_generator(seed, f"{step_label}_{label}_white"), white_D=white_D,
        forcings=forcings, rng_by_forcing=rngs, lower=p["ell"],
    )


def run(p: dict, **_: object) -> dict:
    # intensities of the matched white controls from the discretised chain
    gU = quartic_gradient()
    dom = Domain(p["ell"], p["bnd"], int(p["n_x_chain"]))
    cand_chain = composite_generator(gU, D_C=p["D_C"], x0=p["x0"], ou=(p["D_Q"], p["tau"]),
                                     telegraph=(p["D_Q"], p["telegraph_rate"]), dom=dom, m_eta=int(p["m_eta_chain"]))
    D_eff = p["D_C"] + 2 * p["D_Q"]
    mk = lambda D: white_generator(gU, D=D, x0=p["x0"], dom=dom)  # noqa: E731
    D_esc, _ = match_rate(mk, spectrum(cand_chain).lam1, D_lo=D_eff / 3, D_hi=D_eff * 2.5)
    D_mean, _ = match_mean(mk, mean_only(cand_chain), D_lo=D_eff / 3, D_hi=D_eff * 2.5)

    H = p["horizon"]
    labels = p["arm_labels"]
    rng_b = make_generator(p["master_seed"], "bootstrap")
    by_step = {}
    for dt in p["dt_values"]:
        step_label = f"dt{dt:g}"
        n_steps = int(np.ceil(H / dt))
        common = {"dt": dt, "n_steps": n_steps, "step_label": step_label}
        arms = {
            "candidate": _arm(p, labels["candidate"], white_D=p["D_C"], with_forcing=True, **common),
            "candidate_independent_run": _arm(p, labels["candidate_independent_run"], white_D=p["D_C"],
                                              with_forcing=True, **common),
            "white_D_eff": _arm(p, labels["white_D_eff"], white_D=D_eff, with_forcing=False, **common),
            "white_D_esc": _arm(p, labels["white_D_esc"], white_D=D_esc, with_forcing=False, **common),
            "white_D_mean": _arm(p, labels["white_D_mean"], white_D=D_mean, with_forcing=False, **common),
        }
        cand = observed_law_from_times(arms["candidate"].times, H)
        km_median = cand.median()
        margin = float(p["equivalence_margin_fraction"]) * (float(km_median) if np.isfinite(km_median) else H)
        controls = {}
        for name, res in arms.items():
            if name == "candidate":
                continue
            law = observed_law_from_times(res.times, H)
            test = rmst_difference(cand, law, rng=rng_b)
            controls[name] = {
                "n_events": law.n_events,
                "censor_fraction": law.censor_fraction,
                "km_median": law.median(),
                "rmst_difference_candidate_minus_control": test["difference"],
                "ci": [test["ci_low"], test["ci_high"]],
                "ci_method": test["method"],
                **equivalence_verdict(test["ci_low"], test["ci_high"], margin),
            }
        power = power_positive_control(cand, rng=rng_b, shift_fraction=p["power_control_shift"],
                                       n_reps=int(p["power_control_reps"]))
        by_step[step_label] = {
            "dt": dt,
            "candidate_km_median": km_median,
            "candidate_censor_fraction": cand.censor_fraction,
            "margin": margin,
            "controls": controls,
            "positive_control": power,
        }
    return {"D_eff": D_eff, "D_esc": D_esc, "D_mean": D_mean, "by_time_step": by_step}
