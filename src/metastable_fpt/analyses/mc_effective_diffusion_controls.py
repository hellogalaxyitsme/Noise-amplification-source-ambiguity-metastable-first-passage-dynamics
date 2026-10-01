"""Monte Carlo comparison of a structured candidate with controls matched in effective diffusion.

Candidate: white noise ``D_C`` plus an OU forcing ``(D_Q, tau)`` and a telegraph
forcing ``(D_Q, r)``, with effective diffusion ``D_eff = D_C + 2 D_Q``. Controls
with the same ``D_eff``:

* ``white``: white noise at ``D_eff``;
* ``colored``: white noise ``D_C + D_Q`` plus the OU forcing;
* ``telegraph``: white noise ``D_C + D_Q`` plus the telegraph forcing;
* ``compound_poisson``: white noise ``D_C + D_Q`` plus Gaussian-amplitude jumps at
  rate ``r`` with long-time diffusion ``D_Q``;

and an independent run of the candidate. All models share the drift of
``U(x) = x^4/4 - x^2/2``, start at ``x0 = -1``, reflect at ``-2`` and commit at ``0``.

Each arm is simulated with the Euler-Maruyama scheme; every noise source of every
arm has its own labelled random stream under the master seed (the labels are part
of the parameter set and fix the random numbers). Arms are compared by the
difference in restricted mean survival time (RMST) up to the horizon, with a
normal 95 % interval, against a margin of 5 % of the candidate's Kaplan-Meier
median. These are the Monte Carlo values shown as crosses in Fig. 1(c).
"""

from __future__ import annotations

import numpy as np

from ..forcing import CompoundPoissonStream, OUStream, TelegraphStream
from ..generators import quartic_gradient
from ..seeding import make_generator
from ..simulation import simulate_crossing
from ..survival_stats import equivalence_verdict, observed_law_from_times, rmst_difference

NAME = "mc_effective_diffusion_controls"
DESCRIPTION = "Euler-Maruyama comparison of the candidate with controls matched in effective diffusion"

# Random-stream labels of every arm and noise source. They determine the random
# numbers drawn under the master seed and are kept fixed for reproducibility.
STREAMS = {
    "candidate": {"white": "candidate_white",
                  "ou_init": "candidate_ou_seed_init", "ou": "candidate_ou",
                  "tele_init": "candidate_tele_seed_init", "tele": "candidate_tele"},
    "candidate_independent_run": {"white": "twin_white",
                                  "ou_init": "twin_ou_seed_init", "ou": "twin_ou",
                                  "tele_init": "twin_tele_seed_init", "tele": "twin_tele"},
    "white": {"white": "c1_white"},
    "colored": {"white": "c2_white", "ou_init": "c2_ou_seed_init", "ou": "c2_ou"},
    "telegraph": {"white": "c3_white", "tele_init": "c3_tele_seed_init", "tele": "c3_tele"},
    "compound_poisson": {"white": "c4_white", "jump": "c4_jump_seed"},
}


def parameters(quick: bool = False) -> dict:
    return {
        "master_seed": 20260923,
        "ell": -2.0,
        "bnd": 0.0,
        "x0": -1.0,
        "D_C": 0.04,
        "D_Q": 0.004,
        "tau": 20.0,
        "telegraph_rate": 0.5,
        "n_paths": 1500 if quick else 6000,
        "dt": 0.05 if quick else 0.02,
        "horizon": 900.0 if quick else 1500.0,
        "equivalence_margin_fraction": 0.05,
        "streams": STREAMS,
    }


def _arm(p: dict, name: str, *, white_D: float, ou: bool, tele: bool, jump: bool, n_steps: int):
    s = p["streams"][name]
    seed, n, dt = p["master_seed"], int(p["n_paths"]), p["dt"]
    forcings, rngs = [], {}
    if ou:
        forcings.append(("ou", OUStream(D=p["D_Q"], tau=p["tau"], dt=dt, n_paths=n,
                                        rng=make_generator(seed, s["ou_init"]))))
        rngs["ou"] = make_generator(seed, s["ou"])
    if tele:
        forcings.append(("tele", TelegraphStream(D=p["D_Q"], rate=p["telegraph_rate"], dt=dt, n_paths=n,
                                                 rng=make_generator(seed, s["tele_init"]))))
        rngs["tele"] = make_generator(seed, s["tele"])
    if jump:
        forcings.append(("jump", CompoundPoissonStream(D=p["D_Q"], rate=p["telegraph_rate"], dt=dt, n_paths=n)))
        rngs["jump"] = make_generator(seed, s["jump"])
    gU = quartic_gradient()
    return simulate_crossing(
        drift=lambda x: -gU(x), threshold=p["bnd"], x0=p["x0"], dt=dt, n_steps=n_steps, n_paths=n,
        rng=make_generator(seed, s["white"]), white_D=white_D, forcings=forcings, rng_by_forcing=rngs,
        lower=p["ell"],
    )


def run(p: dict, **_: object) -> dict:
    DC, DQ = p["D_C"], p["D_Q"]
    D_eff = DC + 2.0 * DQ
    H = p["horizon"]
    n_steps = int(np.ceil(H / p["dt"]))
    arms = {
        "candidate": _arm(p, "candidate", white_D=DC, ou=True, tele=True, jump=False, n_steps=n_steps),
        "candidate_independent_run": _arm(p, "candidate_independent_run", white_D=DC, ou=True, tele=True,
                                          jump=False, n_steps=n_steps),
        "white": _arm(p, "white", white_D=D_eff, ou=False, tele=False, jump=False, n_steps=n_steps),
        "colored": _arm(p, "colored", white_D=DC + DQ, ou=True, tele=False, jump=False, n_steps=n_steps),
        "telegraph": _arm(p, "telegraph", white_D=DC + DQ, ou=False, tele=True, jump=False, n_steps=n_steps),
        "compound_poisson": _arm(p, "compound_poisson", white_D=DC + DQ, ou=False, tele=False, jump=True,
                                 n_steps=n_steps),
    }
    cand = observed_law_from_times(arms["candidate"].times, H)
    km_median = cand.median()
    margin = float(p["equivalence_margin_fraction"]) * (float(km_median) if np.isfinite(km_median) else H)
    controls = {}
    for name, res in arms.items():
        if name == "candidate":
            continue
        law = observed_law_from_times(res.times, H)
        test = rmst_difference(cand, law)
        controls[name] = {
            "n_events": law.n_events,
            "censor_fraction": law.censor_fraction,
            "km_median": law.median(),
            "rmst": law.restricted_mean(),
            "rmst_difference_candidate_minus_control": test["difference"],
            "ci": [test["ci_low"], test["ci_high"]],
            "ci_method": test["method"],
            **equivalence_verdict(test["ci_low"], test["ci_high"], margin),
        }
    return {
        "D_eff": D_eff,
        "candidate": {"n_events": cand.n_events, "censor_fraction": cand.censor_fraction,
                      "km_median": km_median, "rmst": cand.restricted_mean()},
        "margin": margin,
        "controls": controls,
    }
