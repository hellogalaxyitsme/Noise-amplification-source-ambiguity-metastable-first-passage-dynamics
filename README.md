# Noise amplification and source ambiguity in metastable first-passage dynamics

The code reproduces every numerical result of the article and its supplementary
material:

* commitment-time laws of killed continuous-time Markov chains that discretise a
  one-dimensional metastable diffusion driven by white, colored
  (Ornstein-Uhlenbeck), telegraph, jump and state-dependent noise, and the
  Kullback-Leibler divergence between them;
* the weighted-barrier identities for the noise elasticity of the mean
  first-passage time and for its variance, and the variability-amplification
  bound;
* transition-path duration laws;
* Euler-Maruyama simulations of the same models with censoring-aware
  comparisons;
* the analysis of self-paced waiting times from the public Libet-clock dataset of
  Jeay-Bizot *et al.* (PLOS Biology, 2026).

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/hellogalaxyitsme/Noise-amplification-source-ambiguity-metastable-first-passage-dynamics.git
cd Noise-amplification-source-ambiguity-metastable-first-passage-dynamics
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

The only run-time dependencies are NumPy and SciPy. The results reported in the
article were produced with Python 3.10.1, NumPy 2.2.6 and SciPy 1.15.3, and
reproduced with Python 3.12.1, NumPy 2.5.3 and SciPy 1.18.1.

## Data

The human analysis needs only the BrainVision header and marker files
(`P01_RespRP.vhdr`, `P01_RespRP.vmrk`, ..., 34 files, about 0.4 MB) of the
dataset released with Jeay-Bizot, Chishti, Maoz and Schurger, *No evidence for
modulation of the readiness potential by respiratory phase during natural
breathing*, PLOS Biology 24(9): e3003982 (2026), on the Open Science Framework
(https://doi.org/10.17605/OSF.IO/3CVYK). Download and verify them with

```bash
python -m metastable_fpt download-data
```

which writes the files to `data/markers/` and checks each one against the SHA-256
manifest `data/marker_checksums.sha256`. No EEG signal file is required. The dataset
is not redistributed in this repository.

## Running the analyses

```bash
python -m metastable_fpt list                       # the five analyses
python -m metastable_fpt run all                    # full settings, as in the article
python -m metastable_fpt run timing_equivalence     # a single analysis
python -m metastable_fpt run all --quick            # reduced settings for a fast check
```

Each analysis writes `results/<analysis>/results.json`, together with the
parameters used (`parameters.json`) and the software environment and run time
(`environment.json`). `--workers N` sets the number of processes for the
accumulator fits; the results do not depend on it.

| Analysis | Content | Run time (full settings) |
|---|---|---|
| `timing_equivalence` | Divergence between commitment-time laws under effective-diffusion, escape-rate and mean matching; the matched-control design; spatial grid refinement | about 30 s |
| `identities_duality` | Lemmas 4 and 5, Theorem 6 on random potentials and for two stages, waiting-time versus transition-path information | about 40 s |
| `human_waiting_times` | Waiting-time statistics, the Theorem 6 bound and accumulator fits for 17 participants | about 10 min |
| `mc_effective_diffusion_controls` | Euler-Maruyama comparison of the candidate with controls matched in effective diffusion | about 2 min |
| `mc_escape_matched_controls` | Euler-Maruyama comparison with escape-matched and mean-matched white controls at two time steps | about 7 min |

Run times were measured on a 28-thread workstation, with 20 processes for the accumulator fits.

## Correspondence with the article

| Article | Analysis | Entry in `results.json` |
|---|---|---|
| Sec. III, Fig. 1(a, b) | `timing_equivalence` | `summary_by_mechanism`, `rows` |
| Sec. III (matched-control design), Fig. 1(c), bars | `timing_equivalence` | `matched_control_design` |
| Fig. 1(c), crosses | `mc_effective_diffusion_controls` | `controls` |
| Sec. III (Monte Carlo confirmation), Fig. 2, Sec. S2 | `mc_escape_matched_controls` | `by_time_step` |
| Sec. IV, Fig. 3(a, b) | `identities_duality` | `identities`, `random_potentials` |
| Sec. V, Fig. 3(c) | `identities_duality` | `duality` |
| Sec. VI, Fig. 4 | `human_waiting_times` | `subjects`, `group` |
| Table S1 | `timing_equivalence`, `identities_duality`, `mc_escape_matched_controls` | `numerical_checks`; `identities`, `random_potentials`, `two_stage`; `positive_control` |
| Table S2 | `timing_equivalence` | `grid_convergence` |

Lemma and theorem numbers in the code refer to the article: Lemma 1 (hazard
identity), Theorem 2 (rate-transient decomposition), Corollary 3 (escape-rate
equivalence), Lemmas 4 and 5 (elasticity and variance identities), Theorem 6
(variability-amplification inequality) and Theorem 7 (duality).

Model time is in units in which `U''(-1) = 2`; the article quotes these times in
milliseconds. Human waiting times are in seconds.

## Reproducibility

All calculations on the discretised chains are deterministic. Every random
quantity (random potentials, bootstrap resamples, Monte Carlo paths) is drawn
from a NumPy generator seeded by a master seed and a fixed stream label
(`metastable_fpt.seeding.make_generator`); each noise source of each simulated
arm has its own stream. The master seeds and stream labels are part of the
parameter set written to `parameters.json`, so a run reproduces the reported
values exactly, up to floating-point differences between library versions.

## Package layout

```
src/metastable_fpt/
    generators.py         killed Markov generators for the noise mechanisms
    timing_law.py         timing laws, spectrum, moments, divergence, decomposition, matching
    transition_paths.py   committor, quasi-stationary law, transition-path generator
    elasticity.py         weighted-barrier identities and the elasticity bound
    accumulator.py        likelihood fits of the leaky accumulator
    forcing.py            OU, telegraph and compound-Poisson forcing samplers
    simulation.py         Euler-Maruyama first-passage simulation
    survival_stats.py     Kaplan-Meier restricted means and margin-based comparisons
    waiting_times.py      marker parsing and waiting-time statistics
    data_download.py      download and verification of the marker files
    seeding.py            labelled random streams and input validation
    results_io.py         JSON output
    cli.py                command-line interface
    analyses/             the five analyses
tests/                    unit tests against closed forms and independent references
data/                     checksum manifest of the marker files
```

Run the tests with `pytest`.
