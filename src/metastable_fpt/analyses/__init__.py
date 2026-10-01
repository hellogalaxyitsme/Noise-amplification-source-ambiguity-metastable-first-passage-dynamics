"""Analyses reported in the article, one module each."""

from . import (
    human_waiting_times,
    identities_duality,
    mc_effective_diffusion_controls,
    mc_escape_matched_controls,
    timing_equivalence,
)

ANALYSES = {
    m.NAME: m
    for m in (
        timing_equivalence,
        identities_duality,
        human_waiting_times,
        mc_effective_diffusion_controls,
        mc_escape_matched_controls,
    )
}
