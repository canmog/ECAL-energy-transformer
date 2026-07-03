# dual_v1 (2026-06-20): in-place upgrades to transformer/

All config-gated; the code in the v1-sw_d192 commit already contains them.
This tag marks the milestone; the preceding commit's files ARE the dual_v1 build.

* `model.bottleneck_residual=false` — remove the `tokens +` bypass at the tap, so
  the energy path can no longer route around the physics sub-space.
* `model.dual_energy_head=true` — energy = e_phys (from pooled h_phys, gauge-fixed
  by its own aux loss) + e_free (deep correction).
* `loss.weighting=normalized` — NormalizedWeighter: EMA-normalised losses with
  FIXED manual weights (energy 1.0, aux 0.3); removes the learnable log-variance
  that ran away (w_energy -> 451).
* `loss.energy_rolloff` — power-law down-weighting above 2 TeV (index 2.7).
* `data.concept_use` — 7-concept trim (drop x0/y0/kx/ky).
* evaluate/probe: E<=2 TeV summary + dual-head decomposition.

Outcome (2026-06-21 ablation, cache_full10, d192, 50 ep): the new STACK cost
+1.66% binned sigma/E (abl 7.28% vs sw_d192 5.62%); end-split (m4) neutral,
mid-tap+bypass-off hurt, anchor (m2) helped only sub-TeV. All variants REJECTED;
v2 resets to the sw_d192 configuration. See transformerReport/report.tex §9.
