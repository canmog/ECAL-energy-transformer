# Angle reconstruction snapshot

This branch preserves the synchronized `angleTransformer` research tree as it
stood on 2026-08-17, before selective integration into `main`.

The source directory's `.git` directory was an empty Mutagen placeholder, so no
independent angle-development commit history was available. Commit `162d085`
therefore records the source, tests, ROOT preparation tools, batch launchers, and
research notes as a content snapshot based on `ecalTransformer` commit `63f422d`.

Generated caches, checkpoints, plots, SLURM output, training logs, bytecode, and
the complete `runs/` tree are deliberately excluded. The `results/` directory
contains only the resolved configurations, cache metadata, and final metric JSON
needed to audit the promoted direction model, the three joint-model seeds, and
the fixed-10-MeV uncut-population evaluation.

The corrected physical projection convention is:

```text
stored view 1 -> X projection -> kx
stored view 0 -> Y projection -> ky
component_views = [1, 0]
```

The principal selected-domain results are:

- direction model of record (`angle_residual_physicsaux_d192_t10`): angle p68
  `0.2719 degree`, versus `0.3885 degree` for the same-event 3D fit;
- joint seed 73111, best-angle checkpoint: angle p68 `0.2770 degree` and energy
  Gaussian-core resolution `1.0965%` below 2 TeV;
- joint seed 123: angle p68 `0.2791 degree`;
- joint seed 777: angle p68 `0.2776 degree`;
- joint seed 73111 on the retained uncut test population: angle p68
  `0.8316 degree`, versus `1.4081 degree` for the 3D fit.

The JSON files are the authoritative machine-readable records; rounded values
above are only a navigation summary.
