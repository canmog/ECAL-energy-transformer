# transformer_v2cham/ — champion: wdx + v2m1 + v2b framework

Stacks the two proven, **orthogonal** wins on the `sw_d192` model and runs them inside the
`v2b` selection framework, so a single run produces the best checkpoint on each axis:

| ingredient | from | effect |
|------------|------|--------|
| `wd_exclude_1d: true` | wdx | exclude 1-D params (LayerNorm/bias/AttnPool-query) from weight decay → best robust core + calibration |
| `loss.fit_quality_weight` | v2m1 | down-weight concept+recon on poorly-fit events (token-level residual) → suppresses the raw-std tail |
| `train.select_metric` + multi-best ckpts | v2b | composite selection (`tail_weight=0` here) + saves `best.pt` / `best_robust.pt` / `best_raw.pt` |

## Why this should be the new best

- `wdx+v2m1` already gave the best robust core (1.38% ≤2 TeV) and best calibration; here it
  also gets a **raw-selected checkpoint** for the tail, for free, from the same run.
- On raw-std being noisy: rather than tuning a `tail_weight`, we rely on **multi-best** —
  `best_raw.pt` is the lightest-tail epoch of the best model. (Optional later A/B:
  `select_metric.tail_term=excess, tail_weight≈0.3`.)

## Pick the checkpoint per metric

- `best_robust.pt` → robust core (expected ~1.38% ≤2 TeV);
- `best_raw.pt`    → raw-std tail (lightest-tail epoch of this model);
- `best.pt`        → bias-aware robust composite (drives early stop).

`job_v2cham.sub` trains once then evaluates all three into `metrics_<ck>.json`.

## Run

```bash
sbatch job_v2cham.sub        # train (multi-best) -> eval 3 ckpts -> probe; cache_full10, d192, 50 ep
bash   smoke_test.sh         # 2-epoch end-to-end check on a private throwaway cache
```

See `../transformer_v2/EXPERIMENT_LOG.md` for the full sweep and `../transformerReport/` for
the study.
