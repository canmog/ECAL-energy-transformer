"""Evaluate the fixed energy-centroid shower-axis baseline on a cache split."""
import argparse
import json

import numpy as np
import torch

from data.dataset import EcalTokens, load_meta, make_loader
from data.geometry import COMPONENT_VIEWS
from models.physics import LayerCentroidSlope
from utils.angle import angular_error_np, containment_summary
from utils.config import load_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/angle.yaml")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--component-views", default=",".join(map(str, COMPONENT_VIEWS)),
                        choices=("0,1", "1,0"))
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    cfg, _ = load_config(args.config, args.overrides)
    meta = load_meta(cfg.paths.cache_dir)
    dataset = EcalTokens(
        cfg.paths.cache_dir, args.split, meta,
        threshold_mev=cfg.data.get("runtime_threshold_mev", None),
        max_events=cfg.data.get("eval_max_events", None), subset_seed=cfg.seed + 2,
        task_mode="angle", required_fields={"angle"}, compute_fit_resid=False,
        operation="centroid angle evaluation")
    # The loader consults cfg.device nowhere; override workers explicitly when
    # running this small diagnostic on a login node.
    estimator = LayerCentroidSlope(
        cfg.geometry.data_type, args.weight_power,
        [int(v) for v in args.component_views.split(",")]).to(args.device)
    predictions, truth = [], []
    for batch in make_loader(dataset, cfg, shuffle=False):
        gpu = {key: value.to(args.device) for key, value in batch.items()
               if key not in ("run", "event")}
        predictions.append(estimator(gpu).cpu().numpy())
        truth.append(batch["angle"].numpy())
    summary = containment_summary(angular_error_np(
        np.concatenate(predictions), np.concatenate(truth)))
    print(json.dumps({
        "split": args.split,
        "events": len(dataset),
        "layer_weight_power": args.weight_power,
        "component_views": args.component_views,
        **{(key if key == "count" else key + "_mrad"):
           (value if key == "count" else 1e3 * value)
           for key, value in summary.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
