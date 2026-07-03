"""Print a one-line summary per sweep run: best bias-aware val metric (model
selection) + test-set numbers from evaluate.py. Sorted by val metric."""
import csv
import glob
import json
import os

rows = []
for d in sorted(glob.glob("runs/s*")):
    if not os.path.isdir(d):
        continue
    csvp = os.path.join(d, "metrics.csv")
    if not os.path.exists(csvp):
        continue
    best = None
    with open(csvp) as f:
        for r in csv.DictReader(f):
            m = float(r["val_metric"])
            if best is None or m < best[0]:
                best = (m, int(r["epoch"]), float(r["val_res"]), float(r["val_bias"]))
    if best is None:
        continue
    test = {}
    mj = os.path.join(d, "metrics.json")
    if os.path.exists(mj):
        with open(mj) as f:
            test = json.load(f)
    rows.append((os.path.basename(d), best, test))

rows.sort(key=lambda r: r[1][0])
print(f"{'run':<12} {'val_metric':>10} {'@ep':>4} {'val_res':>8} {'val_bias':>9} "
      f"{'test_res':>9} {'test_binavg':>11} {'test_bias':>10}")
for name, (m, ep, res, bias), t in rows:
    tr = t.get("overall_res")
    tb = t.get("overall_bias")
    ta = t.get("binwise_mean_res")
    fmt = lambda x: f"{x*100:9.2f}%" if isinstance(x, float) else f"{'-':>10}"
    print(f"{name:<12} {m*100:9.2f}% {ep:>4} {res*100:7.2f}% {bias*100:+8.2f}% "
          f"{fmt(tr)} {fmt(ta)} {fmt(tb)}")
