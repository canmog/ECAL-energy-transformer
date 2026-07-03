"""Slice the 11-concept strict cache (cache_s11) down to its first 8 concepts
(cache_s08) for a perfectly controlled A/B: identical events, only concept count
differs. The 3 new concepts (tmax, lat_width, long_width) are appended last, so
[:8] keeps exactly the original 8.
"""
import json
import os
import numpy as np

SRC, DST = "cache_s11", "cache_s08"
os.makedirs(DST, exist_ok=True)
for sp in ("train", "val", "test"):
    z = dict(np.load(f"{SRC}/{sp}.npz"))
    z["concepts"] = z["concepts"][:, :8]
    np.savez(f"{DST}/{sp}.npz", **z)
m = json.load(open(f"{SRC}/meta.json"))
m["n_concepts"] = 8
for k in ("concept_names", "concept_reflect_x", "concept_reflect_y",
          "concept_coord", "concept_mean", "concept_std"):
    m[k] = m[k][:8]
json.dump(m, open(f"{DST}/meta.json", "w"), indent=2)
print("wrote", DST, "n_concepts=8 names=", m["concept_names"])
