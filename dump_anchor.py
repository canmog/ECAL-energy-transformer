"""Rebuild the cache_full10 TEST split from the ROOT files, scalars only, and dump
the 3D-fit energies (kx_EleEne = plain fit, kx_EneL2Cor = leakage-corrected) for the
same 507,602 events the transformers are scored on.

Exact replication of data/preprocess.py: same selection (mcEne>0, contained
(stat[0]&7)==7, single shower), same seed-42 permutation split, same sorted-index
gather. The preprocess log shows ZERO zero-token drops at the 10 MeV threshold, so
the scalar-only event list is identical; this is VERIFIED against cache_m2's stored
anchor (kx_EneL2Cor) before writing.

    python dump_anchor.py          # -> runs/methodfig2/anchors_test.npz
"""
import glob as globmod
import os
import sys

import numpy as np
import uproot

BASE = "/aifs/user/data/lishanglin/chenhao"
ROOT_GLOB = "/aifs/user/data/lishanglin/datasets/full/*-full.root"
TREE = "tree3dfit"
OUT = f"{BASE}/ecalTransformer/runs/methodfig2"
M2_TEST = f"{BASE}/transformer_m2/cache_m2/test.npz"
SEED, TEST_FRAC, VAL_FRAC = 42, 0.15, 0.15
N_EXPECTED = 3384017


def _scalar(arr, idx=None):
    a = np.asarray(arr)
    if idx is not None:                      # branch is [3]; take primary shower
        a = np.stack([np.asarray(x).ravel()[idx] for x in a]) if a.dtype == object else a[:, idx]
    return a.astype(np.float64).ravel()


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(globmod.glob(ROOT_GLOB))
    if not paths:
        sys.exit(f"no ROOT files match {ROOT_GLOB}")
    print(f"reading {len(paths)} files (scalars only)")
    read = ["mcEne", "kx_ShwrStat", "kx_N_Shwr", "kx_EleEne", "kx_EneL2Cor"]
    mc_l, ele_l, l2_l = [], [], []
    n_seen = 0
    for chunk in uproot.iterate([f"{p}:{TREE}" for p in paths], read,
                                step_size=20000, library="np"):
        mc = np.asarray(chunk["mcEne"]).astype(np.float64).ravel()
        stat0 = _scalar(chunk["kx_ShwrStat"], 0).astype(np.int64)
        nsh = np.asarray(chunk["kx_N_Shwr"]).ravel().astype(np.int64)
        keep = (mc > 0) & ((stat0 & 7) == 7) & (nsh == 1)
        mc_l.append(mc[keep])
        ele_l.append(_scalar(chunk["kx_EleEne"])[keep])
        l2_l.append(_scalar(chunk["kx_EneL2Cor"])[keep])
        n_seen += len(mc)
        print(f"  processed {n_seen}, kept {sum(len(x) for x in mc_l)}", end="\r")
    print()
    energy = np.concatenate(mc_l)
    ele = np.concatenate(ele_l)
    l2 = np.concatenate(l2_l)
    N = len(energy)
    print(f"kept {N} events (expected {N_EXPECTED})")
    if N != N_EXPECTED:
        sys.exit("EVENT COUNT MISMATCH — selection replication failed, aborting")

    perm = np.random.default_rng(SEED).permutation(N)
    test = np.sort(perm[:int(TEST_FRAC * N)])          # preprocess gathers SORTED
    energy, ele, l2 = energy[test], ele[test], l2[test]

    # verification against the stored m2 anchor (same events, same split)
    z = np.load(M2_TEST)
    ok_e = np.allclose(energy, z["energy"].astype(np.float64), rtol=1e-5)
    ok_a = np.allclose(l2, z["anchor"].astype(np.float64), rtol=1e-4)
    print(f"verify vs cache_m2: energy match={ok_e}  EneL2Cor match={ok_a}")
    if not (ok_e and ok_a):
        sys.exit("VERIFICATION FAILED — not writing output")
    np.savez(f"{OUT}/anchors_test.npz", energy=energy.astype(np.float32),
             ele=ele.astype(np.float32), l2cor=l2.astype(np.float32))
    frac_bad = float(np.mean(ele <= 0))
    print(f"wrote {OUT}/anchors_test.npz  (n={len(energy)}, "
          f"kx_EleEne<=0 fraction: {frac_bad*100:.2f}%)")


if __name__ == "__main__":
    main()
