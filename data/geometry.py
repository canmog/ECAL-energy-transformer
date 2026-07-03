"""AMS-02 ECAL cell geometry — physical coordinates, NOT raw indices.

Source of truth: ../geo.md (DeadSideCorrDB) + the AMS-02 ECAL papers.

Detector facts that drive this module
-------------------------------------
* 648 x 648 x 166.5 mm active volume = 17 X0; 0.7 nuclear interaction lengths.
* 9 superlayers, each 18.5 mm thick. Fibres in ONE superlayer all run the SAME
  direction. The view therefore alternates at the *superlayer* level:
      superlayer = ilayer // 2          (two longitudinal readouts per superlayer)
      readout    = ilayer %  2          (which of the two depth samplings)
      view       = superlayer % 2       (0 = X, 1 = Y)   <-- per SUPERLAYER, not per layer
  -> X = superlayers {0,2,4,6,8} (5),  Y = superlayers {1,3,5,7} (4).  [confirmed]
* Readout granularity: 18 longitudinal samplings x 72 lateral cells. Each anode
  covers a 9 x 9 mm cell (pitch = 0.9 cm) and ~1 X0 in depth (depth_X0 ~ ilayer).
* OffSetMC[9][2] is indexed [superlayer][readout=ilayer%2] (per-layer alignment,
  microns), NOT [superlayer][view]. CommonOffSetMC[2] is the per-view common shift.

A single layer measures only ONE projection, so a cell (ilayer, icell) is a strip:
known depth z and known transverse coordinate t along the view axis; the orthogonal
coordinate is unmeasured. The Transformer fuses the two views via attention — which
is exactly what the bottleneck axis-concepts (ShwrX0/Y0/Z0) then read out.
"""
import numpy as np

N_LAYER = 18
N_CELL = 72
PITCH_CM = 0.9                     # 9 mm anode pitch  (648 mm / 72)  [confirmed]
UM2CM = 1.0e-4                     # geo.md
ENEDEP_THRESHOLD_STD = 5.0         # geo.md standard reco threshold (MeV); we go lower

# Per-layer depth z (cm), from geo.md Ecal_Z[18] (the +/-0.005 already folded in).
ECAL_Z = np.array([
    -143.220, -144.130, -145.070, -145.980, -146.920, -147.830,
    -148.770, -149.680, -150.620, -151.530, -152.470, -153.380,
    -154.320, -155.230, -156.170, -157.080, -158.020, -158.930,
], dtype=np.float64)

# Per-view common shift (cm); index = view (0=X, 1=Y).
COMMON_OFFSET = {
    "MC":  np.array([-0.13,  0.075], dtype=np.float64),
    "TB":  np.array([-0.17,  0.085], dtype=np.float64),
    "ISS": np.array([-0.13,  0.075], dtype=np.float64),
}

# Per-(superlayer, readout=ilayer%2) fine alignment. [superlayer][ilayer%2]
# Units differ per dataset (geo.md): MC/ISS are microns, TB is already cm —
# OFFSET_UNIT converts each to cm so build_geometry_table works for all three.
OFFSET = {
    "MC": np.array([
        [ 843.7, 1150.9], [-597.1, -1183.0], [  -1.6,  118.5],
        [-503.3, -634.0], [ 873.0,   402.6], [-350.4, -383.1],
        [-252.0, -386.3], [1613.9,  1373.9], [-192.5, -1333.3],
    ], dtype=np.float64),
    "ISS": np.array([
        [-108.7,   14.7], [  46.1, -223.1], [ -60.5,  -57.1],
        [ 157.5,   22.8], [  25.5,  -86.3], [ 215.4,  123.4],
        [  25.6,  -67.1], [ 245.7,    2.2], [ 170.7,  -70.3],
    ], dtype=np.float64),
    "TB": np.array([
        [-0.009,  0.005], [ 0.005, -0.005], [-0.005, -0.002],
        [ 0.001,  0.005], [-0.000, -0.002], [ 0.004,  0.003],
        [ 0.003, -0.002], [ 0.004, -0.008], [-0.001,  0.005],
    ], dtype=np.float64),
}
OFFSET_UNIT = {"MC": UM2CM, "ISS": UM2CM, "TB": 1.0}

# Small per-superlayer fibre rotation (rad). Couples the unmeasured coordinate;
# negligible (<0.3 mm over the half-width) and left out of the 1-D strip position.
FIBER_ROTATION = np.array([
    0.0004797, 0.0009035, 0.0001849, 0.0002104, -0.0007271,
    3.608e-05, -0.000221, 0.0003725, -0.001776,
], dtype=np.float64)


def superlayer_of(ilayer):
    return ilayer // 2


def view_of(ilayer):
    """0 = X (superlayers 0,2,4,6,8), 1 = Y (superlayers 1,3,5,7)."""
    return (ilayer // 2) % 2


def build_geometry_table(data_type="MC"):
    """Return arrays indexed [ilayer, icell] with physical coordinates.

    Returns a dict of [18, 72] arrays:
        t_cm   : transverse coordinate along the layer's view axis (cm)
        z_cm   : longitudinal depth (cm), constant per layer
        view   : 0 (X) or 1 (Y), constant per layer
        depth  : X0 depth proxy ~ ilayer (0..17), constant per layer
    """
    common = COMMON_OFFSET[data_type]
    offset = OFFSET[data_type] * OFFSET_UNIT[data_type]

    t_cm = np.zeros((N_LAYER, N_CELL), dtype=np.float64)
    z_cm = np.zeros((N_LAYER, N_CELL), dtype=np.float64)
    view = np.zeros((N_LAYER, N_CELL), dtype=np.int64)
    depth = np.zeros((N_LAYER, N_CELL), dtype=np.float64)

    cells = np.arange(N_CELL, dtype=np.float64)
    base = (cells - (N_CELL - 1) / 2.0) * PITCH_CM          # centred, 1-indexed COG convention

    for il in range(N_LAYER):
        sl = il // 2
        rd = il % 2
        vw = (sl % 2)
        t_cm[il] = base + common[vw] + offset[sl, rd]
        z_cm[il] = ECAL_Z[il]
        view[il] = vw
        depth[il] = float(il)                              # ~1 X0 per layer

    return {"t_cm": t_cm, "z_cm": z_cm, "view": view, "depth": depth}


# Normalisation constants for token features (kept here so dataset + probe agree).
_Z_MID = float(ECAL_Z.mean())
_Z_HALF = float((ECAL_Z.max() - ECAL_Z.min()) / 2.0 + 1e-6)
_T_HALF = (N_CELL / 2.0) * PITCH_CM + 1.0                   # ~ 33 cm


def token_geometry_features(ilayer, icell, table):
    """Build the per-token geometry feature block (used by the dataset).

    Returns [t_norm, z_norm, depth_norm, view_x, view_y] for each token.
    `ilayer`, `icell` are integer arrays of equal length; `table` from
    build_geometry_table().
    """
    t = table["t_cm"][ilayer, icell]
    z = table["z_cm"][ilayer, icell]
    vw = table["view"][ilayer, icell]
    dp = table["depth"][ilayer, icell]

    t_norm = t / _T_HALF
    z_norm = (z - _Z_MID) / _Z_HALF
    depth_norm = dp / (N_LAYER - 1)
    view_x = (vw == 0).astype(np.float32)
    view_y = (vw == 1).astype(np.float32)
    return np.stack([t_norm, z_norm, depth_norm, view_x, view_y], axis=-1).astype(np.float32)


GEOM_FEATURE_DIM = 5  # [t_norm, z_norm, depth_norm, view_x, view_y]


def mirror_centers(table):
    """Per-view mean transverse array centre c̄_v (cm), keyed by view (0=X, 1=Y).

    A cell-index flip (icell -> 71-icell) mirrors each layer about its OWN array
    centre c_l = common + offset (the centred base grid is symmetric), not about
    the global 0. Concept targets measured in global coordinates must therefore
    flip about the view's mean centre:  x -> 2*c̄_v - x  (the naive -x is off by
    2*c̄ ~ 2 mm). Slopes still flip exactly (-k). Residual after this fix is only
    the per-layer scatter of c_l about c̄_v, which averages out over a shower.
    """
    out = {}
    for vw in (0, 1):
        rows = [il for il in range(N_LAYER) if int(table["view"][il, 0]) == vw]
        out[vw] = float(np.mean([table["t_cm"][il].mean() for il in rows]))
    return out


def t_norm_table(table):
    """(1296,) lookup of normalised t per pos_id = layer*72 + cell (for the probe's
    exact mirror: the flipped cell's TRUE coordinate, not just -t_norm)."""
    return (table["t_cm"] / _T_HALF).astype(np.float32).reshape(-1)


if __name__ == "__main__":
    # Quick eyeball check — run on the node: python -m data.geometry
    tab = build_geometry_table("MC")
    print("view per layer (0=X,1=Y):", tab["view"][:, 0].tolist())
    nx = int((tab["view"][:, 0] == 0).sum())
    ny = int((tab["view"][:, 0] == 1).sum())
    print(f"X layers={nx} (expect 10),  Y layers={ny} (expect 8)")
    print("superlayers X:", sorted({il // 2 for il in range(N_LAYER) if view_of(il) == 0}))
    print("superlayers Y:", sorted({il // 2 for il in range(N_LAYER) if view_of(il) == 1}))
    print("t range (cm): [%.2f, %.2f]" % (tab["t_cm"].min(), tab["t_cm"].max()))
    print("z range (cm): [%.2f, %.2f]" % (tab["z_cm"].min(), tab["z_cm"].max()))
