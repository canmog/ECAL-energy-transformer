"""Cache-schema contracts shared by preprocessing, loading, and entry points."""
from pathlib import Path


CACHE_SCHEMA_VERSION = 2

MODEL_INPUT_FIELDS = frozenset({"off", "tok_layer", "tok_cell", "tok_ehit"})
TASK_TARGET_FIELDS = {
    "energy": frozenset({"energy", "tok_expe", "concepts"}),
    "angle": frozenset({"angle"}),
    "joint": frozenset({"energy", "tok_expe", "concepts", "angle"}),
    "predict": frozenset(),
}
REFERENCE_FIELDS = frozenset({"fit_angle"})
IDENTITY_FIELDS = frozenset({"run", "event"})
VALID_TASK_MODES = tuple(TASK_TARGET_FIELDS)


class MissingCacheFieldsError(ValueError):
    """Raised before work starts when a requested cache capability is absent."""


def normalize_task_mode(mode):
    mode = str(mode or "energy").lower()
    if mode not in TASK_TARGET_FIELDS:
        choices = ", ".join(VALID_TASK_MODES)
        raise ValueError(f"unknown task mode {mode!r}; expected one of: {choices}")
    return mode


def required_fields_for(mode, *, extra=(), require_fit_angle=False,
                        require_identity=False):
    """Return the exact cache fields required for one operation."""
    mode = normalize_task_mode(mode)
    required = set(MODEL_INPUT_FIELDS)
    required.update(TASK_TARGET_FIELDS[mode])
    required.update(extra)
    if require_fit_angle:
        required.update(REFERENCE_FIELDS)
    if require_identity:
        required.update(IDENTITY_FIELDS)
    return frozenset(required)


def available_split_fields(cache_dir, split):
    """Inspect a packed NPZ or memory-mapped NPY split without loading its data."""
    cache = Path(cache_dir)
    split_dir = cache / split
    if split_dir.is_dir():
        return frozenset(path.stem for path in split_dir.glob("*.npy"))

    packed = cache / f"{split}.npz"
    if not packed.is_file():
        raise FileNotFoundError(
            f"cache split {split!r} not found: expected {packed} or {split_dir}/")

    import numpy as np
    with np.load(packed) as arrays:
        return frozenset(arrays.files)


def require_fields(available, required, *, cache_dir, split, operation):
    """Fail with an actionable error rather than fabricating missing targets."""
    available = frozenset(available)
    required = frozenset(required)
    missing = sorted(required - available)
    if missing:
        raise MissingCacheFieldsError(
            f"{operation} cannot use cache split {split!r} at {cache_dir!r}: "
            f"missing required field(s) {missing}; available fields are "
            f"{sorted(available)}")


def require_meta(meta, required, *, cache_dir, operation):
    missing = sorted(set(required) - set(meta))
    if missing:
        raise MissingCacheFieldsError(
            f"{operation} cannot use cache metadata at {cache_dir!r}: "
            f"missing required metadata field(s) {missing}")
