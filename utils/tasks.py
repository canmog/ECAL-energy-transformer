"""Resolve the explicit energy, angle, and joint task contracts."""
from data.schema import normalize_task_mode


PRIMARY_BY_MODE = {
    "energy": ("energy",),
    "angle": ("angle",),
    "joint": ("energy", "angle"),
    "predict": (),
}
DEFAULT_AUXILIARY = {
    "energy": ("recon", "concept"),
    "angle": (),
    "joint": ("recon", "concept"),
    "predict": (),
}
ALLOWED_TASKS = frozenset({"energy", "angle", "recon", "concept"})


def task_mode(cfg):
    task = cfg.get("task", None)
    return normalize_task_mode(task.get("mode", "energy") if task else "energy")


def active_tasks(cfg):
    """Return trainable tasks, omitting explicitly zero-weight auxiliaries."""
    mode = task_mode(cfg)
    task = cfg.get("task", None)
    auxiliary = (task.get("auxiliary", list(DEFAULT_AUXILIARY[mode]))
                 if task else list(DEFAULT_AUXILIARY[mode]))
    auxiliary = list(auxiliary or [])
    unknown = sorted(set(auxiliary) - ALLOWED_TASKS)
    if unknown:
        raise ValueError(f"unknown task.auxiliary entries: {unknown}")

    primary = list(PRIMARY_BY_MODE[mode])
    weighting = str(cfg.loss.get(
        "weighting", "uncertainty" if cfg.loss.get("uncertainty_weighting", True)
        else "fixed")).lower()
    weights = cfg.loss.get("weights", None)
    weights = weights.to_dict() if hasattr(weights, "to_dict") else (weights or {})
    if weighting == "normalized":
        for name in primary:
            if float(weights.get(name, 1.0)) == 0.0:
                raise ValueError(
                    f"primary task {name!r} cannot have zero loss weight")
        auxiliary = [name for name in auxiliary
                     if float(weights.get(name, 1.0)) != 0.0]

    ordered = []
    for name in primary + auxiliary:
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def validate_task_heads(cfg, tasks=None):
    tasks = tuple(tasks or active_tasks(cfg))
    for name in ("energy", "angle", "recon"):
        if name in tasks and not getattr(cfg.heads, name).enabled:
            raise ValueError(
                f"task {name!r} is active but heads.{name}.enabled is false")
    return tasks


def cache_fields_for_tasks(tasks, *, fit_quality=False):
    fields = set()
    if "energy" in tasks:
        fields.add("energy")
    if "angle" in tasks:
        fields.add("angle")
    if "recon" in tasks:
        fields.add("tok_expe")
    if "concept" in tasks:
        fields.add("concepts")
    if fit_quality:
        fields.add("tok_expe")
    return frozenset(fields)
