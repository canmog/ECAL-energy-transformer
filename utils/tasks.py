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
    if isinstance(auxiliary, str):
        raise ValueError(
            "task.auxiliary must be a YAML list, for example [recon, concept], "
            "not a string")
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
        if name not in tasks:
            continue
        head = cfg.heads.get(name, None)
        if head is None:
            raise ValueError(
                f"task {name!r} is active but required config section "
                f"heads.{name} is missing")
        if not head.get("enabled", False):
            raise ValueError(
                f"task {name!r} is active but heads.{name}.enabled is false")
    return tasks


def validate_training_contract(cfg):
    """Validate every config capability consumed by training before data loading."""
    mode = task_mode(cfg)
    tasks = validate_task_heads(cfg)
    for name in tasks:
        section = cfg.loss.get(name, None)
        if section is None:
            raise ValueError(
                f"task {name!r} is active but required config section "
                f"loss.{name} is missing")
        if section.get("delta", None) is None:
            raise ValueError(
                f"task {name!r} is active but required config field "
                f"loss.{name}.delta is missing")

    task = cfg.get("task", None)
    selection = (task.get("selection", "angle" if mode == "joint" else mode)
                 if task else mode)
    if selection not in ("energy", "angle"):
        raise ValueError(
            f"task.selection={selection!r} is invalid; expected 'energy' or 'angle'")
    if selection not in tasks:
        raise ValueError(
            f"task.selection={selection!r} is not one of the active tasks {tasks}")
    return mode, tasks, selection


def cache_fields_for_tasks(tasks, *, fit_quality=False,
                           angle_energy_weight=False):
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
    if angle_energy_weight:
        fields.add("energy")
    return frozenset(fields)
