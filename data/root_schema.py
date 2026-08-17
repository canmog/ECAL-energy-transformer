"""Strict ROOT input contracts shared by inspection and cache builders."""
import uproot


class MissingRootBranchesError(ValueError):
    """Raised before preprocessing when a configured ROOT input is incomplete."""


def required_root_branches(cfg):
    data = cfg.data
    required = {
        data.ehit_branch,
        data.expehit_branch,
        data.target_branch,
        data.stat_branch,
        data.nshwr_branch,
        data.ref_branch,
    }
    required.update(data.get("angle_branches", []) or [])
    required.update(data.get("fit_angle_branches", []) or [])
    run_branch = data.get("run_branch", None)
    event_branch = data.get("event_branch", None)
    if bool(run_branch) != bool(event_branch):
        raise ValueError(
            "data.run_branch and data.event_branch must either both be set or "
            "both be absent")
    if run_branch:
        required.update((run_branch, event_branch))
    required.update(
        concept.branch for concept in data.concepts if concept.get("branch"))
    return frozenset(required)


def require_root_branches(paths, tree_name, required, operation="ROOT preprocessing"):
    """Inspect every file before processing so late files cannot fail mid-build."""
    required = frozenset(required)
    for path in paths:
        try:
            root_file = uproot.open(path)
        except Exception as exc:
            raise MissingRootBranchesError(
                f"{operation} cannot open ROOT file {path!r}: {exc}") from exc
        with root_file:
            if tree_name not in root_file:
                raise MissingRootBranchesError(
                    f"{operation} cannot use ROOT file {path!r}: missing tree "
                    f"{tree_name!r}; available top-level objects are "
                    f"{sorted(root_file.keys(cycle=False))}")
            available = frozenset(root_file[tree_name].keys())
            missing = sorted(required - available)
            if missing:
                raise MissingRootBranchesError(
                    f"{operation} cannot use ROOT tree {tree_name!r} in {path!r}: "
                    f"missing required branch(es) {missing}; configured branches "
                    "must be present in every input file")

