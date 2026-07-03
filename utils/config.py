"""Minimal YAML config -> attribute-access namespace, with CLI overrides.

    cfg = load_config("config/base.yaml", overrides=["train.lr=1e-4", "model.d_model=256"])
    cfg.train.lr            # 1e-4 (float)
    cfg.model.d_model       # 256 (int)
    cfg.data.selection.select_contained   # nested access

Lists of dicts (e.g. data.concepts) stay as plain lists of NS for iteration.
"""
import ast
import copy

import yaml


class NS:
    """Recursive namespace with dict-like .get and round-trip to dict."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _wrap(v))

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            out[k] = _unwrap(v)
        return out

    def __repr__(self):
        return f"NS({self.to_dict()!r})"


def _wrap(v):
    if isinstance(v, dict):
        return NS(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _unwrap(v):
    if isinstance(v, NS):
        return v.to_dict()
    if isinstance(v, list):
        return [_unwrap(x) for x in v]
    return v


def _coerce(s):
    """Turn a CLI string value into a python literal when possible.

    Handle YAML-style scalars first (true/false/null/yes/no) — ast.literal_eval
    does NOT understand lowercase 'false', which would otherwise leak through as
    the truthy string "false".
    """
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _set_path(d, dotted, value):
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path, overrides=None):
    with open(path) as f:
        raw = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        _set_path(raw, key.strip(), _coerce(val.strip()))
    return NS(copy.deepcopy(raw)), raw
