"""Answer schema for submit(answer) (PREREG 7, 8). Validation runs in the parent when the proposer
calls submit; a ValueError's text is returned to the proposer and the run continues.

    {"faulty": bool,
     "counterexample": [d numbers] or None,      # a point x with ||x|| <= R
     "unit": int in [0, H) or None,              # the hidden unit responsible
     "edit": [ {"param": "W1"|"b1"|"v"|"c", "index": ..., "value": number}, ... ] or None}

index: [i, j] for W1 (row i < d, column j < H), j for b1 and v, null (or omitted) for c.
Every key must be present; unknown keys are rejected.
"""
import math

KEYS = ("faulty", "counterexample", "unit", "edit")
MAX_EDITS = 256


def _num(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"submit: {what} must be a finite number, got {x!r}")
    return float(x)


def _int(x, lo, hi, what):
    if isinstance(x, bool) or not isinstance(x, int) and not (isinstance(x, float) and x.is_integer()):
        raise ValueError(f"submit: {what} must be an integer, got {x!r}")
    x = int(x)
    if not lo <= x < hi:
        raise ValueError(f"submit: {what} must be in [{lo}, {hi}), got {x}")
    return x


def validate(answer, d, H):
    if not isinstance(answer, dict):
        raise ValueError("submit: the answer must be a dict with keys " + ", ".join(KEYS))
    extra, missing = set(answer) - set(KEYS), [k for k in KEYS if k not in answer]
    if extra or missing:
        raise ValueError(f"submit: keys must be exactly {list(KEYS)}; missing {missing}, unknown {sorted(extra)}")
    if not isinstance(answer["faulty"], bool):
        raise ValueError("submit: 'faulty' must be true or false")
    out = {"faulty": answer["faulty"]}
    ce = answer["counterexample"]
    if ce is not None:
        if not isinstance(ce, list) or len(ce) != d:
            raise ValueError(f"submit: 'counterexample' must be a list of {d} numbers or None")
        ce = [_num(v, "each counterexample entry") for v in ce]
    out["counterexample"] = ce
    out["unit"] = None if answer["unit"] is None else _int(answer["unit"], 0, H, "'unit'")
    ed = answer["edit"]
    if ed is not None:
        if not isinstance(ed, list) or len(ed) > MAX_EDITS:
            raise ValueError(f"submit: 'edit' must be a list of at most {MAX_EDITS} entries or None")
        norm = []
        for e in ed:
            if not isinstance(e, dict) or set(e) - {"param", "index", "value"} or "param" not in e or "value" not in e:
                raise ValueError("submit: each edit must be a dict with keys param, index, value")
            prm, idx = e["param"], e.get("index")
            if prm == "W1":
                if not isinstance(idx, list) or len(idx) != 2:
                    raise ValueError("submit: a W1 edit needs index [i, j]")
                idx = [_int(idx[0], 0, d, "W1 row index"), _int(idx[1], 0, H, "W1 column index")]
            elif prm in ("b1", "v"):
                idx = _int(idx[0] if isinstance(idx, list) and len(idx) == 1 else idx, 0, H, f"{prm} index")
            elif prm == "c":
                if idx not in (None, [], 0):
                    raise ValueError("submit: a c edit takes no index")
                idx = None
            else:
                raise ValueError(f"submit: edit param must be one of W1, b1, v, c, got {prm!r}")
            norm.append({"param": prm, "index": idx, "value": _num(e["value"], "edit value")})
        ed = norm
    out["edit"] = ed
    return out
