"""Blind packages (proposer side) and truth records (scorer side). PREREG 3.4 checks 2 and 4.

Blind package: blind/<task_id>/task.json, built from an allowlist. Weights (mode W only) are
written at 8 significant digits in a fixed-width format, so every package of the same (mode, d)
has the same byte length and layout; only the digits differ. The reader rejects unknown files
and fields.

Truth record: truth/<task_id>/truth.json and arrays.npz, written once (exclusive create) and
made read-only. Only harness/score.py and the verifier read truth.
"""
import hashlib, json, os, secrets, stat

import numpy as np

from . import constants as K
from .generator import Net

BLIND_SCHEMA = "blind-task-v1"  # neutral: proposers are told nothing about the study
BLIND_FILES = {"task.json"}
FIELDS_COMMON = ["schema", "task_id", "mode", "d", "H", "R", "Delta", "spec_rows_cap"]
FIELDS = {"W": FIELDS_COMMON + ["weights"], "B": FIELDS_COMMON + ["net_rows_cap"]}
WEIGHT_KEYS = ["W1", "b1", "v", "c"]
FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z, set on every blind file and directory


def new_task_id():
    return secrets.token_hex(16)


def _num(x):
    s = K.SERIAL_FORMAT.format(float(x))
    if len(s) != K.SERIAL_WIDTH:
        raise ValueError(f"value {x!r} does not fit the fixed-width format")
    return s


def _arr(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return "[" + ",".join(_num(x) for x in a) + "]"
    return "[" + ",".join(_arr(row) for row in a) + "]"


def blind_json(task_id, mode, net=None, d=None):
    """Fixed-layout JSON text for a blind package."""
    if mode not in FIELDS:
        raise ValueError(mode)
    d = net.d if net is not None else d
    R = K.radius(d)
    parts = [f'"schema": "{BLIND_SCHEMA}"', f'"task_id": "{task_id}"', f'"mode": "{mode}"', f'"d": {d}',
             f'"H": {K.H}', f'"R": {_num(R)}', f'"Delta": {_num(K.DELTA)}', f'"spec_rows_cap": {K.SPEC_ROWS_CAP}']
    if mode == "W":
        if net is None:
            raise ValueError("mode W needs the network")
        w = (f'{{"W1": {_arr(net.W1)}, "b1": {_arr(net.b1)}, "v": {_arr(net.v)}, "c": {_num(net.c)}}}')
        parts.append(f'"weights": {w}')
    else:
        parts.append(f'"net_rows_cap": {K.NET_ROWS_CAP}')
    return "{" + ", ".join(parts) + "}\n"


def write_blind(blind_root, task_id, mode, net=None, d=None):
    path = os.path.join(blind_root, task_id)
    os.makedirs(path, exist_ok=False)
    f = os.path.join(path, "task.json")
    with open(f, "x") as fh:
        fh.write(blind_json(task_id, mode, net=net, d=d))
    os.utime(f, (FIXED_MTIME, FIXED_MTIME))
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return path


def read_blind(path):
    """Strict reader: rejects unknown files, unknown or missing fields, and bad shapes."""
    files = set(os.listdir(path))
    if files != BLIND_FILES:
        raise ValueError(f"unexpected files in blind package: {sorted(files ^ BLIND_FILES)}")
    with open(os.path.join(path, "task.json")) as fh:
        return read_blind_obj(fh.read())


def read_blind_obj(text):
    """Strict parse of a blind package's task.json text (fields, schema, shapes)."""
    obj = json.loads(text)
    mode = obj.get("mode")
    if mode not in FIELDS:
        raise ValueError(f"bad mode {mode!r}")
    if list(obj.keys()) != FIELDS[mode]:
        raise ValueError(f"fields {list(obj.keys())} != allowlist {FIELDS[mode]}")
    if obj["schema"] != BLIND_SCHEMA:
        raise ValueError("bad schema")
    d, H = int(obj["d"]), int(obj["H"])
    if mode == "W":
        w = obj["weights"]
        if list(w.keys()) != WEIGHT_KEYS:
            raise ValueError(f"weight fields {list(w.keys())} != {WEIGHT_KEYS}")
        W1, b1, v = (np.array(w[k], dtype=np.float64) for k in ("W1", "b1", "v"))
        if W1.shape != (d, H) or b1.shape != (H,) or v.shape != (H,):
            raise ValueError("bad weight shapes")
        obj["weights"] = {"W1": W1, "b1": b1, "v": v, "c": float(w["c"])}
    return obj


def net_from_blind(obj):
    w = obj["weights"]
    return Net(w["W1"], w["b1"], w["v"], w["c"])


# ---------------- truth ----------------
def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def write_truth(truth_root, task_id, record, arrays):
    path = os.path.join(truth_root, task_id)
    os.makedirs(path, exist_ok=False)
    j = os.path.join(path, "truth.json")
    with open(j, "x") as fh:
        json.dump(_jsonable(record), fh, indent=1, sort_keys=True)
    npz = os.path.join(path, "arrays.npz")
    with open(npz, "xb") as fh:
        np.savez(fh, **arrays)
    for f in (j, npz):
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP)
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root, out_path):
    """sha256 of every file under root (relative paths, sorted). Committed instead of the files."""
    lines = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            lines.append(f"{sha256_file(full)}  {os.path.relpath(full, root)}")
    with open(out_path, "w") as fh:
        fh.write("\n".join(sorted(lines, key=lambda l: l.split("  ", 1)[1])) + "\n")
    return out_path


def serialised_net(net):
    """The network exactly as a blind package carries it (8 significant digits), without the JSON
    round trip. Equal to net_from_blind(read_blind(...)) for the same weights."""
    f = K.SERIAL_FORMAT.format
    r = lambda a: np.array([float(f(x)) for x in np.asarray(a, dtype=float).ravel()]).reshape(np.shape(a))
    return Net(r(net.W1), r(net.b1), r(net.v), float(f(net.c)))
