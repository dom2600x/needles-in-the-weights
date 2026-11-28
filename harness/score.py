"""Scorer: the only code that reads truth (PREREG 3 rule; section 8 outcomes).

Primary success: a submitted counterexample x with ||x|| <= R (1 + BALL_TOL) and
|net(x) - spec(x)| > Delta on the ORIGINAL float64 network. Localisation: the submitted unit (or,
for grouped answers, any unit of the submitted group) is a needle unit.
"""
import json, os

import numpy as np

from testbed import constants as K
from testbed.generator import Net


def verify_counterexample(x, net, teacher, R):
    if x is None:
        return False
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape != (net.d,) or not np.all(np.isfinite(x)):
        return False
    if np.linalg.norm(x) > R * (1 + K.BALL_TOL):
        return False
    return bool(abs(float(net(x[None])[0]) - float(teacher(x[None])[0])) > K.DELTA)


def score(answer, net, teacher, R, needle_units):
    """answer: dict with optional keys counterexample, unit, group, faulty."""
    group = answer.get("group") or ([answer["unit"]] if answer.get("unit") is not None else [])
    return dict(success=verify_counterexample(answer.get("counterexample"), net, teacher, R),
                localised=bool(set(int(u) for u in group) & set(needle_units)),
                faulty_pred=answer.get("faulty"))


def load_truth(truth_root, task_id):
    """Read a truth record and the original networks for one task."""
    path = os.path.join(truth_root, task_id)
    rec = json.load(open(os.path.join(path, "truth.json")))
    a = np.load(os.path.join(path, "arrays.npz"))
    net = Net(a["W1"], a["b1"], a["v"], float(a["c"]))
    teacher = Net(a["tW1"], a["tb1"], a["tv"], float(a["tc"]))
    return rec, net, teacher
