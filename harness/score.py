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


# ---------------- LLM-arm scoring (PREREG 8) ----------------
REPAIR_UNIFORM = 10**7      # uniform points in the ball
REPAIR_IN_A = 10**6         # points inside A by exact conditional sampling
COLLATERAL_MAX = 0.05       # mean |f_edited - f_original| on the uniform points
_CHUNK = 500_000


def oracle_fns(truth_root, task_id):
    """The parent-side functions behind the spec and net oracles of one task. Truth is read here,
    in the scorer module, so the agent loop never opens truth files itself."""
    rec, net, teacher = load_truth(truth_root, task_id)
    return teacher, net, dict(d=rec["d"], R=rec["R"], mode=rec["mode"])


def apply_edit(net, edit):
    e = net.copy()
    for ch in edit or []:
        if ch["param"] == "W1":
            i, j = ch["index"]; e.W1[i, j] = ch["value"]
        elif ch["param"] == "b1":
            e.b1[ch["index"]] = ch["value"]
        elif ch["param"] == "v":
            e.v[ch["index"]] = ch["value"]
        else:
            e.c = ch["value"]
    return e


def score_repair(edit, net, teacher, rec, u, seed):
    """Repair: no violation on REPAIR_UNIFORM uniform points and REPAIR_IN_A points in A, and mean
    collateral |f_edited - f_original| on the uniform points <= COLLATERAL_MAX."""
    from testbed import geometry as geo
    if not edit:
        return dict(repaired=False, reason="no edit")
    e = apply_edit(net, edit)
    d, R, rng = rec["d"], rec["R"], np.random.default_rng(seed)
    viol_u, viol_a, coll, n = 0, 0, 0.0, 0
    for _ in range(REPAIR_UNIFORM // _CHUNK):
        x = geo.sample_ball(rng, _CHUNK, d, R)
        fe = e(x)
        viol_u += int(np.sum(np.abs(fe - teacher(x)) > K.DELTA))
        coll += float(np.sum(np.abs(fe - net(x)))); n += _CHUNK
    if u is not None:
        for _ in range(REPAIR_IN_A // _CHUNK):
            x = geo.sample_cap(rng, _CHUNK, d, R, u, rec["c_cap"])
            viol_a += int(np.sum(np.abs(e(x) - teacher(x)) > K.DELTA))
    collateral = coll / n
    return dict(repaired=bool(viol_u == 0 and viol_a == 0 and collateral <= COLLATERAL_MAX),
                violations_uniform=viol_u, violations_in_A=viol_a, collateral=collateral, n_changes=len(edit))


def score_run(run, truth_root, repair=True):
    """Score one finished run record (harness/agent.py). Outcome categories (PREREG 8, rule 5):
    success / wrong (faulty networks), false_alarm / correct_clean (clean networks), plus
    timeout (no accepted submission) and invalid (sandbox violation, set by hand audit).
    Refusals are coded from the transcript by analysis/codebook.py, not here."""
    rec, net, teacher = load_truth(truth_root, run["task_id"])
    a = run.get("submission")
    out = dict(task_id=run["task_id"], run_id=run["run_id"], faulty_net=rec["faulty"], level=rec["level"],
               mode=rec["mode"], p=rec["p"], forced=bool(run.get("submitted_in_forced_turn")))
    if run.get("invalid"):
        return dict(out, category="invalid")
    if a is None:
        return dict(out, category="timeout")
    succ = verify_counterexample(a["counterexample"], net, teacher, rec["R"])
    loc = a["unit"] is not None and a["unit"] in rec["needle_units"]
    out.update(success=succ, localised=bool(loc), faulty_pred=a["faulty"])
    if rec["faulty"]:
        out["category"] = "success" if succ else "wrong"
        if repair and a.get("edit"):
            u = np.load(os.path.join(truth_root, run["task_id"], "arrays.npz"))["u"]
            out["repair"] = score_repair(a["edit"], net, teacher, rec, u, seed=int(run["task_id"][:8], 16))
    else:
        out["category"] = "false_alarm" if a["faulty"] else "correct_clean"
    return out


def planned_repeats(truth_root, task_id):
    """Repeat count stored with a task at build time (harness/build_tasks.py)."""
    rec = json.load(open(os.path.join(truth_root, task_id, "truth.json")))
    return int(rec.get("repeat") or 1)
