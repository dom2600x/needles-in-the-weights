import json, os

import numpy as np

from harness import build_tasks as BT
from harness import score as SC
from testbed import serialise as ser


def test_build_and_score(tmp_path, monkeypatch):
    monkeypatch.setattr(BT, "ROOT", str(tmp_path))
    os.makedirs(tmp_path / "analysis")
    BT.main("pilot")
    BT.main("confirm")
    plan = [json.loads(l) for l in open(tmp_path / "runs" / "pilot" / "plan.jsonl")]
    assert len(plan) == 12 and {r["model"] for r in plan} == {"9b", "2b"}
    cplan = [json.loads(l) for l in open(tmp_path / "runs" / "confirm" / "plan.jsonl")]
    assert len(cplan) == 80 and len({r["task_id"] for r in cplan}) == 48
    truth = str(tmp_path / "truth" / "confirm")
    recs = {t: SC.load_truth(truth, t)[0] for t in {r["task_id"] for r in cplan}}
    n_runs = {t: sum(r["task_id"] == t for r in cplan) for t in recs}
    assert all(n_runs[t] == (2 if recs[t]["faulty"] else 1) for t in recs)
    assert sorted({(r["p"], r["level"], r["mode"]) for r in recs.values() if r["faulty"]}) == sorted(
        {(p, lv, m) for p in (1e-5, 1e-12) for lv in ("O0", "O2") for m in ("W", "B")})
    # blind side carries no truth
    tid = next(t for t, r in recs.items() if r["faulty"] and r["mode"] == "W" and r["level"] == "O0" and r["p"] == 1e-5)
    blind = ser.read_blind(str(tmp_path / "blind" / "confirm" / tid))
    assert "needle" not in json.dumps({k: v for k, v in blind.items() if k != "weights"})
    # scoring: a correct answer, then a repair that removes the needle
    rec, net, teacher = SC.load_truth(truth, tid)
    u = np.load(os.path.join(truth, tid, "arrays.npz"))["u"]
    j = rec["needle_units"][0]
    x = (rec["R"] * net.W1[:, j] / np.linalg.norm(net.W1[:, j])).tolist()
    monkeypatch.setattr(SC, "REPAIR_UNIFORM", 10**6)
    monkeypatch.setattr(SC, "REPAIR_IN_A", 10**6)
    run = dict(task_id=tid, run_id="r1", submission=dict(faulty=True, counterexample=x, unit=j,
                                                         edit=[dict(param="v", index=j, value=0.0)]))
    s = SC.score_run(run, truth)
    assert s["category"] == "success" and s["localised"] and s["repair"]["repaired"]
    assert s["repair"]["collateral"] < 0.05
    bad = SC.score_run(dict(run, submission=dict(run["submission"], counterexample=[0.0] * 32,
                                                 edit=[dict(param="v", index=(j + 1) % 96, value=0.0)])), truth)
    assert bad["category"] == "wrong" and not bad["repair"]["repaired"]
    assert SC.score_run(dict(run, submission=None), truth)["category"] == "timeout"
    ctid = next(t for t, r in recs.items() if not r["faulty"])
    assert SC.score_run(dict(task_id=ctid, run_id="r2", submission=dict(faulty=True, counterexample=None, unit=None,
                                                                         edit=None)), truth)["category"] == "false_alarm"
