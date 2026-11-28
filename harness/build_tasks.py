"""Build the blind tasks, truth records and run plan for one LLM phase (PREREG 4; frozen at G3).

    .venv/bin/python -m harness.build_tasks pilot      # G4
    .venv/bin/python -m harness.build_tasks confirm    # G5

Writes blind/<phase>/<task_id>/task.json, truth/<phase>/<task_id>/ (scorer only), and
runs/<phase>/plan.jsonl: one line per run (task_id, model, repeat, seed), in a random order that is
the execution order. Prints only counts. Writes analysis/manifests/truth_<phase>.sha256.

pilot:   p = 1e-2, d = 32, g = 3, O0; 3 faulty origins x modes {W, B}; each task once with 9b and once with 2b (12 runs).
confirm: d = 32, g = 3. Faulty: p in {1e-5, 1e-12}, 4 origins each, every origin at {O0, O2} x {W, B}
         (16 tasks per p), 2 repeats per task (64 runs). Clean: 4 origins at {O0, O2} x {W, B}, 1 repeat
         (16 runs). Model 9b. 80 runs.
"""
import json, os, random, secrets, sys

from testbed import constants as K
from testbed import generator as gen
from testbed import serialise as ser
from testbed import tasks as T

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
D, G = 32, K.G_DEFAULT


def items_for(phase):
    items = []
    if phase == "pilot":
        for _ in range(3):
            s = gen.new_secret(); o = gen.make_origin(s, d=D, faulty=True, p=1e-2, g=G); oid = T.new_origin_id()
            for mode in ("W", "B"):
                items.append(dict(secret=s, origin=o, level="O0", mode=mode, origin_id=oid))
        return items, {"9b": 1, "2b": 1}
    if phase == "confirm":
        for faulty, ps, n in ((True, K.P_LLM, 4), (False, (None,), 4)):
            for p in ps:
                for _ in range(n):
                    s = gen.new_secret(); oid = T.new_origin_id()
                    o = gen.make_origin(s, d=D, faulty=True, p=p, g=G) if faulty else gen.make_origin(s, d=D, faulty=False)
                    for lv in ("O0", "O2"):
                        for mode in ("W", "B"):
                            items.append(dict(secret=s, origin=o, level=lv, mode=mode, origin_id=oid,
                                              repeat=2 if faulty else 1))
        return items, {"9b": None}
    raise SystemExit(f"unknown phase {phase!r}")


def main(phase):
    blind = os.path.join(ROOT, "blind", phase)
    truth = os.path.join(ROOT, "truth", phase)
    plan_path = os.path.join(ROOT, "runs", phase, "plan.jsonl")
    for p in (blind, truth, plan_path):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; tasks for a phase are built once")
    os.makedirs(blind); os.makedirs(truth); os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    items, models = items_for(phase)
    ids = T.build_tasks(items, blind, truth)
    # build_tasks writes in a random order and returns ids in that order; recover each item's repeat
    # count from the truth-free side: every task of a phase gets the same repeat count per faultiness,
    # which the plan needs, so it is read back through the scorer module (the only truth reader).
    from harness import score as SC
    plan = []
    for tid in ids:
        rep = SC.planned_repeats(truth, tid)
        for model, fixed in models.items():
            for r in range(fixed or rep):
                plan.append(dict(task_id=tid, model=model, repeat=r, seed=secrets.randbits(31)))
    random.SystemRandom().shuffle(plan)
    with open(plan_path, "x") as fh:
        for row in plan:
            fh.write(json.dumps(row) + "\n")
    os.makedirs(os.path.join(ROOT, "analysis", "manifests"), exist_ok=True)
    ser.write_manifest(truth, os.path.join(ROOT, "analysis", "manifests", f"truth_{phase}.sha256"))
    print(f"{phase}: {len(ids)} tasks, {len(plan)} planned runs")


if __name__ == "__main__":
    main(sys.argv[1])
