"""Execute a phase's run plan (PREREG_v1 4.1): rows in plan order, one at a time, each in a fresh
process (python -m harness.agent run ...). Resumable: a row whose seed already has a completed run
in runs/<phase>/runs.jsonl is skipped. A harness crash (non-zero exit, no completed record) is
re-run with the same seed, at most MAX_RETRIES times, and logged in runs/<phase>/driver.jsonl.

    .venv/bin/python -m harness.run_plan pilot [--limit N]

Reads no truth; it only passes the truth directory to the agent, which reads oracle functions
through harness.score.
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MAX_RETRIES = 2


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def done_seeds(phase):
    p = os.path.join(ROOT, "runs", phase, "runs.jsonl")
    if not os.path.exists(p):
        return set()
    return {json.loads(l)["seed"] for l in open(p) if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    plan = [json.loads(l) for l in open(os.path.join(ROOT, "runs", a.phase, "plan.jsonl"))]
    drv = open(os.path.join(ROOT, "runs", a.phase, "driver.jsonl"), "a")
    n = 0
    for i, row in enumerate(plan):
        if row["seed"] in done_seeds(a.phase):
            continue
        if a.limit is not None and n >= a.limit:
            break
        for attempt in range(MAX_RETRIES + 1):
            t0 = time.time()
            cmd = [sys.executable, "-m", "harness.agent", "run", "--task", os.path.join(ROOT, "blind", a.phase, row["task_id"]),
                   "--truth", os.path.join(ROOT, "truth", a.phase), "--model", row["model"], "--seed", str(row["seed"]),
                   "--phase", a.phase]
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            ok = p.returncode == 0 and row["seed"] in done_seeds(a.phase)
            drv.write(json.dumps(dict(utc=utc(), plan_index=i, model=row["model"], seed=row["seed"], attempt=attempt,
                                      returncode=p.returncode, ok=ok, secs=round(time.time() - t0, 1),
                                      stdout=p.stdout[-500:], stderr=p.stderr[-2000:] if not ok else "")) + "\n")
            drv.flush()
            print(f"[{utc()}] plan {i + 1}/{len(plan)} {row['model']}: "
                  f"{'ok ' + p.stdout.strip().splitlines()[-1] if ok else 'CRASH rc=' + str(p.returncode)}", flush=True)
            if ok:
                break
        else:
            print(f"plan {i + 1}: failed {MAX_RETRIES + 1} times; stopping", flush=True)
            raise SystemExit(1)
        n += 1
    print(f"[{utc()}] done: {len(done_seeds(a.phase))} of {len(plan)} runs complete", flush=True)


if __name__ == "__main__":
    main()
