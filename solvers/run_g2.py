"""G2 scripted ladder runner: resumable, checkpointed batches (PREREG 4, 5; G2 protocol).

    .venv/bin/python -m solvers.run_g2 origins                 # once: draw all origin secrets
    .venv/bin/python -m solvers.run_g2 fuzz --d 32 --p 1e-12   # one (d, p) batch of fuzzing
    .venv/bin/python -m solvers.run_g2 struct --d 32 --p 1e-12 # one (d, p) batch of structural scans
    .venv/bin/python -m solvers.run_g2 clean --d 32            # clean networks (calibration + false alarm)
    .venv/bin/python -m solvers.run_g2 status

Outputs (runs/g2/, gitignored): origins.jsonl (secrets; truth), fuzz.jsonl, struct.jsonl,
clean.jsonl, batches.jsonl (timing and pmset thermal state). Every record carries a key; work
already recorded is skipped, so an interrupted batch resumes where it stopped.
"""
import argparse, json, os, subprocess, time
from datetime import datetime, timezone

import numpy as np

from harness import score as SC
from solvers import fuzz as F
from solvers import structural as S
from testbed import constants as K
from testbed import generator as gen
from testbed import serialise as ser

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
OUT = os.path.join(ROOT, "runs", "g2")
N_FAULTY = 200
N_CLEAN = 400            # first 200 calibrate, last 200 measure false alarms
QMAX = 10**6
VARIANTS = [(lv, "mixed") for lv in K.LEVELS] + [("O2", "cancel_only"), ("O3", "cancel_only")]


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def therm():
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        return " | ".join(l.strip() for l in out.splitlines() if l.strip())
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def _path(name):
    return os.path.join(OUT, name)


def _append(name, rec):
    with open(_path(name), "a") as fh:
        fh.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def _read(name):
    if not os.path.exists(_path(name)):
        return []
    with open(_path(name)) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def pkey(p):
    return f"{p:.0e}"


def origins():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(_path("origins.jsonl")):
        raise SystemExit("origins.jsonl exists; origins are drawn once and never redrawn")
    with open(_path("origins.jsonl"), "x") as fh:
        for d in K.D_VALUES:
            for p in K.P_GRID:
                for i in range(N_FAULTY):
                    fh.write(json.dumps(dict(kind="faulty", d=d, p=pkey(p), i=i, secret=gen.new_secret().hex())) + "\n")
            for i in range(N_CLEAN):
                fh.write(json.dumps(dict(kind="clean", d=d, p=None, i=i, secret=gen.new_secret().hex())) + "\n")


def _origins(kind, d, p=None):
    rows = [r for r in _read("origins.jsonl") if r["kind"] == kind and r["d"] == d and (p is None or r["p"] == pkey(p))]
    if not rows:
        raise SystemExit("no origins; run `origins` first")
    return rows


def _batch(name, args, fn):
    t0, th0 = time.time(), therm()
    n = fn()
    _append("batches.jsonl", dict(phase=name, args=args, n_new=n, secs=round(time.time() - t0, 1),
                                  therm_before=th0, therm_after=therm(), utc=utc()))
    print(f"{name} {args}: {n} new records in {time.time() - t0:.0f} s", flush=True)


def fuzz(d, p):
    done = {(r["i"], r["solver"]) for r in _read("fuzz.jsonl") if r["d"] == d and r["p"] == pkey(p)}

    def run():
        n = 0
        for o_row in _origins("faulty", d, p):
            secret = bytes.fromhex(o_row["secret"])
            o = gen.make_origin(secret, d=d, faulty=True, p=p, g=K.G_DEFAULT)
            net = gen.build_level(secret, o, "O0").net
            for name, f in F.FUZZERS.items():
                if (o_row["i"], name) in done:
                    continue
                rng = gen._rng(secret, "fuzz", name)
                idx, x = f(net, o.teacher, d, o.R, rng, qmax=QMAX)
                ok = SC.verify_counterexample(x, net, o.teacher, o.R) if x is not None else False
                if idx is not None and not ok:
                    raise RuntimeError(f"fuzzer {name} reported a hit the scorer rejects (origin {o_row['i']})")
                _append("fuzz.jsonl", dict(d=d, p=pkey(p), i=o_row["i"], solver=name, first_hit=idx,
                                           verified=ok, masses=o.masses, utc=utc()))
                n += 1
        return n
    _batch("fuzz", dict(d=d, p=pkey(p)), run)


def struct(d, p):
    done = {(r["i"], r["g"], r["level"], r["decoys"]) for r in _read("struct.jsonl") if r["d"] == d and r["p"] == pkey(p)}

    def run():
        n = 0
        for o_row in _origins("faulty", d, p):
            secret = bytes.fromhex(o_row["secret"])
            for g in K.G_VALUES:
                o = gen.make_origin(secret, d=d, faulty=True, p=p, g=g)
                for lv, dec in VARIANTS:
                    if (o_row["i"], g, lv, dec) in done:
                        continue
                    L = gen.build_level(secret, o, lv, decoys=dec)
                    sn = ser.serialised_net(L.net)
                    rec = dict(d=d, p=pkey(p), i=o_row["i"], g=g, level=lv, decoys=dec, utc=utc())
                    for name, f in S.STRUCTURAL.items():
                        a = f(sn, o.teacher)
                        r = SC.score(a, L.net, o.teacher, o.R, L.needle_units)
                        rec[name] = dict(score=a["score"], localised=r["localised"],
                                         **({"faulty": a["faulty"], "success": r["success"]} if name == "S5" else {}))
                    _append("struct.jsonl", rec)
                    n += 1
        return n
    _batch("struct", dict(d=d, p=pkey(p)), run)


def clean(d):
    done = {(r["i"], r["level"], r["decoys"]) for r in _read("clean.jsonl") if r["d"] == d}

    def run():
        n = 0
        for o_row in _origins("clean", d):
            secret = bytes.fromhex(o_row["secret"])
            o = gen.make_origin(secret, d=d, faulty=False)
            for lv, dec in VARIANTS:
                if (o_row["i"], lv, dec) in done:
                    continue
                L = gen.build_level(secret, o, lv, decoys=dec)
                sn = ser.serialised_net(L.net)
                rec = dict(d=d, i=o_row["i"], split="calibration" if o_row["i"] < N_CLEAN // 2 else "false_alarm",
                           level=lv, decoys=dec, utc=utc())
                for name, f in S.STRUCTURAL.items():
                    a = f(sn, o.teacher)
                    rec[name] = dict(score=a["score"], **({"faulty": a["faulty"],
                                     "success": SC.verify_counterexample(a["counterexample"], L.net, o.teacher, o.R)}
                                     if name == "S5" else {}))
                # Sanity fuzz on clean networks (1e4 queries each): a hit would be an exactness failure.
                if lv == "O0" and dec == "mixed":
                    for name, f in F.FUZZERS.items():
                        idx, _ = f(L.net, o.teacher, d, o.R, gen._rng(secret, "fuzz", name), qmax=10**4)
                        rec[f"{name}_hit"] = idx
                _append("clean.jsonl", rec)
                n += 1
        return n
    _batch("clean", dict(d=d), run)


def status():
    fz, st, cl = _read("fuzz.jsonl"), _read("struct.jsonl"), _read("clean.jsonl")
    print(f"fuzz: {len(fz)} / {len(K.D_VALUES) * len(K.P_GRID) * N_FAULTY * 3}")
    print(f"struct: {len(st)} / {len(K.D_VALUES) * len(K.P_GRID) * N_FAULTY * 3 * len(VARIANTS)}")
    print(f"clean: {len(cl)} / {len(K.D_VALUES) * N_CLEAN * len(VARIANTS)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["origins", "fuzz", "struct", "clean", "status"])
    ap.add_argument("--d", type=int)
    ap.add_argument("--p", type=float)
    a = ap.parse_args()
    if a.phase == "origins":
        origins()
    elif a.phase == "fuzz":
        fuzz(a.d, a.p)
    elif a.phase == "struct":
        struct(a.d, a.p)
    elif a.phase == "clean":
        clean(a.d)
    else:
        status()


if __name__ == "__main__":
    main()
