"""PREREG 3.4 exactness checks 1-3, and the G1 verification run over the design grid.

    .venv/bin/python -m testbed.verify            # full grid -> runs/g1/verify.jsonl + summary
    .venv/bin/python -m testbed.verify --quick    # small grid

Check 1: every level computes teacher + needle: max |f_level - f_ref| < 1e-9 on 1e6 uniform points.
Check 2: the serialised (8 significant digit) network deviates from the original by less than
         1e-3 x FAULT_MEDIAN on 1e6 uniform points plus 1e5 exact points in the needle region A.
Check 3: closed-form p, p_s, p_V, p_Vs agree within 5% with Monte Carlo (value >= 1e-4) or
         importance sampling (value < 1e-4). Also: the network's empirical violation rate in A,
         by exact conditional sampling, matches p_V / p within 5%; and, where p_V >= 1e-4, the
         empirical violation rate on uniform points matches p_V.
"""
import argparse, json, math, os, time
from datetime import datetime, timezone

import numpy as np

from . import constants as K
from . import generator as gen
from . import geometry as geo
from . import serialise as ser

EXACT_TOL = 1e-9
SERIAL_TOL = 1e-3 * K.FAULT_MEDIAN
MASS_TOL = 0.05
CHUNK = 500_000


def _np_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def _max_abs_diff(f, g, xs_iter):
    m = 0.0
    for x in xs_iter:
        m = max(m, float(np.max(np.abs(f(x) - g(x)))))
    return m


def _uniform_chunks(rng, n, d, R):
    done = 0
    while done < n:
        k = min(CHUNK, n - done)
        yield geo.sample_ball(rng, k, d, R)
        done += k


def check_exact(secret, origin, lv, rng, n=10**6):
    ref = gen.reference(origin)
    m = _max_abs_diff(lv.net, ref, _uniform_chunks(rng, n, origin.d, origin.R))
    return dict(check="exact", max_abs=m, tol=EXACT_TOL, ok=m < EXACT_TOL)


def check_serialised(origin, lv, rng, n=10**6, n_cap=10**5):
    obj = json.loads(ser.blind_json("0" * 32, "W", net=lv.net))
    w = obj["weights"]
    sn = gen.Net(np.array(w["W1"]), np.array(w["b1"]), np.array(w["v"]), float(w["c"]))
    m_u = _max_abs_diff(lv.net, sn, _uniform_chunks(rng, n, origin.d, origin.R))
    m_a = 0.0
    if origin.faulty:
        xa = geo.sample_cap(rng, n_cap, origin.d, origin.R, origin.u, origin.c_cap)
        m_a = float(np.max(np.abs(lv.net(xa) - sn(xa))))
    m = max(m_u, m_a)
    return dict(check="serialised", max_abs_uniform=m_u, max_abs_cap=m_a, tol=SERIAL_TOL, ok=m < SERIAL_TOL)


def _mc_mass(rng, u, c, d, sphere, target_count=4000, n_max=4 * 10**7):
    p_guess = float((geo.sphere_mass if sphere else geo.ball_mass)(c, d))
    n = int(min(n_max, max(10**6, target_count / p_guess)))
    hits, done = 0, 0
    while done < n:
        k = min(CHUNK, n - done)
        x = geo.sample_sphere(rng, k, d, 1.0) if sphere else geo.sample_ball(rng, k, d, 1.0)
        hits += int(np.count_nonzero(x @ u > c)); done += k
    est = hits / n
    return est, (math.sqrt(est * (1 - est) / n) / est if est > 0 else float("inf")), n


def check_masses(origin, rng):
    """Closed form vs MC (value >= 1e-4) or IS (value < 1e-4), for p, p_s, p_V, p_Vs."""
    out = []
    d = origin.d
    for name, c, sphere in (("p", origin.c_cap, False), ("p_s", origin.c_cap, True),
                            ("p_V", origin.c_V, False), ("p_Vs", origin.c_V, True)):
        closed = origin.masses[name]
        if closed >= 1e-4:
            est, rse, n = _mc_mass(rng, origin.u, c, d, sphere)
            method = f"MC n={n}"
        else:
            est, rse = geo.cap_mass_importance(rng, c, d, sphere=sphere)
            method = "IS n=200000"
        rel = est / closed - 1
        out.append(dict(check="mass", name=name, closed=closed, estimate=est, rel_err=rel, rse=rse,
                        method=method, ok=abs(rel) <= MASS_TOL))
    return out


def check_violation_rates(origin, lv, rng, n_cap=10**5):
    """Network-level: violation fraction in A (exact conditional sampling) vs p_V / p; and, when
    p_V >= 1e-4, the uniform violation rate vs p_V. Ties the generator to the closed forms."""
    res = []
    xa = geo.sample_cap(rng, n_cap, origin.d, origin.R, origin.u, origin.c_cap)
    fa = np.abs(lv.net(xa) - origin.teacher(xa))
    active = np.all(fa > 0)  # the needle is active everywhere in A
    frac = float(np.mean(fa > K.DELTA))
    want = origin.masses["p_V"] / origin.masses["p"]
    res.append(dict(check="viol_in_A", frac=frac, expected=want, rel_err=frac / want - 1,
                    needle_active_in_A=bool(active), median_fault=float(np.median(fa)),
                    ok=abs(frac / want - 1) <= MASS_TOL and bool(active)))
    pV = origin.masses["p_V"]
    if pV >= 1e-4:
        n = int(min(4 * 10**7, max(10**6, 4000 / pV)))
        hits = 0
        for x in _uniform_chunks(rng, n, origin.d, origin.R):
            hits += int(np.count_nonzero(np.abs(lv.net(x) - origin.teacher(x)) > K.DELTA))
        est = hits / n
        res.append(dict(check="viol_uniform", estimate=est, expected=pV, rel_err=est / pV - 1, n=n,
                        ok=abs(est / pV - 1) <= MASS_TOL))
    return res


def run_grid(quick=False, out=None, seed=None):
    rng = np.random.default_rng(seed)
    ds = (32,) if quick else K.D_VALUES
    ps = (1e-1, 1e-5, 1e-12) if quick else K.P_GRID
    gs = (3,) if quick else K.G_VALUES
    rows, t0 = [], time.time()

    def emit(rec):
        rec["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(rec)
        if out:
            with open(out, "a") as fh:
                fh.write(json.dumps(rec, default=_np_default) + "\n")
        if not rec["ok"]:
            print("FAIL", json.dumps(rec, default=_np_default), flush=True)

    configs = [(d, p, g, True) for d in ds for p in ps for g in gs] + [(d, None, None, False) for d in ds for _ in range(3)]
    for d, p, g, faulty in configs:
        secret = gen.new_secret()
        o = gen.make_origin(secret, d=d, faulty=faulty, p=p or 1e-12, g=g or K.G_DEFAULT)
        meta = dict(d=d, p=p, g=g, faulty=faulty, fp=o.secret_fingerprint)
        variants = [(lv, "mixed") for lv in K.LEVELS] + [("O2", "cancel_only"), ("O3", "cancel_only")]
        for level, decoys in variants:
            L = gen.build_level(secret, o, level, decoys=decoys)
            for r in (check_exact(secret, o, L, rng, n=10**5 if quick else 10**6),
                      check_serialised(o, L, rng, n=10**5 if quick else 10**6)):
                emit({**meta, "level": level, "decoys": decoys, **r})
            if faulty and level == "O0":
                for r in check_violation_rates(o, L, rng):
                    emit({**meta, "level": level, "decoys": decoys, **r})
        if faulty:
            for r in check_masses(o, rng):
                emit({**meta, **r})
    summary = {}
    for r in rows:
        key = r["check"] + (":" + r["name"] if "name" in r else "")
        s = summary.setdefault(key, dict(n=0, fail=0, worst=0.0))
        s["n"] += 1; s["fail"] += 0 if r["ok"] else 1
        worst = r.get("max_abs", max(r.get("max_abs_uniform", 0), r.get("max_abs_cap", 0))) \
            if r["check"] in ("exact", "serialised") else abs(r.get("rel_err", 0))
        s["worst"] = max(s["worst"], worst)
    return rows, summary, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "runs", "g1")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "verify_quick.jsonl" if a.quick else "verify.jsonl")
    if os.path.exists(out):
        os.replace(out, out + f".prev-{int(time.time())}")  # keep earlier runs; never delete
    rows, summary, secs = run_grid(quick=a.quick, out=out)
    print(json.dumps(summary, indent=1, default=_np_default))
    print(f"{len(rows)} checks, {sum(not r['ok'] for r in rows)} failed, {secs:.0f} s -> {out}")
    raise SystemExit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
