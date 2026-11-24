"""G0: numpy forward-pass throughput INSIDE the proposer sandbox (PREREG section 7, brute-force accounting).

Network: one hidden ReLU layer, d = 32, H = 96, random weights (throughput does not depend on values).
Measures evaluations per second for float64 and float32, forward only and sample-in-ball + forward,
over several batch sizes, with BLAS limited to 1 thread and unrestricted (8).
Usage: .venv/bin/python harness/g0_throughput.py   -> runs/g0/throughput.jsonl
"""
import json, os, secrets, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "sandbox"))
from host import REPO, SandboxKernel  # noqa: E402

OUT = os.path.join(REPO, "runs", "g0", "throughput.jsonl")
CELL = r'''
import time
d, H, R = 32, 96, 1.0
rng = np.random.default_rng(0)
W1 = rng.normal(size=(d, H)).astype(DT); b1 = rng.normal(size=H).astype(DT)
v = rng.normal(size=H).astype(DT); c = DT(0.1)
def fwd(X):
    return np.maximum(X @ W1 + b1, 0) @ v + c
def ball(n):
    g = rng.standard_normal((n, d), dtype=DT)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * (R * rng.random(n, dtype=DT) ** (1.0 / d))[:, None]
X = ball(BATCH)
fwd(X)  # warm-up
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < SECS:
    if MODE == "fwd":
        y = fwd(X)
    else:
        y = fwd(ball(BATCH))
    n += BATCH
dt = time.perf_counter() - t0
print(n / dt)
'''


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for threads in (1, 8):
        k = SandboxKernel(os.path.join(REPO, "runs", "g0", "sandbox_scratch", secrets.token_hex(8)),
                          cell_timeout=60, tool_budget=3000, blas_threads=threads)
        info = k.start()
        for dt in ("float64", "float32"):
            for mode in ("fwd", "sample+fwd"):
                for batch in (4096, 65536, 1048576):
                    code = f"DT = np.{dt}\nBATCH = {batch}\nMODE = {mode!r}\nSECS = 8.0\n" + CELL
                    r = k.exec(code)
                    if r["status"] != "ok":
                        print("FAILED", threads, dt, mode, batch, r); continue
                    eps = float(r["stdout"].strip())
                    rec = dict(utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), blas_threads=threads,
                               dtype=dt, mode=mode, batch=batch, evals_per_s=round(eps), d=32, H=96,
                               numpy=info["numpy"], python=info["python"], audit_hook=info["audit_hook"])
                    rows.append(rec)
                    with open(OUT, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"threads={threads} {dt:7s} {mode:10s} batch={batch:>8d}  {eps:,.0f} evals/s", flush=True)
        k.close()
    best = max(rows, key=lambda r: r["evals_per_s"])
    best_s = max((r for r in rows if r["mode"] == "sample+fwd"), key=lambda r: r["evals_per_s"])
    for label, b in (("forward only (upper bound)", best), ("sample + forward", best_s)):
        n = b["evals_per_s"] * 600
        print(f"\nBest {label}: {b['evals_per_s']:,} evals/s ({b['dtype']}, batch {b['batch']}, {b['blas_threads']} threads)")
        print(f"  in the 10-min tool cap: {n:.3g} evals; expected hits at p=1e-9: {n * 1e-9:.3g}; at p=1e-10: {n * 1e-10:.3g}")


if __name__ == "__main__":
    main()
