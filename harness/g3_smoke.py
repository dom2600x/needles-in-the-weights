"""G3 harness smoke test on the real models with a TOY prompt (not the study prompt, not a study
network; nothing here is study data). Checks the MLX path of harness/agent.py end to end.

    .venv/bin/python -m harness.g3_smoke 2b|9b

1. Cache restore: greedy continuation after a restored snapshot equals greedy continuation from a
   fresh prefill of the same tokens.
2. A toy run through run_task: the model sums an array in the sandbox and submits; checks code-block
   stop, kernel exec, tool responses, cache reuse (prefill tokens per turn << context), submit.
3. Deadline: with a 45 s wall-clock cap and a prompt that invites long reasoning, the cut-off turn
   is discarded and the forced turn (thinking off) runs.
Writes runs/g3/smoke_<model>.jsonl.
"""
import json, os, sys, time

import numpy as np

from harness import agent as A
from harness import prompt as P

ROOT = A.ROOT
OUT = os.path.join(ROOT, "runs", "g3")

TOY_SYSTEM = P.SYSTEM
TOY_USER = ("An array `W1` of shape (4, 3) is defined in the session. Compute the sum of all its entries "
            "with numpy and print it. Then, in a separate reply after you have seen the printed value, "
            "call submit({'faulty': False, 'counterexample': None, 'unit': None, 'edit': None}). "
            "Use exactly one python code block per reply.")
LONG_USER = ("Before writing any code, reason very carefully and at length, considering at least twenty "
             "different approaches, about how one might prove that there are infinitely many primes. "
             "Only then write one python block that prints 'done'.")


def check_restore(gen, log):
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler
    gen.sampler = make_sampler(temp=0.0)
    gen.reset(0)
    first = gen.render_first(TOY_SYSTEM, "Count from 1 to 5, then stop.", thinking=False)
    g1 = gen.turn(first, 24, None, time.monotonic, thinking=False)
    snap, n0, ctx0 = gen._snapshot(), gen.n_cached, list(gen.ctx)
    nxt = "\n<|im_start|>user\nNow count from 6 to 10.<|im_end|>" + A.ASSISTANT_NOTHINK
    # a turn that is then discarded (deadline already passed after the first token)
    g_cut = gen.turn("\n<|im_start|>user\nWrite a long poem.<|im_end|>" + A.ASSISTANT_NOTHINK, 50,
                     time.monotonic(), time.monotonic, thinking=False)
    assert g_cut["finish"] == "deadline" and gen.ctx == ctx0 and gen.n_cached == n0
    a = gen.turn(nxt, 24, None, time.monotonic, thinking=False)["text"]
    # fresh prefill of exactly the same logical context
    gen.reset(0)
    gen.ctx, gen.n_cached = list(ctx0), 0
    b = gen.turn(nxt, 24, None, time.monotonic, thinking=False)["text"]
    log(dict(check="restore", after_restore=a, fresh=b, equal=a == b, first=g1["text"]))
    gen.sampler = make_sampler(**A.SAMPLER)
    return a == b


def toy_run(gen, user, log, wall_s=None):
    saved = dict(A.CAPS)
    orig_build = P.build
    try:
        if wall_s:
            A.CAPS["wall_s"] = wall_s
        P.build = lambda blind, caps: (TOY_SYSTEM, user)
        blind = dict(mode="W", d=4, H=3, R_exact=1.0,
                     weights=dict(W1=np.arange(12.0).reshape(4, 3), b1=np.zeros(3), v=np.ones(3), c=0.0))
        scratch = os.path.join(OUT, "scratch", f"smoke_{int(time.time())}")
        events = []
        s = A.run_task(gen, blind, (lambda x: np.zeros(len(x)), None), scratch, 1, lambda e: (events.append(e), log(e)))
        return s, events
    finally:
        A.CAPS.clear(); A.CAPS.update(saved); P.build = orig_build


def main(key):
    os.makedirs(OUT, exist_ok=True)
    fh = open(os.path.join(OUT, f"smoke_{key}.jsonl"), "a")

    def log(e):
        fh.write(json.dumps(e, default=str) + "\n"); fh.flush()

    t0 = time.time()
    gen = A.MLXGenerator(key)
    log(dict(ev="loaded", model=key, secs=round(time.time() - t0, 1), utc=A.utc()))
    ok_restore = check_restore(gen, log)
    s, ev = toy_run(gen, TOY_USER, log)
    gens = [e for e in ev if e["ev"] == "gen"]
    cells = [e for e in ev if e["ev"] == "cell"]
    printed = any("66" in (c.get("stdout") or "") for c in cells)
    ok_run = s["submission"] is not None and printed
    reuse = [(g["prefill_tokens"], g["ctx_tokens"]) for g in gens]
    s2, ev2 = toy_run(gen, LONG_USER, log, wall_s=45)
    g2 = [e for e in ev2 if e["ev"] == "gen"]
    ok_deadline = s2["stop_reason"] == "wall" and s2["forced_turn"] and g2[0]["finish"] == "deadline"
    res = dict(ev="summary", model=key, restore_equal=ok_restore, toy_submitted=s["submission"] is not None,
               printed_66=printed, toy_turns=s["turns"], toy_wall_s=s["wall_s"], prefill_vs_ctx=reuse,
               gen_tps=[g.get("gen_tps") for g in gens], deadline_ok=ok_deadline,
               deadline_forced_text=g2[-1]["text"][:300], deadline_wall_s=s2["wall_s"], utc=A.utc())
    log(res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
