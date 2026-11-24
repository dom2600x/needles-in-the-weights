"""G0 smoke test and speed benchmark for one MLX model.

Usage: .venv/bin/python harness/g0_bench.py <hf_repo> <revision> [--lengths 2000,10000,20000]

Writes one JSONL record per measurement to runs/g0/bench.jsonl and prints a summary.
Loads only the pinned revision from the local HF cache; it downloads nothing.
"""
import argparse, json, os, platform, subprocess, time
from datetime import datetime, timezone

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# Qwen3.5 model-card thinking-mode sampler (temperature 0.6, top_p 0.95, top_k 20).
SAMPLER = dict(temp=0.6, top_p=0.95, top_k=20)
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "g0", "bench.jsonl")


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def therm():
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        return " | ".join(l.strip() for l in out.splitlines() if l.strip())
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def filler_text(n_rows):
    # Weight-dump-like filler, close to what auditors will actually read.
    rows = []
    for i in range(n_rows):
        vals = ", ".join(f"{((i * 7919 + j * 104729) % 20001 - 10000) / 10000:+.4f}" for j in range(8))
        rows.append(f"W1[{i % 32}, {8 * (i // 32) % 96}:{8 * (i // 32) % 96 + 8}] = [{vals}]")
    return "\n".join(rows)


def build_prompt(tok, n_tokens):
    question = "\n\nIn one sentence: what kind of data is listed above?"
    lo, hi = 1, 4 * n_tokens
    while lo < hi:  # binary search on filler rows to hit the target token count
        mid = (lo + hi + 1) // 2
        msgs = [{"role": "user", "content": filler_text(mid) + question}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)
        if len(ids) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    msgs = [{"role": "user", "content": filler_text(lo) + question}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)


def run(model, tok, prompt_ids, max_tokens, seed):
    mx.random.seed(seed)
    mx.reset_peak_memory()
    sampler = make_sampler(**SAMPLER)
    text, last, t0 = "", None, time.perf_counter()
    for r in stream_generate(model, tok, prompt_ids, max_tokens=max_tokens, sampler=sampler):
        text += r.text
        last = r
    wall = time.perf_counter() - t0
    return text, dict(prompt_tokens=last.prompt_tokens, prompt_tps=round(last.prompt_tps, 2),
                      generation_tokens=last.generation_tokens, generation_tps=round(last.generation_tps, 2),
                      peak_memory_gb=round(last.peak_memory, 3), finish_reason=last.finish_reason,
                      wall_s=round(wall, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("revision")
    ap.add_argument("--lengths", default="2000,10000,20000")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"

    t0 = time.perf_counter()
    model, tok = load(a.repo, revision=a.revision)
    load_s = time.perf_counter() - t0
    base = dict(repo=a.repo, revision=a.revision, mlx=mx.__version__, sampler=SAMPLER, seed=a.seed,
                machine=platform.machine(), load_s=round(load_s, 1))

    def log(rec):
        rec = {**base, **rec, "utc": utc()}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k not in ("sampler", "output")}), flush=True)

    # Smoke test: a checkable question with thinking on.
    msgs = [{"role": "user", "content": "What is 17 * 23? Reply with the number only."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)
    text, m = run(model, tok, ids, 2048, a.seed)
    answer = text.split("</think>")[-1].strip()
    log(dict(kind="smoke", therm=therm(), answer=answer[-200:], correct="391" in answer,
             saw_think_close="</think>" in text, output=text, **m))

    for n in [int(x) for x in a.lengths.split(",")]:
        ids = build_prompt(tok, n)
        before = therm()
        _, m = run(model, tok, ids, a.gen_tokens, a.seed)
        log(dict(kind="speed", target_prompt_tokens=n, therm_before=before, therm_after=therm(), **m))


if __name__ == "__main__":
    main()
