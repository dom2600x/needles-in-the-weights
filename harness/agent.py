"""MLX agent loop for LLM proposers (PREREG 7; G0 decision D6/F2). Frozen at G3 by commit hash.

One run = one blind task, one model, one seed:

    .venv/bin/python -m harness.agent run --task blind/<id> --truth truth --model 9b --seed 7 --phase pilot

Turn structure. Each turn the model generates (thinking on) until it closes a ```python block after
</think>, emits <|im_end|>, or reaches the per-turn token cap. The first python block after </think>
is run in the persistent sandboxed kernel (harness/sandbox); its output goes back as a tool response.
A reply with no complete block gets a fixed reminder instead. Thinking stays in the context (this is
what the Qwen3.5 chat template does for multi-step tool turns, and it lets the KV cache be reused).

Stops. An accepted submit() ends the run. Otherwise, when the wall-clock cap is reached
(mid-generation: the cut-off turn is discarded) or the turn cap is used up, one forced final turn is
given: FORCED message, thinking off, at most FORCED_TOKENS tokens, FORCED_TOOL_S of tool time, and its
block is run only if it calls submit. No accepted answer after that = timeout.
After WARN_AT_S of wall-clock, WARNING is appended to the next tool output (once).

Wall-clock counts everything from the start of the first prefill: generation, prefill and tool time.
Truth is never opened here: the oracle functions come from harness.score.oracle_fns.
"""
import argparse, json, os, re, secrets, subprocess, sys, time
from datetime import datetime, timezone

import numpy as np

from harness import answer as ANS
from harness import prompt as P
from harness.oracles import MeteredOracle
from harness.sandbox.host import SandboxKernel
from testbed import constants as K
from testbed import serialise as ser

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

CAPS = dict(turns=12, tokens_per_turn=6144, cell_s=60.0, tool_s=600.0, wall_s=1800.0, out_cap=4000)
WARN_AT_S = 1500.0
FORCED_TOKENS = 512
FORCED_TOOL_S = 30.0

# Pinned models (G0; local HF cache only). Sampler: Qwen3.5 model card, thinking mode,
# "precise coding" preset (temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0).
MODELS = {
    "9b": ("mlx-community/Qwen3.5-9B-4bit", "8b2b98c00a6b4d291155e4890773ca8f769aee53"),
    "2b": ("mlx-community/Qwen3.5-2B-4bit", "674aaa7240b91e8012fcad5d791b7dfe5ba90207"),
}
SAMPLER = dict(temp=0.6, top_p=0.95, top_k=20, min_p=0.0)
PRESENCE_PENALTY = 0.0
REPETITION_PENALTY = 1.0

BLOCK_RE = re.compile(r"```python[ \t]*\n(.*?)```", re.S)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def therm():
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        return " | ".join(l.strip() for l in out.splitlines() if l.strip())
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def answer_part(text):
    """The reply after the reasoning (thinking-mode replies start inside <think>)."""
    return text.split("</think>", 1)[1] if "</think>" in text else None


def first_block(text, thinking=True):
    part = answer_part(text) if thinking else text
    if part is None:
        return None
    m = BLOCK_RE.search(part)
    return m.group(1) if m else None


# ---------------- generators ----------------
class MLXGenerator:
    """Qwen3.5 via mlx-lm with one persistent prompt cache per run. The cache cannot be trimmed
    (linear-attention layers), so a discarded turn is undone by restoring a snapshot taken before it."""

    def __init__(self, key):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        self.mx = mx
        self.repo, self.revision = MODELS[key]
        self.model, self.tok = load(self.repo, revision=self.revision)
        self.sampler = make_sampler(**SAMPLER)
        self.procs = make_logits_processors(presence_penalty=PRESENCE_PENALTY or None,
                                            repetition_penalty=REPETITION_PENALTY if REPETITION_PENALTY != 1.0 else None)
        self.eos = self.tok.encode("<|im_end|>", add_special_tokens=False)
        assert len(self.eos) == 1
        self.eos = self.eos[0]

    def reset(self, seed):
        from mlx_lm.models import cache as C
        self.mx.random.seed(seed)
        self.cache = C.make_prompt_cache(self.model)
        self.ctx = []          # every token logically in the conversation
        self.n_cached = 0      # how many of them are in the cache

    def _kv_offset(self):
        from mlx_lm.models.cache import KVCache
        for c in self.cache:
            if isinstance(c, KVCache):
                return c.offset
        raise RuntimeError("no KVCache layer to read the offset from")

    def _snapshot(self):
        from mlx_lm.models.cache import KVCache
        return [("kv", c.offset) if isinstance(c, KVCache) else ("arr", list(c.cache)) for c in self.cache]

    def _restore(self, snap):
        for c, (kind, s) in zip(self.cache, snap):
            if kind == "kv":
                c.offset = s
            else:
                c.cache = list(s)

    def render_first(self, system, user, thinking=True):
        return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            add_generation_prompt=True, enable_thinking=thinking, tokenize=False)

    def turn(self, prefix, max_tokens, deadline, clock, thinking=True):
        from mlx_lm.generate import generate_step
        mx = self.mx
        new = self.tok.encode(prefix, add_special_tokens=False)
        feed = self.ctx[self.n_cached:] + new
        snap, n_cached0 = self._snapshot(), self.n_cached
        detok = self.tok.detokenizer
        detok.reset()
        out, finish, t0, t_first = [], "length", clock(), None
        in_answer = not thinking
        for tok, _ in generate_step(mx.array(feed), self.model, max_tokens=max_tokens, sampler=self.sampler,
                                    logits_processors=self.procs or None, prompt_cache=self.cache):
            if t_first is None:
                t_first = clock()
            if tok == self.eos:
                finish = "stop"
                break
            out.append(tok)
            detok.add_token(tok)
            seg = detok.last_segment
            if deadline is not None and clock() >= deadline:
                finish = "deadline"
                break
            if not in_answer and "</think>" in detok.text:
                in_answer = True
            if in_answer and "`" in seg and BLOCK_RE.search(answer_part(detok.text) if thinking else detok.text):
                finish = "block"
                break
        detok.finalize()
        t1 = clock()
        if finish == "deadline":
            self._restore(snap)
            self.n_cached = n_cached0
        else:
            self.ctx += new + out + [self.eos]
            try:
                self.n_cached = self._kv_offset()
            except RuntimeError:
                self.n_cached = len(self.ctx) - 1
        n_prefill = len(feed)
        prefill_s = (t_first - t0) if t_first else (t1 - t0)
        gen_s = max(t1 - (t_first or t1), 1e-9)
        return dict(text=detok.text, n_tokens=len(out), finish=finish, prefill_tokens=n_prefill,
                    prefill_s=round(prefill_s, 3), gen_s=round(gen_s, 3),
                    gen_tps=round(len(out) / gen_s, 2) if out else None,
                    peak_mem_gb=round(mx.get_peak_memory() / 1e9, 3), ctx_tokens=len(self.ctx))


# ---------------- kernel setup ----------------
def make_kernel(scratch, blind, oracles, tool_budget):
    k = SandboxKernel(scratch, oracles=oracles, cell_timeout=CAPS["cell_s"], tool_budget=tool_budget)
    k.start()
    d, H = int(blind["d"]), int(blind["H"])
    k.validate_submit = lambda a: ANS.validate(a, d, H)
    k.set_value("d", d)
    k.set_value("R", float(blind["R_exact"]))
    if blind["mode"] == "W":
        w = blind["weights"]
        for name in ("W1", "b1", "v"):
            k.set_array(name, w[name])
        k.set_value("c", float(w["c"]))
    return k


def format_output(r):
    parts = []
    if r["stdout"]:
        parts.append(r["stdout"].rstrip("\n"))
    if r["stderr"]:
        parts.append(r["stderr"].rstrip("\n"))
    if r["status"] == "timeout":
        parts.append(f"[the block hit the {int(CAPS['cell_s'])} s limit]")
    elif r["status"] == "killed":
        parts.append("[the session was stopped and restarted; all variables were lost, call budgets are unchanged]")
    elif r["status"] == "budget_exhausted":
        parts.append("[the total code-time limit is used up; code blocks no longer run]")
    return "\n".join(parts) if parts else "[no output]"


def tool_segment(text):
    return f"\n<|im_start|>user\n<tool_response>\n{text}\n</tool_response><|im_end|>"


ASSISTANT_THINK = "\n<|im_start|>assistant\n<think>\n"
ASSISTANT_NOTHINK = "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------------- the run ----------------
def run_task(gen, blind, oracle_fns, scratch, seed, log, clock=time.monotonic, kernel_factory=make_kernel):
    """gen: MLXGenerator-like. blind: read_blind dict plus R_exact. oracle_fns: (spec_fn, net_fn).
    log(event_dict) records each event. Returns the run summary."""
    d, R = int(blind["d"]), float(blind["R_exact"])
    spec_fn, net_fn = oracle_fns
    oracles = {"spec": MeteredOracle(spec_fn, d, R, K.SPEC_ROWS_CAP, "spec")}
    if blind["mode"] == "B":
        oracles["net"] = MeteredOracle(net_fn, d, R, K.NET_ROWS_CAP, "net")
    system, user = P.build(blind, CAPS)
    gen.reset(seed)
    kernel = kernel_factory(scratch, blind, oracles, CAPS["tool_s"])
    tool_used_before_restarts = 0.0
    log(dict(ev="start", utc=utc(), seed=seed, therm=therm(), kernel=kernel.info))
    t0 = clock()
    deadline = t0 + CAPS["wall_s"]
    prefix = gen.render_first(system, user, thinking=True)
    warned, submission, forced, n_turns, stop_reason = False, None, False, 0, None
    for turn in range(1, CAPS["turns"] + 1):
        if clock() >= deadline:
            stop_reason = "wall"
            break
        g = gen.turn(prefix, CAPS["tokens_per_turn"], deadline, clock, thinking=True)
        log(dict(ev="gen", turn=turn, t=round(clock() - t0, 2), **g))
        if g["finish"] == "deadline":
            stop_reason = "wall"   # the cut-off turn is discarded; its pending tool output is re-sent below
            break
        n_turns = turn
        code = first_block(g["text"])
        if code is None:
            out, status = P.NO_BLOCK, "no_block"
        else:
            r = kernel.exec(code)
            status = r["status"]
            if status == "killed":
                tool_used_before_restarts += kernel.tool_used
                kernel.close()
                kernel = kernel_factory(scratch, blind, oracles, max(0.0, CAPS["tool_s"] - tool_used_before_restarts))
            out = format_output(r)
            log(dict(ev="cell", turn=turn, t=round(clock() - t0, 2), code=code, status=status, wall_s=r["wall_s"],
                     stdout=r["stdout"], stderr=r["stderr"], oracles={k: o.log() for k, o in oracles.items()}))
            if status == "submitted":
                submission = kernel.submission
                stop_reason = "submitted"
                break
        if not warned and clock() - t0 >= WARN_AT_S:
            out, warned = out + "\n" + P.WARNING, True
        prefix = tool_segment(out) + ASSISTANT_THINK
    else:
        stop_reason = "turns"
    if submission is None:
        # Forced final turn. The pending tool output (if any) is sent before the forced message.
        forced = True
        pending = prefix[: -len(ASSISTANT_THINK)] if prefix.endswith(ASSISTANT_THINK) else ""
        fprefix = pending + f"\n<|im_start|>user\n{P.FORCED}<|im_end|>" + ASSISTANT_NOTHINK
        g = gen.turn(fprefix, FORCED_TOKENS, None, clock, thinking=False)
        log(dict(ev="gen", turn="forced", t=round(clock() - t0, 2), **g))
        code = first_block(g["text"], thinking=False)
        if code is not None and "submit(" in code:
            tool_used = tool_used_before_restarts + kernel.tool_used
            kernel.tool_budget = kernel.tool_used + FORCED_TOOL_S
            kernel.cell_timeout = FORCED_TOOL_S
            r = kernel.exec(code)
            log(dict(ev="cell", turn="forced", t=round(clock() - t0, 2), code=code, status=r["status"],
                     wall_s=r["wall_s"], stdout=r["stdout"], stderr=r["stderr"],
                     oracles={k: o.log() for k, o in oracles.items()}, tool_used_before=tool_used))
            if r["status"] == "submitted":
                submission = kernel.submission
        else:
            log(dict(ev="forced_not_run", reason="no block" if code is None else "block does not call submit"))
    tool_total = tool_used_before_restarts + kernel.tool_used
    kernel.close()
    summary = dict(ev="end", utc=utc(), stop_reason=stop_reason, turns=n_turns, forced_turn=forced,
                   submitted_in_forced_turn=bool(forced and submission is not None), submission=submission,
                   wall_s=round(clock() - t0, 2), tool_s=round(tool_total, 2),
                   oracles={k: o.log() for k, o in oracles.items()}, therm=therm())
    log(summary)
    return summary


# ---------------- CLI ----------------
def run_cli(a):
    from harness import score as SC
    task_dir = os.path.realpath(a.task)
    blind = ser.read_blind(task_dir)
    blind["R_exact"] = K.radius(int(blind["d"]))
    tid = blind["task_id"]
    spec_fn, net_fn, info = SC.oracle_fns(os.path.realpath(a.truth), tid)
    assert info["mode"] == blind["mode"] and info["d"] == int(blind["d"])
    run_id = secrets.token_hex(8)
    out_dir = os.path.join(ROOT, "runs", a.phase)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_id}.jsonl")
    fh = open(path, "x")

    def log(ev):
        fh.write(json.dumps(ev, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"); fh.flush()

    gen = MLXGenerator(a.model)
    log(dict(ev="meta", run_id=run_id, task_id=tid, mode=blind["mode"], model=a.model, repo=gen.repo,
             revision=gen.revision, sampler=SAMPLER, presence_penalty=PRESENCE_PENALTY,
             repetition_penalty=REPETITION_PENALTY, caps=CAPS, warn_at_s=WARN_AT_S, forced_tokens=FORCED_TOKENS,
             forced_tool_s=FORCED_TOOL_S, phase=a.phase, harness_commit=git_head()))
    scratch = os.path.join(out_dir, "scratch", run_id)
    s = run_task(gen, blind, (spec_fn, net_fn), scratch, a.seed, log)
    fh.close()
    with open(os.path.join(out_dir, "runs.jsonl"), "a") as f:
        f.write(json.dumps(dict(run_id=run_id, task_id=tid, model=a.model, seed=a.seed, mode=blind["mode"],
                                **{k: s[k] for k in ("stop_reason", "turns", "forced_turn", "submitted_in_forced_turn",
                                                     "submission", "wall_s", "tool_s", "oracles")}), default=str) + "\n")
    print(f"run {run_id}: {s['stop_reason']}, {s['turns']} turns, {s['wall_s']:.0f} s, "
          f"submitted={s['submission'] is not None}", flush=True)


def git_head():
    try:
        h = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "harness", "testbed"],
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True)
    r.add_argument("--truth", required=True)
    r.add_argument("--model", choices=sorted(MODELS), required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--phase", required=True, choices=["pilot", "confirm", "claude"])
    a = ap.parse_args()
    run_cli(a)


if __name__ == "__main__":
    main()
