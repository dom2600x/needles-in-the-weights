"""G4 pilot analysis (exploratory; pilot runs are excluded from confirmatory analysis).

    .venv/bin/python -m analysis.pilot      -> report/g4_pilot.md, runs/pilot/scores.jsonl

Run only after every pilot run has ended (blindness rule 3). Truth is read only via harness.score.
Per run: outcome category, localisation, repair, turns, wall-clock, tool time, tokens, forced-turn
use, code-cell flags (for hand reading), provisional strategy classes, refusal candidates.
Stop rule (PREREG 4): if every local model is below 10% success here, the local arm stops.
"""
import json, math, os
from collections import defaultdict

from analysis import codebook as CB
from harness import flags as FL
from harness import score as SC

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PH = os.path.join(ROOT, "runs", "pilot")


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main():
    runs = [json.loads(l) for l in open(os.path.join(PH, "runs.jsonl"))]
    plan = [json.loads(l) for l in open(os.path.join(PH, "plan.jsonl"))]
    assert len(runs) == len(plan), "pilot not finished"
    truth = os.path.join(ROOT, "truth", "pilot")
    rows, flagged = [], []
    for r in runs:
        ev = [json.loads(l) for l in open(os.path.join(PH, f"{r['run_id']}.jsonl"))]
        gens = [e for e in ev if e["ev"] == "gen"]
        cells = [e for e in ev if e["ev"] == "cell"]
        s = SC.score_run(dict(r, task_id=r["task_id"]), truth)
        texts = [g["text"] for g in gens]
        if s["category"] == "timeout" and CB.refusal_candidate(texts):
            s["refusal_candidate"] = True
        cls = defaultdict(int)
        for c in cells:
            for k in CB.classify_cell(c["code"], c["wall_s"]):
                cls[k] += 1
            f = FL.flag_cell(c["code"], c.get("stderr") or "")
            if f:
                flagged.append(dict(run_id=r["run_id"], turn=c["turn"], flags=f))
        sub_cell = next((c for c in cells if c["status"] == "submitted"), None)
        s.update(model=r["model"], turns=r["turns"], wall_s=r["wall_s"], tool_s=r["tool_s"], stop_reason=r["stop_reason"],
                 gen_tokens=sum(g["n_tokens"] for g in gens), no_block_turns=sum(1 for g in gens if g["turn"] != "forced")
                 - sum(1 for c in cells if c["turn"] != "forced"), length_turns=sum(1 for g in gens if g["finish"] == "length"),
                 classes=dict(cls), submit_cell_classes=CB.classify_cell(sub_cell["code"], sub_cell["wall_s"]) if sub_cell else None,
                 spec_rows=r["oracles"]["spec"]["used"], net_rows=r["oracles"].get("net", {}).get("used"),
                 gen_tps=[g.get("gen_tps") for g in gens])
        rows.append(s)
    with open(os.path.join(PH, "scores.jsonl"), "w") as fh:
        for s in rows:
            fh.write(json.dumps(s, default=str) + "\n")
    out = ["# G4 pilot (exploratory; excluded from confirmatory analysis)\n",
           "p = 1e-2, d = 32, g = 3, O0; 3 faulty origins x modes W and B; each task once per model.\n",
           "| model | mode | n | success | wrong | timeout | localised | repaired | forced-turn answers | mean turns | mean wall s |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for s in rows:
        by[(s["model"], s["mode"])].append(s)
    for (m, mode), L in sorted(by.items()):
        k = sum(s["category"] == "success" for s in L)
        out.append(f"| {m} | {mode} | {len(L)} | {k} | {sum(s['category'] == 'wrong' for s in L)} | "
                   f"{sum(s['category'] == 'timeout' for s in L)} | {sum(bool(s.get('localised')) for s in L)} | "
                   f"{sum(bool(s.get('repair', {}).get('repaired')) for s in L)} | {sum(s['forced'] for s in L)} | "
                   f"{sum(s['turns'] for s in L) / len(L):.1f} | {sum(s['wall_s'] for s in L) / len(L):.0f} |")
    out.append("\n## Stop rule (PREREG 4)\n")
    stop = True
    for m in sorted({s["model"] for s in rows}):
        L = [s for s in rows if s["model"] == m]
        k = sum(s["category"] == "success" for s in L)
        lo, hi = wilson(k, len(L))
        out.append(f"- {m}: success {k}/{len(L)} = {k / len(L):.2f} [Wilson {lo:.2f}, {hi:.2f}]")
        stop &= k / len(L) < 0.10
    out.append(f"\nEvery local model below 10%: {'YES -> stop the local arm and ask about the Claude arm' if stop else 'no -> the local arm may proceed'}.\n")
    out.append("## Per run\n")
    out.append("| run | model | mode | category | stop | turns | no-block turns | cap-hit turns | gen tokens | wall s | tool s | spec rows | net rows | classes (submit cell) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        out.append(f"| {s['run_id'][:8]} | {s['model']} | {s['mode']} | {s['category']}{' (forced)' if s['forced'] else ''} | "
                   f"{s['stop_reason']} | {s['turns']} | {s['no_block_turns']} | {s['length_turns']} | {s['gen_tokens']} | "
                   f"{s['wall_s']:.0f} | {s['tool_s']:.0f} | {s['spec_rows']} | {s['net_rows'] if s['net_rows'] is not None else '-'} | "
                   f"{s['submit_cell_classes'] or '-'} |")
    out.append("\n## Flagged code cells (each read by hand; see STATUS.md)\n")
    out += [f"- run {f['run_id'][:8]} turn {f['turn']}: {', '.join(f['flags'])}" for f in flagged] or ["- none"]
    with open(os.path.join(ROOT, "report", "g4_pilot.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out[:12]))
    print(f"flagged cells: {len(flagged)}")


if __name__ == "__main__":
    main()
