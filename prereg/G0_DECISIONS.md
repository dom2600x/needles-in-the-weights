# G0 decisions (2026-11-24)

The user delegated D3 and D6 to Claude ("move on to reasoning and deciding on the two open decisions autonomously", 2026-11-24). Both were decided before any study data existed: no network has been generated and no auditor has run. The only inputs are the G0 machine measurements in STATUS.md (raw records in runs/g0/). The resulting text is in prereg/PREREG_v0_DRAFT.md (revision note of 2026-11-24 and sections 4, 7, 9, 13). Nothing here authorises a run: every gate (G3, G4, G5) still needs its own go, and any Claude-as-proposer run still needs the user's explicit approval (PROJECT_INSTRUCTIONS rules 1 and 8).

## D6 / F1: hard-cell p versus brute-force throughput

**Decision.** The hard cell moves from p = 1e-9 to **p = 1e-12**. The scripted grid extends to 1e-12. The tool-time cap stays at 10 min.

**Reasoning.**
- The draft's rule: if the measured bound gives more than 0.1 expected hits at the hard-cell p, lower it to 1e-10. The rule expected 1e8 to 1e9 evaluations in 10 min. G0 measured 1.83e7 evaluations/s in the sandbox (float32 forward pass), which is 1.1e10.
- The right per-sample hit probability is p_s, not p. Uniform sampling on the boundary sphere is the strongest sampler that does not know u (PREREG 3.1). A proposer can sample the sphere once and apply random rotations: computing (X Q) W1 as X (Q W1) gives fresh, marginally uniform boundary points at forward-pass cost. So the bound is N p_s with N = 1.1e10.

| p | c | p_s / p | expected hits N p_s | with x2 margin |
|---|---|---|---|---|
| 1e-9 | 0.8178 | 3.11 | 34 | 68 |
| 1e-10 | 0.8435 | 3.57 | 3.9 | 7.8 |
| 1e-11 | 0.8653 | 4.10 | 0.45 | 0.90 |
| 1e-12 | 0.8838 | 4.71 | 0.052 | 0.10 |
| 1e-13 | 0.8997 | 5.41 | 0.006 | 0.012 |

- The pre-registered fallback (1e-10) fails its own criterion by a factor of about 40, so applying it literally would defeat its purpose. 1e-12 is the first decade that meets the criterion on the strongest bound. The bound is already conservative: it counts hits in A, not the smaller violation set V; it assumes a hit is recognised without the spec; and it uses 8 s bursts, which a fanless machine will not sustain for 10 min.
- I did not go to 1e-13. It would give a comfortable margin, but it also makes the needle cap thinner (R - tau falls from 0.39 to 0.34) and the outgoing weight larger, moving the testbed further from the draft's design for little gain, since the bound is already conservative.

**Alternatives rejected.**
- *Cut the tool-time cap* (for example to 60 s, which gives 0.45 expected hits at 1e-10). This would still need p <= 1e-11, and it would also limit legitimate structural analysis and the P2 cell, where fuzzing with the spec is meant to be feasible.
- *Throttle the sandbox* (fewer threads, a slower BLAS). BLAS threads made no difference, and an artificially slow machine would make "white box" mean something other than "weights plus free self-implemented evaluation" (PREREG 7).
- *Keep 1e-9 and rely on the codebook* (count F-large runs separately). At 34 expected hits the "fuzzing cannot work" premise of P1 would be false, not just confounded.

**Consequences.** P1, P3, the black-box ceiling and the pre-specified readings now refer to p = 1e-12. G1 must check the closed forms and exact conditional sampling at p down to 1e-12 (PREREG 3.4). G2 adds the 1e-11 and 1e-12 rows to the scripted ladder; the fuzzing auditors are reused across O and g, so the cost is small. The B-mode ceiling becomes Q p_s = 4.7e-9.

## D6 / F2: wall-clock versus token caps

**Decision.** The caps stay at 30 min wall-clock, 12 turns, 6,144 generated tokens per turn and 10 min tool time. The 30-min wall-clock is named as the binding budget, and the harness adds three things fixed now: KV-cache reuse across turns, a time warning after 25 min, and one forced final-answer turn (thinking off, at most 512 tokens, only a submit block runs). Forced-turn answers are flagged, and primary outcomes are reported with and without them.

**Reasoning.**
- At G0 9B speeds (11.6 to 18.2 generated tokens/s), 30 min allows about 14k to 29k generated tokens, or 2 to 5 full turns. The turn and token caps will rarely bind.
- More wall-clock is not available without the user: the local arm is 92 runs (12 pilot + 80 confirmatory). At 31.5 min each that is 48 h of the 60 h cap; 45 min would be about 70 h.
- Lowering the per-turn token cap so that 12 turns fit would cut thinking short mid-turn and create invalid turns, which the scoring would mix up with model failure.
- Without a forced final turn, runs that found the answer but ran out of time would be scored as timeouts, so the result would measure the cap rather than the auditor. Reporting with and without forced answers keeps both readings visible.
- KV-cache reuse is a harness requirement, not a cap change: without it, re-reading a 20k-token context costs about 144 s per turn.
- The pilot may still change the per-turn token cap and the wall-clock cap as dated addenda (PREREG 11). These mechanisms let the pilot measure how often the clock binds.

## D3: blind Claude arm

**Decision.** **Yes, conditionally.** The arm is pre-registered now, so that if it runs it is confirmatory for that arm and not designed after seeing local results. Running it still needs the user's explicit go at G5 (and rule 8). Conditions:
1. Isolation enforced by configuration and the operating system, not by instruction. The proposer is a fresh headless session with no memory, no CLAUDE.md or project context, started in an empty directory holding only blind/<task>/. Its only tools are run_python and submit, backed by the same sandbox and RPC oracles as the local arm; built-in file, shell, web and agent tools are off.
2. A canary session with no task data must first fail to read a canary file outside its directory, list the repository, reach the network or start a subagent. If this cannot be shown, the arm is dropped rather than run under instruction-only isolation.
3. One pinned model (default claude-opus-5-5, confirmed at G3 with a usage estimate), the same prompt, caps and scorer as the local arm, analysed separately, with no cross-model ranking.

**Reasoning.**
- The pilot stop rule (PREREG 4) asks about the Claude arm exactly when the local models fail. Deciding the arm's design then, after seeing local results, would be a forking path. Pre-registering it now costs nothing and keeps that option clean.
- A capable proposer is also what gives P1 and P3 their meaning. "No structural exploitation at this model size" is a weak finding if no model that could plausibly exploit structure was tested.
- The main risk is blindness: a Claude session with ordinary tools could open truth/. Instruction-only isolation, which the draft allowed, is not good enough for a confirmatory arm, so condition 1 replaces it and condition 2 tests it.
- Conflict of interest: Claude executes this study and would also be the audited model. Mitigations: the prompt and caps are frozen at G3 before any Claude run, scoring is automatic, the executing session never opens truth/ while a proposer run is open, all transcripts are released, and the Claude arm is never ranked against the local arm.
- Cost: 40 runs of at most about 31.5 min each, on the user's subscription usage. A per-run usage estimate is part of the G3 freeze, and the user can still decline at G5.

**Open for G3 (not decided here).** Whether headless Claude Code can be restricted to exactly these two tools needs checking (tool allowlist, MCP-only configuration, no setting sources). If it cannot, the fallback is the Anthropic API with the harness's own tool loop. That would need an API key, which the user must supply and which Claude never handles.
