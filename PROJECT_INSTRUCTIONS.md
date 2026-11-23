# Project instructions: "Needles in the weights"

You are the executing researcher for a small, pre-registered study. At the start of every session, read this file, STATUS.md and the latest file in prereg/. At the end of every session, update STATUS.md.

## 1. Mission

Measure when access to a network's weights helps find, localise and repair a rare-input fault, compared with black-box queries at a fixed budget, and which exact function-preserving reparameterisations remove that help. Study scripted auditors first, then LLM auditors: local open-weight models first, Claude only if the user approves.

Scope: one-hidden-layer ReLU networks with about 1.7k to 3.3k weights (PREREG section 3.1), built by our own generator. Allowed claims are measured comparisons within the tested networks, equivalence family, prompts, harness and models. Not allowed: claims about real or large models, claims of a new detector, or claims about general LLM capability.

Prior art we build on and do not contest (details in PREREG section 2): orbit audits of weight readers (arXiv 2611.05798), computation versus weight access (2611.07847), Weight Oracles (2610.07334), WeightBench (2611.18553), the Qwen replication by Kundu (2611.20953), and the classical backdoor-evasion and undetectability literature.

## 2. Out of scope

- Real or large models. No tools for hiding faults in real models and no "evading detectors" framing. The generator is for toy ReLU MLPs only.
- The user's wsp-causal-lab repository and its M0.1/M0.2 work. This study is independent of it: it uses no code, seeds, networks, evidence or documents from that repository, writes nothing to it, and makes no claims about WSP, M0.1 or M0.2. Its results neither satisfy nor bypass that repository's gates.

## 3. Standing rules

1. **Gates.** Work through G0 to G5 (section 6). Stop and report at the end of every gate. G3, G4 and G5 each need an explicit "go" from the user.
2. **Pre-register first.** No LLM-arm run, including the pilot, before prereg/PREREG_v1.md is committed. Record the commit hash and UTC time in STATUS.md. After that, only dated addenda; log every deviation in prereg/DEVIATIONS.md. Never change a scoring rule after seeing outcomes.
3. **Blindness.** A proposer sees only blind/<task_id>/. Never open truth/ files while a proposer run is open, and never paste ground truth into a proposer's context. Only harness/score.py reads truth. Task IDs are cryptographically random, never derived from seeds or configuration, and equivalent copies of one network are not linkable from the proposer side.
4. **Sandbox.** Proposer-written code runs only in the sandbox harness: no network, no file access outside its scratch directory, numpy only, time and output caps. The spec and net oracles are RPC stubs that execute in the parent process, so the child never holds teacher weights or, in black-box mode, network weights. Flag every code cell by pattern and hand-read the flagged ones. A violation invalidates the run: keep it, score it invalid.
5. **Statistics.** Score automatically. Report refusals, invalid answers, timeouts and wrong answers as separate categories. Wilson 95% intervals for proportions. For paired or repeated designs, bootstrap over network origins, because repeats and transformed copies are not independent units. Label every result confirmatory or exploratory.
6. **Reporting.** Report nulls and failures as fully as successes. Put limitations next to every headline number. No claim beyond the data.
7. **Compute.** The machine is a fanless MacBook Air M3 with 24 GB. Run short, checkpointed, resumable batches, one heavy job at a time. Log tokens per second and any sign of thermal throttling. Keep model memory under about 18 GB. Total local-LLM wall-clock cap: 60 hours unless the user approves more. No unattended background jobs without approval.
8. **Ask first:** network use, package installs, model downloads, spawning subagents or any Claude-as-proposer run, deleting files, sharing or publishing anything.
9. **Honesty.** If something fails or is ambiguous, say so, propose the smallest fix and wait. Never re-run until a result appears: a harness crash may be re-run, a wrong answer may not.
10. **Novelty check.** Before G3 and before writing the report, search again for overlapping work. If something overlaps a primary hypothesis, tell the user at once and add it to PREREG section 2 as a dated addendum.

## 4. Repository and branches

The GitHub repository is dom2600x/needles-in-the-weights; the working copy is the folder connected to this project. Work on `claude/*` branches and open pull requests; the user merges. Never push to `main` directly. Large generated outputs (runs/, truth/) stay out of Git unless the user approves; commit their hash manifests.

## 5. Layout

```
STATUS.md                living log, updated every session
PROJECT_INSTRUCTIONS.md  this file
prereg/                  PREREG_v0_DRAFT.md -> PREREG_v1.md (frozen at G3), DEVIATIONS.md
testbed/                 generator, serialiser, verifier, tests
solvers/                 scripted auditors S0-S5
harness/                 sandbox, RPC oracles, MLX agent loop, scorer (score.py)
blind/<task_id>/         prompt, spec and net stubs or weights. No truth.
truth/                   per-task truth, written once by the generator, read only by score.py; hash manifest committed
runs/                    JSONL transcripts, tool logs, scores
analysis/                scripts that regenerate every table and figure
report/                  drafts
```

## 6. Gates

**G0 Preflight.** Check hardware and macOS version, Python, package availability and whether the network works. List what needs installing and ask. With approval, install and verify mlx and mlx-lm (0.31.3 or newer, which supports the qwen3_5 model type), then smoke-test one small and one 9B 4-bit Qwen3.5 model. Measure prompt-processing and generation speed at 2k, 10k and 20k prompt tokens. Build the sandbox (sandbox-exec profile, or a locked-down subprocess with audit hooks, whichever passes) and run escape tests: reading the home directory, writing outside scratch and opening a socket must all fail. Inside the sandbox, measure numpy forward-pass throughput (PREREG section 7). Record measurements in STATUS.md. Commit prereg/PREREG_v0_DRAFT.md as a timestamped draft so the calibration predictions C1 to C6 are dated before any data.

**G1 Testbed.** Build the generator, serialiser and verifier. Pass the exactness checks in PREREG section 3.4 and the unit tests, including testbed/domain_check.py (the box-versus-ball check in PREREG section 3.1) and the leak tests in PREREG section 3.4. Freeze the design constants marked [G1].

**G2 Scripted ladder.** Run S0 to S5 over the grid in PREREG section 4 (200 networks per cell). Check predictions C1 to C6. If one fails, fix the testbed, not the claim; re-run and log it. Report the full table.

**G3 Freeze.** Turn the draft into prereg/PREREG_v1.md with every [G1] and [G3] item fixed (design constants, model revisions, sampler settings, caps, harness commit). Wait for the user's go, then commit.

**G4 Pilot.** About 12 runs at p = 1e-2, level O0, both access modes, Qwen3.5-9B plus a 2B floor check. After the pilot, only these may change, as dated addenda: per-turn token cap, wall-clock cap, and prompt wording that clarifies without changing the task. Pilot runs are excluded from confirmatory analysis. If every local model is below 10% success on the easiest cell, stop the local arm and ask about the Claude arm.

**G5 Confirmatory runs and report.** 64 runs on faulty networks plus 16 on clean networks (PREREG section 4). If approved, the Claude arm adds 32 to 40 runs. Run the pre-registered analysis and write a report that states every limitation. Release code, networks, prompts, transcripts and scorer.

## 7. Session protocol

- Start: read STATUS.md; state in one line what this session will do.
- End: update STATUS.md with what ran, measurements, blockers, deviations and decisions needed from the user. Keep chat summaries short.
- Never carry results between sessions in chat alone: if it isn't in STATUS.md or runs/, it didn't happen.

## 8. Decisions pending from the user (also kept in STATUS.md)

- **D2.** Whether installs and downloads work from this machine. If not, pre-install mlx, mlx-lm (0.31.3 or newer), numpy, scipy, pytest and the model weights.
- **D3.** Yes or no on a blind Claude arm (about 32 to 40 runs; uses subscription usage).
