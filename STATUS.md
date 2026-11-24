# STATUS

## Current gate
G0 Preflight: complete. PR dom2600x/needles-in-the-weights#1 merged by the user (merge 15577e8); user re-ran the sandbox tests. D3 and D6 decided (session 4, prereg/G0_DECISIONS.md). Next: G1 testbed.

## Log
- 2026-11-23: Repository initialised with project instructions, pre-registration draft v0 and testbed/domain_check.py. No data collected.
- 2026-11-23 (session 2, 13:30Z): Started G0.
  - Draft pre-registration dated: prereg/PREREG_v0_DRAFT.md is in commit 1e81cbddb41a7cc1786d57535d7d09fa7edc5fec, committed 2026-11-23T13:16:00Z, on origin/main. Calibration predictions C1 to C6 are dated before any data.
  - Claude's shell is a Linux VM on the Mac, not macOS: Ubuntu 22.04 aarch64, 4 cores, 3 GB RAM, Python 3.10.12, numpy 2.2.6; scipy, pytest and mlx missing; PyPI and Hugging Face return 403 (not on the network allowlist). MLX (needs Metal) and sandbox-exec (macOS only) cannot run there, so none of the G0 measurements can be taken from the VM and VM numpy throughput is not a valid stand-in for the Mac.
  - Added harness/g0_preflight.sh (read-only; installs and downloads nothing). Committed in session 3.
  - Git from the VM is unsafe for now: the connected folder forbids file deletion, so git would leave .lock files. No commits or branches made this session.
- 2026-11-23 (session 3, 13:33Z): This session runs natively on macOS, so B1 is resolved. Ran harness/g0_preflight.sh except its network section (network use not yet approved).
  - Hardware: Mac15,12 (MacBook Air), Apple M3, 8 cores, 24 GB (25769803776 B). macOS 27.2.1 (26B131). Disk: 159 GiB free. pmset: no thermal or performance warnings recorded.
  - Tools: /usr/bin/python3 3.9.6; /opt/homebrew/bin/python3.13 3.13.18 (arm64); uv 0.12.17; git (Homebrew); sandbox-exec present; brew present. python3.11 and 3.12 missing.
  - Packages: numpy, scipy, pytest, mlx, mlx-lm, mlx-metal, transformers, huggingface-hub all MISSING in both python3 and python3.13.
  - HF cache: only sentence-transformers/all-MiniLM-L6-v2. No Qwen weights.
  - Git: no stale .lock files; native git works from this session.
- 2026-11-23 (session 3, cont.): User approved network use (in reply to a question covering package installs into a project venv and the Qwen3.5 model downloads; read as covering both).
  - Network: pypi.org 200, huggingface.co 200, github.com 200 (files.pythonhosted.org root 404, which is normal).
  - Venv: .venv (uv, CPython 3.13.18, gitignored). Installed: numpy 2.5.5, scipy 1.18.1, pytest 9.2.0, mlx 0.32.4, mlx-metal 0.32.4, mlx-lm 0.32.2 (needs >= 0.31.3), transformers 5.24.0, huggingface-hub 2.4.0. mlx runs on Metal (Device gpu,0); mlx_lm.models.qwen3_5 imports.
  - testbed/domain_check.py runs under the venv (output not checked against PREREG 3.1 numbers in this session).
  - Models (pinned revisions; both are mlx-vlm 0.3.12 conversions of the instruct models Qwen/Qwen3.5-2B and -9B; the "-Base" base_model tag is lineage metadata, which Qwen's own instruct card also carries):
    - mlx-community/Qwen3.5-2B-4bit @ 674aaa7240b91e8012fcad5d791b7dfe5ba90207 (1.72 GB)
    - mlx-community/Qwen3.5-9B-4bit @ 8b2b98c00a6b4d291155e4890773ca8f769aee53 (5.95 GB)
    - All safetensors SHA-256 match their LFS hashes. Note: huggingface-hub 2.x stores blobs in a shared hub/blobs/xx/ store, so du on a model folder shows only about 6 MB.
  - Benchmark: harness/g0_bench.py; raw records in runs/g0/bench.jsonl. Sampler temp 0.6, top_p 0.95, top_k 20, seed 0, thinking on, 256 generated tokens, weight-dump-like filler prompt.

    | model | prompt tok | prompt tok/s | gen tok/s | peak mem GB | wall s |
    |---|---|---|---|---|---|
    | 2B | 1,937 | 906 | 69.3 | 2.21 | 5.8 |
    | 2B | 9,998 | 875 | 63.9 | 2.55 | 15.5 |
    | 2B | 19,991 | 817 | 57.8 | 3.02 | 28.9 |
    | 9B | 1,937 | 192 | 18.2 | 6.51 | 24.2 |
    | 9B | 9,998 | 172 | 16.3 | 7.19 | 74.0 |
    | 9B | 19,991 | 139 | 11.6 | 8.19 | 166.3 |

  - Smoke tests (17*23, thinking on): 2B answered 391 in 529 tokens; 9B answered 391 in 507 tokens. Both closed </think>.
  - Thermal: pmset recorded no thermal or performance warning before or after any run. The 9B slowdown at 20k (gen 16.3 -> 11.6 tok/s) is consistent with longer attention context, but mild throttling cannot be ruled out because pmset shows no warning levels on this machine. Runs were short (about 6 min total); sustained-load throttling is untested.
  - Memory: 9B peaks at 8.2 GB at 20k context, well under the 18 GB cap.
  - FLAG F2 for G3 (no change made): PREREG section 7 caps are 12 turns x 6,144 generated tokens per turn and 30 min wall-clock per run. At 9B speeds of about 12 to 18 gen tok/s, 30 min allows roughly 22k to 32k generated tokens, about a third of the 73.7k per-turn cap total, before prompt processing. The wall-clock cap, not the token cap, will bind. This needs a decision when freezing caps at G3.
- 2026-11-24 (session 3, cont.): Built the proposer sandbox (user said "Yes, build sandbox").
  - Files: harness/sandbox/profile.sb (SBPL), harness/sandbox/kernel.py (child), harness/sandbox/host.py (parent), harness/sandbox/test_sandbox.py, harness/g0_throughput.py.
  - Design: three layers. (1) macOS sandbox-exec, deny by default: no network, exec only the Python interpreter (no fork or spawn), reads only /, /System, /usr/lib, /usr/share, the Python install, .venv, kernel.py and scratch, writes only to scratch. (2) An audit hook in the child: only numpy (and modules already loaded) may be imported; os.system, exec, spawn, fork, subprocess, socket, ctypes and similar events are refused; open, listdir and friends outside scratch are refused. (3) rlimits (64 MB per file, 256 fds, CPU), a 60 s per-cell soft timeout (SIGALRM, raised as a BaseException) plus a hard kill 5 s later, a total tool-time budget (600 s), an RSS cap (default 4 GB) and a 4,000-character output cap. Clean environment, HOME and TMPDIR set to scratch, python -I -B, only the two protocol pipes inherited.
  - Oracles run in the parent; the child holds only RPC stubs. The wire format is a JSON header plus raw float64 bytes, never pickle, so the child cannot execute code in the parent. Header and payload sizes are capped. Weight arrays passed in for mode W are read-only.
  - Profile adjustments found while testing: Homebrew's framework Python re-execs Resources/Python.app/Contents/MacOS/Python, so that binary is on the exec allowlist (anything it runs inherits the sandbox); the root directory "/" is readable (top-level names only); numpy.random, linalg, fft and polynomial are pre-imported before the hook goes on, because their lazy imports pull in stdlib modules the hook refuses. The host refuses scratch paths containing quotes or backslashes (SBPL injection guard).
  - Escape tests: harness/sandbox/test_sandbox.py, 90 passed (pytest 9.2.0, 103 s). Every escape attack runs twice, with the audit hook and with the OS layer alone, and must fail both times. Attacks covered: reading STATUS.md, PROJECT_INSTRUCTIONS.md, README.md, ~/.zshrc, /etc/passwd; listing $HOME, ~/.ssh, /Users and the repository; stat of repository files; writing to $HOME, the repository, /tmp, .venv and kernel.py (mtime unchanged); symlinking out of scratch; TCP, UDP, IPv6 and Unix sockets, DNS, binding a listener, loopback connection to a host listener (no connection received), urllib over https and plain http; os.system, subprocess, fork, posix_spawn, execv of /bin/sh, ctypes libc.open of a repository file, multiprocessing. In os-only mode every block is a genuine OS denial (EPERM, DNS failure, or libc.open returning -1), checked by reading each error message. Also tested: clean environment, no extra file descriptors, soft timeout keeps kernel state, hard kill when SIGALRM is ignored or the timeout is caught, memory cap, output cap, file-size cap, tool-time budget, and controls (numpy, scratch I/O, spec RPC, oracle errors reported inside the cell, the hook refusing a non-numpy import).
  - Known limits of the sandbox: the hook allows importing stdlib modules that numpy or the kernel already loaded (e.g. json, pickle, ctypes, threading); their dangerous operations are blocked by the hook and the OS layer instead. The RSS cap is enforced by polling ps every 0.5 s, so a very fast allocation can briefly exceed it. There is no scratch-directory total-size quota (only 64 MB per file). sandbox-exec is deprecated by Apple but works on macOS 27.2.1. A proposer can write raw protocol messages to its own pipes; this can only affect its own results and its oracle calls, whose metering lives in the parent.
  - Throughput in the sandbox (harness/g0_throughput.py; raw records in runs/g0/throughput.jsonl). d = 32, H = 96, 8 s per measurement, no thermal or performance warnings before or after:

    | dtype | mode | best evals/s | batch |
    |---|---|---|---|
    | float64 | forward only | 1.20e7 | 4,096 |
    | float64 | sample in ball + forward | 3.76e6 | 4,096 |
    | float32 | forward only | 1.83e7 | 4,096 |
    | float32 | sample in ball + forward | 5.30e6 | 4,096 |

    BLAS threads 1 versus 8 made no measurable difference at this size (within 3%), so the thread limit is not a meaningful brute-force cap.
  - FLAG F1 for G3 (no change made): PREREG section 7 estimated 1e8 to 1e9 evaluations in the 10-min tool cap. Measured: 1.1e10 (forward-only upper bound, float32) and 3.2e9 (realistic sample + forward). Expected hits in 600 s: at p = 1e-9, 11 (upper bound) or 3.2 (sample + forward); at p = 1e-10, 1.1 or 0.32. The pre-registered fallback (lower the hard cell to 1e-10 if over 0.1 expected hits) therefore still fails the 0.1 threshold. Keeping expected hits under 0.1 needs p <= about 3e-11 (sample + forward) or about 9e-12 (forward-only bound), or a smaller tool-time cap, or both. Caveats: these are 8 s bursts, not 10 min sustained on a fanless machine; and a hit found by net-only sampling still has to be recognised without the spec (needs a unit-activation tell), which the expected-hit count ignores.
- 2026-11-24 (session 3, end): Committed G0 as 453d9b3b6db49bdb93a9b4181b92d98ad64382c6 (2026-11-24T09:17:22Z) on claude/g0-preflight, pushed, and opened PR dom2600x/needles-in-the-weights#1 against main. No CI is configured on the repository. G0 ends here; next is G1 (testbed) after the user reviews and merges.
- 2026-11-24 (session 4): User merged PR #1, re-ran the tests, and delegated D3 and D6 to Claude. Decided both before any study data (no network generated, no auditor run). Full reasoning and rejected alternatives: prereg/G0_DECISIONS.md. Applied to prereg/PREREG_v0_DRAFT.md as a dated revision note and edits to sections 4, 7, 9 and 13. Branch claude/g0-decisions.
  - D6/F1: hard cell p = 1e-12 (was 1e-9). The bound uses p_s (boundary sampling via random rotations at forward-pass cost) and N = 1.1e10, giving 0.052 expected hits at 1e-12, against 3.9 at the draft's fallback of 1e-10 and 34 at 1e-9. Scripted grid extended to 1e-12. Tool-time cap unchanged.
  - D6/F2: caps unchanged (30 min, 12 turns, 6,144 tokens per turn, 10 min tool time); 30-min wall-clock named as the binding budget. Added: KV-cache reuse, a warning at 25 min, and a forced final-answer turn (thinking off, 512 tokens, submit only), with outcomes reported with and without forced answers. 92 local runs x 31.5 min = 48 h of the 60 h cap.
  - D3: Claude arm pre-registered, conditional on OS-enforced isolation (only run_python and submit tools, empty working directory, no memory or project context) and a passing canary test; otherwise dropped. Default model claude-opus-5-5, pinned at G3. Runs still need the user's go at G5.
  - For G1 (from these decisions): closed forms and exact conditional sampling must be checked down to p = 1e-12.
  - For G3: verify that headless Claude Code can be restricted to exactly the two tools; otherwise the fallback is the API, with a user-supplied key.

## Blockers
- ~~B1~~ resolved in session 3 (native macOS shell).
- ~~B2~~ resolved: packages installed and models downloaded (session 3).

## Decisions pending from the user
- ~~D2~~ resolved: installs and downloads work; approved and done in session 3.
- ~~D3~~ decided 2026-11-24 (delegated): yes, conditionally; see prereg/G0_DECISIONS.md.
- ~~D6~~ decided 2026-11-24 (delegated): hard cell p = 1e-12; caps unchanged, with a warning and a forced final turn; see prereg/G0_DECISIONS.md.
- ~~D5~~ resolved 2026-11-24: user approved committing G0 on branch claude/g0-preflight and opening a PR (the user merges).
