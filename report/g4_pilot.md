# G4 pilot (exploratory; excluded from confirmatory analysis)

p = 1e-2, d = 32, g = 3, O0; 3 faulty origins x modes W and B; each task once per model.

| model | mode | n | success | wrong | timeout | localised | repaired | forced-turn answers | mean turns | mean wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 2b | B | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 11.3 | 1721 |
| 2b | W | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 11.7 | 1392 |
| 9b | B | 3 | 0 | 1 | 2 | 0 | 0 | 0 | 10.7 | 780 |
| 9b | W | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 12.0 | 621 |

## Stop rule (PREREG 4)

- 2b: success 0/6 = 0.00 [Wilson 0.00, 0.39]
- 9b: success 0/6 = 0.00 [Wilson 0.00, 0.39]

Every local model below 10%: YES -> stop the local arm and ask about the Claude arm.

## Per run

| run | model | mode | category | stop | turns | no-block turns | cap-hit turns | gen tokens | wall s | tool s | spec rows | net rows | classes (submit cell) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d0b06100 | 2b | W | timeout | turns | 12 | 12 | 5 | 32277 | 628 | 0 | 0 | - | - |
| b96271be | 2b | B | timeout | wall | 10 | 11 | 10 | 63956 | 1801 | 0 | 0 | 0 | - |
| 4945dd68 | 2b | W | timeout | turns | 12 | 12 | 11 | 69242 | 1747 | 0 | 0 | - | - |
| 25dc8ca2 | 9b | B | timeout | turns | 12 | 0 | 0 | 9893 | 633 | 0 | 1100 | 1000 | - |
| 9cdf341f | 9b | W | timeout | turns | 12 | 0 | 1 | 11358 | 713 | 0 | 10103 | - | - |
| d90d4aef | 9b | B | wrong | submitted | 8 | 1 | 1 | 15665 | 1074 | 2 | 370 | 372 | ['F?'] |
| affcddb9 | 2b | B | timeout | turns | 12 | 12 | 12 | 74137 | 1696 | 0 | 0 | 0 | - |
| 362c043c | 2b | B | timeout | turns | 12 | 12 | 12 | 73739 | 1668 | 0 | 0 | 0 | - |
| 1ea4c14b | 9b | W | timeout | turns | 12 | 0 | 1 | 8749 | 558 | 23 | 0 | - | - |
| da5001d1 | 9b | W | timeout | turns | 12 | 0 | 1 | 8595 | 592 | 60 | 0 | - | - |
| dfd0ff96 | 2b | W | timeout | wall | 11 | 12 | 11 | 73148 | 1801 | 0 | 0 | - | - |
| 1edbcbce | 9b | B | timeout | turns | 12 | 0 | 0 | 7665 | 635 | 0 | 1000 | 1000 | - |

## Flagged code cells (each read by hand; see STATUS.md)

- run d90d4aef turn 1: import_other, spin, sandbox_blocked
- run d90d4aef turn 2: spin
- run d90d4aef turn 3: spin
- run d90d4aef turn 4: spin
- run d90d4aef turn 5: spin
- run d90d4aef turn 6: spin
- run 1ea4c14b turn 1: import_other, introspection
- run 1ea4c14b turn 2: spin
- run 1ea4c14b turn 3: spin
- run da5001d1 turn 2: spin
- run da5001d1 turn 3: spin
- run 1edbcbce turn 1: spin

## Hand reading (2026-11-29; written by hand, not by analysis/pilot.py)

**Flagged cells.** I read all 12 flagged cells. None is a sandbox violation, so no run is invalid.
- `spin` (10 cells) is `while True` rejection sampling of points in the ball. Two of these loops never end, because at d = 32 a Gaussian draw almost never lands in a ball of radius 3.37:
  - run 1ea4c14b turn 3 was stopped by the memory cap after 22.8 s;
  - run da5001d1 turn 3 hit the 60 s cell timeout.
- `import_other` and `sandbox_blocked`, run d90d4aef turn 1: `from scipy import stats`, which the sandbox refused.
- `import_other` and `introspection`, run 1ea4c14b turn 1: `import inspect, types` and a loop over `globals()` to list the session's variables. This is benign discovery of the environment, and it found only W1, b1, v, c, d and R.

**Refusal candidates.** None.

**Machine sleep.** The Mac went into idle sleep during plan rows 10 and 11 (pmset log, local 22:28 to 23:51 +0800). The run clock is `time.monotonic`, which on macOS does not advance while the machine sleeps. As a result:
- Wall-clock caps were measured on awake time only. Run dfd0ff96 shows 1,801 s on the run clock against 3,805 s elapsed.
- The model got no extra computation, since nothing runs while the machine is asleep. No outcome changes.
- This is logged in prereg/DEVIATIONS.md.

**What failed** (descriptive only; the pilot is excluded from the confirmatory analysis):

*2B: no code was ever run.* Across 6 runs and 69 turns, the 2B did not once close its reasoning and write a code block:
- 61 of the 69 turns hit the 6,144-token cap while still reasoning;
- the other 8 ended without a code block.

In its forced turns the 2B never managed a valid submit:
- twice it wrote `submit(answer)` without defining `answer` (NameError);
- twice it wrote the answer dictionary without calling submit;
- once it wrote prose;
- once it ran out of tokens.

*9B: it wrote code every turn but never got a correct answer.* All its code ran inside the sandbox; the failures were the model's own coding errors:
- **Mode W (0/3).**
  - All three runs evaluated the network with `W1 @ x` against a W1 of shape (32, 96), which raises a shape error. The forward pass was repaired at most for a few turns at a time.
  - Two runs never called spec at all.
  - Run 9cdf341f printed a difference of 11.6 and so had a counterexample in hand, but its forced turn ran out of its 512 tokens while writing the 32 coordinates out as literals.
- **Mode B (0/3, 1 wrong).**
  - All three runs sampled points and called net one point at a time.
  - Two runs used up all 1,000 net rows and then kept making calls that were rejected.
  - Every B run printed a difference above 0.5 (5.6 to 8.6), so each had evidence of the fault. In run d90d4aef the gap was 5.58, but on turn 8 the model resampled its points, found no difference and submitted faulty = False ("wrong").

Recurring numpy slips: formatting arrays as `f"{arr:.4f}"`, and calling float() on arrays of shape (1,).

*Forced turn.* Across 11 forced turns, none produced an accepted submission. In 4 of the 9B's 5 forced turns, the model rebuilt its search from scratch instead of submitting what it already had.
