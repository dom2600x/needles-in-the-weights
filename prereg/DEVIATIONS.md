# Deviations from PREREG_v1

Every change after the G3 freeze is logged here with its date, what changed, why, whether any
outcome had been seen, and where the pre-registered and amended results are both reported.
Changes before the freeze are revision notes in PREREG_v1.md itself.

**1. (2026-11-29, G4 pilot; seen after the runs, before scoring.)** The Mac went into idle sleep during pilot plan rows 10 and 11 (runs da5001d1 and dfd0ff96; pmset log, 2026-11-28 22:28 to 23:51 +0800).
- **What happened.** The wall-clock cap is measured with `time.monotonic`, which on macOS pauses during sleep. Both runs were therefore capped on awake time: run dfd0ff96 shows 1,801 s on the run clock against 3,805 s elapsed.
- **Effect.** The model and sandbox were suspended while the machine slept, so no run received extra computation. No outcome changes.
- **Reporting.** No re-run, since both runs completed normally. Both runs are reported as they are in report/g4_pilot.md.
- **Going forward.** Any later run phase will hold a sleep-prevention assertion (`caffeinate -i`) around the driver. That would be a driver change, not a change to caps or scoring.

