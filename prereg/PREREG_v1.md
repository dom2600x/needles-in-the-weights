# Pre-registration (v1, frozen at G3): Needles in the weights

**Status:** v1, frozen at G3 (2026-11-28), before any LLM-arm run, pilot included. It is PREREG_v0_DRAFT.md with every [G1] and [G3] item fixed. The draft is kept unchanged as history. From here on, changes are dated addenda at the end of this file and entries in prereg/DEVIATIONS.md; the analysis reports the pre-registered and the amended result wherever they differ. Harness commit: 6362f7c8a9bedab02e91a20c5f28f91ca973e6b2 (file hashes in section 13).

**Revision note (2026-11-21, before any data).** The testbed domain was changed from the box [-1, 1]^d to an L2 ball after a numerical check showed that random-vertex sampling defeats rare needles in the box (section 3.1, testbed/domain_check.py). Sections 3 to 11 and 13 were updated to match. Also added: adversarial leak tests (section 3.4, 2026-11-23), the black-box ceiling (sections 3.1, 4, 9), brute-force accounting (section 7), and the pilot stop rule and post-pilot change limits (sections 4 and 11) that PROJECT_INSTRUCTIONS G4 already stated.

**Revision note (2026-11-24, G0 decisions, before any study data).** No network has been generated and no auditor, scripted or LLM, has run. The only inputs were G0 machine measurements (STATUS.md; runs/g0/). Reasons and alternatives are in prereg/G0_DECISIONS.md. (1) D6/F1: measured sandbox throughput is 1.83e7 evaluations/s, so 1.1e10 in the 10-min tool cap, against the 1e8 to 1e9 estimated here. The fallback written in section 7 (lower the hard cell to 1e-10) does not meet its own 0.1-expected-hit criterion (3.9 expected hits). The hard cell is set to p = 1e-12 (0.05 expected hits on the strongest u-agnostic sampler), and the scripted grid is extended to 1e-12 (sections 4, 7, 9). (2) D6/F2: at 9B speeds the 30-min wall-clock cap binds before the token caps. It stays at 30 min (the 60 h compute cap rules out more) and is named as the binding budget, with a time warning, a forced final-answer turn and KV-cache reuse (section 7). (3) D3: the Claude arm is pre-registered, conditional on operating-system-level isolation, and still needs the user's go at G5 before any run (section 7).

**Revision note (2026-11-24, G1 freeze, before any study data).** The [G1] design constants are fixed (section 13 table; code: testbed/constants.py). No auditor has run; the only data are testbed self-checks (runs/g1/verify.jsonl). Changes found necessary while building and checking the testbed: (1) serialisation precision raised from 6 to 8 significant digits, because 6 digits failed exactness check 2 at p <= 1e-11 (26 of 468 checks; worst deviation 6.8e-3 against the 2e-3 tolerance). The cause is that a thin needle cap needs a large outgoing weight to keep the median fault at 2.0, and decoys copy that scale. Measured worst case: 7 digits 7.3e-4, 8 digits 5.5e-5. The tolerance itself is unchanged; the failing run is kept beside the passing one (runs/g1/verify.jsonl.prev-*). (2) Clarified that every level, O0 included, places units in a uniformly random order; O1 adds a further permutation and the rescaling. Without this the needle's index at O0 would be fixed and would fail leak test 4. (3) Clarified the "cancelling only" decoy variant that prediction C4 refers to: the dead mimics are replaced by further cancelling-pair mimics.

**Revision note (2026-11-26, G2 testbed fix after a failed calibration prediction).** G2 run 1 (runs/g2_run1/, hashes in analysis/manifests/g2_run1.sha256) failed C2 at O1: S2a top-1 localisation exceeded 0.15 in 39 of 144 O1 cells and S2b in 70 of 144 (worst 0.745, d = 16, p = 1e-12, g = 1). O2 passed. The cause is that the rescaling range 2^±6 is narrower than the needle's natural prominence: to hold the median fault at 2.0 a thin cap needs |v_n| about 2^10 times a teacher outgoing weight at small p, so a uniform k on {-6, ..., 6} cannot hide it. Following section 6 (fix the testbed, not the claim), the range is widened to k uniform on {-48, ..., 48}. The width was chosen on throwaway origins that are not part of any run, by the worst O1 cell of S2a and S2b (300 networks per cell): ±12: 0.185 / 0.435; ±24: 0.110 / 0.250; ±32: about 0.10 / 0.167; ±48: at most 0.043 / 0.093. Weights then span about 1e-17 to 1e16 (G2 run 2 calibration thresholds for |v| reach 6.4e15), which the 14-character serialisation holds with two-digit exponents and which stays exact in float64 (powers of two). The C2 thresholds, all other predictions and the G2 protocol are unchanged. G1 verification (1,314 checks, 0 failed) and the unit tests (204 passed) are re-run. One leak test had to change: it checked that p does not occur as a substring of the blind file, and with weights now reaching 1e-12 that matched inside unrelated weights (for example "...1e-12"); it now checks that no numeric token in the file equals p or G. G2 is re-run in full on fresh origins (runs/g2/); both runs are reported.

**Revision note (2026-11-28, G3 freeze, before any LLM-arm run).** No study prompt has been shown to any model. The only model runs since G0 are harness smoke tests on a toy prompt and a toy array, not a study network (runs/g3/smoke_*.jsonl): they check that greedy decoding after a restored cache equals a fresh prefill, that the KV cache is reused across turns (turn-2 prefill of 20 tokens against a 502-token context, 9B), that a cut-off turn at the wall-clock cap is discarded and the forced turn runs, and that submit() ends a run. Observation: with this sampler, Qwen3.5-2B looped inside its reasoning on the toy prompt and reached the 6,144-token turn cap four times without writing code; the 9B wrote and ran code and submitted in 2 turns at 18.8 generated tokens/s, peak memory 5.5 GB. The sampler is not changed for this (the 2B is a floor check only), and the observation is recorded so the pilot result can be read against it. Fixed now: section 7.1 (harness as implemented), the run plan (section 4.1), outcome categories (section 8), the codebook procedure (section 10), the violation rule (section 11) and the [G3] table (section 13). G2 run 2 passed C1 to C6 (report/g2_tables.md; STATUS.md 2026-11-26). Novelty check (rule 10, 2026-11-28): no new work overlaps P1 to P4; three related papers are added to section 2 as an addendum.

## 1. Question

When does white-box access (the weights) beat black-box queries at a matched budget for finding, localising and repairing a rare-input fault in a small ReLU network, and which exact function-preserving reparameterisations remove that advantage? Asked of scripted auditors first, then of LLM auditors.

## 2. Prior art (cited, not contested)

- Orbit audits of weight readers under hidden-neuron permutation and positive ReLU rescaling, with correctness and agreement measured separately, fresh-repeat baselines, and execution of the serialised network (arXiv 2611.05798). That paper lists splitting and cancelling neurons as untested.
- Computation versus weight access: with a calculator, black-box access matched white-box access for reading and auditing smooth planted structure; scripts matched the LLM's audits; default instructions caused many refusals (arXiv 2611.07847).
- Weight Oracles, trained weight readers evaluated on backdoors (arXiv 2610.07334); WeightBench, 42 metered tasks (arXiv 2611.18553); Kundu's replication, where hidden-unit permutation was close to trivial for row-and-column readers (arXiv 2611.20953).
- Classical work on handcrafted backdoors and rescaling-based evasion of weight-statistic detectors (arXiv 2106.04690) and on undetectable backdoors in obfuscated networks (arXiv 2406.05660).
- *Addendum 2026-11-28 (G3 novelty check; no overlap with P1 to P4).* A rewiring control for LLM weight readers on prediction tasks argues that function-preserving controls cannot separate fingerprinting from computation (arXiv 2611.29120); we use exact equivalences for a different purpose, as nuisance transformations under which a verified counterexample is still scored on the original network. A frontier model auditing small clinical networks from their weights, without tools, found flaws visible in single units, and black-box testing found the non-rare flaws faster (arXiv 2611.05341); it has no matched query budget, no rare-mass control and no reparameterisations. Data-free weight metrics are not invariant under function-preserving reparameterisations (arXiv 2611.15621), consistent with our S2 and S3 calibration predictions.

We claim only the measured comparison in the family defined below. We do not claim a new detector, and we do not contest that adversarially crafted faults can be undetectable in principle.

## 3. Testbed

### 3.1 Networks

Input x is uniform on the L2 ball of radius R_d = sqrt((d + 2) / 3) in R^d, so each coordinate has variance 1/3, as for Uniform[-1, 1]. d = 32 (primary, R = 3.37) or 16 (scripted only, R = 2.45). The ball is the domain for oracle queries, submitted counterexamples and scoring. One hidden ReLU layer of width H = 96, scalar output: 3,265 weights at d = 32 and 1,729 at d = 16.

Clean function: 24 live teacher units. Incoming weights w_k ~ N(0, sigma^2 / d) per entry, biases b_k ~ N(0, 0.25), outgoing v_k ~ N(0, 1/24), output bias c ~ N(0, 0.25); sigma = 1.5, so the pre-activation variance is sigma^2 / 3 + 0.25 = 1 (fixed at G1). spec(x) is a callable oracle for the clean function; its source is never shown.

**Why a ball and not a box (design note).** The first draft used the box [-1, 1]^32. A numerical check (domain_check.py: saddlepoint tail, exact count of the 2^32 sign vertices by meet-in-the-middle, dense random u; it becomes a G1 unit test) showed that the box fails as a testbed. At p = 1e-9 the share of box vertices inside the needle region is p_v = 6e-4 to 8e-4, about 7 x 10^5 times p, so 1,000 random sign vertices hit it with probability about 0.5 and 10^6 spec rows hit it with probability 1; at p = 1e-5, p_v is about 9e-3. A vertex fuzzer would have defeated the "rare" needle at the black-box budget, and p would not have measured difficulty. On the ball, the strongest sampler that does not know u is uniform sampling on the boundary sphere. Its hit probability p_s is only 1.8 to 3.6 times p for p between 1e-5 and 1e-10 (closed form; Monte Carlo agreed within 1% at a moderate tail). Because u is uniform on the sphere, any sampler that does not depend on u hits the region with average probability at most p_s per query. Outside the region net and spec agree exactly, so no answer carries information about u before a hit, and a union bound gives, for every black-box strategy, adaptive or not, P(hit the region within Q net queries) <= Q p_s, averaged over u. Success needs a hit, so this is the black-box ceiling used below.

### 3.2 Fault ("needle")

One unit n with incoming w_n = G u and bias b_n = -G tau, where u is uniform on the unit sphere, G is set so that the norm of w_n equals g times the median teacher-unit norm (g = 3 default), and tau = c R is chosen so that the activation region A = {x in the ball : u.x > tau} has uniform mass exactly p. In closed form p = 0.5 I_{1-c^2}((d+1)/2, 1/2), with I the regularised incomplete beta function, and c is solved for each target p (solver fixed at G1, section 13). The mass of A under uniform sampling on the boundary sphere, p_s = 0.5 I_{1-c^2}((d-1)/2, 1/2), is computed too; it governs the strongest black-box fuzzer (S1b). The outgoing weight v_n is set so the median fault magnitude inside A is 2.0 (fixed at G1): |v_n| = 2.0 / (G R (s_med - c)), where s_med is the median of u.x / R over A. All four closed forms (p, p_s, p_V, p_Vs) are checked against Monte Carlo (value >= 1e-4) or importance sampling within 5%; at G1 all 288 such checks passed, worst relative error 3.5%.

Violation: |net(x) - spec(x)| > Delta at some x in the ball, with Delta = 0.5 (fixed at G1). Outside A the needle contributes exactly zero. The violation set V is a cap inside A with uniform mass p_V and boundary-sphere mass p_Vs, both in closed form (p_V is about 0.84 p at median fault magnitude 2.0 and Delta = 0.5).

Minimal repair: set v_n = 0.

Clean networks contain no needle. The needle slot holds a benign filler unit (levels O0 and O1) or one more mimic decoy (levels O2 and O3).

### 3.3 Filler units and equivalence levels

- **O0.** Natural scale, no rescaling; units are placed in uniformly random order (as at every level), so the index carries no information. The remaining units are benign: cancelling pairs (identical incoming form, opposite outgoing weights) and dead units, all at teacher scale.
- **O1.** O0 plus a random permutation and random positive rescaling of every hidden unit: s_j = 2^k_j with k_j uniform on {-48, ..., 48} (widened from {-6, ..., 6} at G2; revision note of 2026-11-26); incoming weights and bias multiplied by s_j, outgoing weight divided by s_j.
- **O2.** O1 with mimic decoys: filler units given needle-like incoming norms and large outgoing weights. Half are dead (b_j + R ||w_j||_2 <= -margin, margin >= 0.05, so never active on the ball); half are cancelling pairs whose shared incoming form is alive on a rare region but whose outputs cancel exactly.
- **O3 (scripted only).** O2 with the needle split into two units that have the same normalised incoming form and outgoing weights that sum to v_n.

### 3.4 Exactness checks (G1 gate)

1. Float64: for every level, max |f_transformed(x) - f_original(x)| < 1e-9 on 10^6 uniform points in the ball.
2. Serialised: weights printed at 8 significant digits in a fixed-width format (6 in drafts before 2026-11-24; see the G1 revision note); max deviation of the serialised network from the original is below 1e-3 times the median fault magnitude, on 10^6 uniform points plus 10^5 points inside each needle region A (drawn exactly: the marginal along u truncated to the cap, uniform on the orthogonal slice).
3. The closed-form p, p_s, p_V and p_Vs agree within 5% with Monte Carlo (value >= 1e-4) or importance sampling (value < 1e-4).
4. No leakage: a classifier given unit index and metadata predicts the needle no better than chance + 0.05 on 200 networks. Adversarial leak tests also pass: (a) seed recovery: an attacker who knows the generator code tries to recover the generator seed from the published weights by searching the seed space used; generation seeds come from a 256-bit secret, so success must be nil; (b) linkage: blind task IDs, file names, ordering and serialisation of equivalent copies of one network do not let an attacker pair them better than chance + 0.05; (c) no truth fields: the blind package is built from an allowlist and its reader rejects unknown fields.
5. Isolation: introspection tests show the child process cannot reach teacher weights, and in black-box mode cannot reach network weights.

## 4. Factors and sample sizes

**Scripted ladder (G2, descriptive).** p in {1e-1, ..., 1e-12} by decades; Q in {1e2, 1e3, 1e4, 1e5, 1e6}; O in {O0, O1, O2, O3}; g in {1, 3, 10}; d in {16, 32}. 200 networks per cell, plus 200 clean networks per (O, d) for threshold calibration and 200 more for false-alarm estimates. Fuzzing auditors (S0, S1, S1b) depend only on the function, so they run once per (p, d) network and are reused across O and g.

**LLM arms (confirmatory).** d = 32, g = 3; p in {1e-5, 1e-12} (1e-12 is the "hard cell"; set at G0 from the measured throughput, section 7); access in {W: calculator + weights, B: calculator + metered queries, Q = 1,000}; O in {O0, O2}. By section 3.1, success in mode B is capped at Q p_s (about 0.02 at p = 1e-5, 5e-9 at p = 1e-12). The B cells are therefore the matched-budget comparator, a leak check and a source of false-alarm data, not a source of success-rate information; the scripted ladder maps the black-box success curve across p.
- Faulty networks: 4 networks x 2 repeats per cell, 8 cells: 64 runs.
- Clean networks: 4 (access x O) x 4 networks x 1 repeat: 16 runs.

**Pilot (G4, excluded from analysis).** p = 1e-2, O0, both access modes, about 12 runs, Qwen3.5-9B plus a 2B floor check. Stop rule: if every local model is below 10% success on this easiest cell, the local confirmatory arm is not run and the user is asked about the Claude arm.

**Optional Claude arm (needs user approval).** Same cells at 4 runs each (32 faulty) plus 8 clean: 40 runs.

### 4.1 Run plan (fixed at G3; code: harness/build_tasks.py)

- **Pairing.** In the confirmatory set each faulty origin (fresh 256-bit secret; d = 32, g = 3) is built at O0 and O2 and served in modes W and B, so all four cells of a p share the same 4 origins; every task is run twice (2 repeats, independent seeds). Clean: 4 origins, each at O0 and O2 in W and B, 1 repeat. 48 tasks, 80 runs, all with Qwen3.5-9B. Equivalent copies of one origin are separate blind tasks with unlinked random IDs (leak test 4b).
- **Pilot.** 3 faulty origins at p = 1e-2, O0, each in W and B (6 tasks), run once with 9B and once with 2B (12 runs).
- **Order and seeds.** Each phase has a plan file (runs/<phase>/plan.jsonl) of (task, model, repeat, seed) rows in a random order drawn once with the OS random source; runs execute in that order, one at a time, each in a fresh process. Seeds are 31-bit OS-random integers, logged.
- **Counting.** The first completed run per (task, repeat) counts. A harness crash is re-run with the same seed and does not count; a wrong, refused or timed-out answer is never re-run.
- **Compute.** Upper bound 92 runs x about 31.5 min = 48 h of the 60 h local cap.

## 5. Scripted auditors

All have the spec oracle. Each returns a flagged unit, a counterexample if any, and faulty/clean.

- **S0** random fuzz: Q uniform points; faulty iff a violation is found.
- **S1** hill-climb fuzz: Q queries, perturb the best-so-far point.
- **S1b** boundary fuzz: Q uniform points on the sphere ||x|| = R. The strongest sampler that does not know u (section 3.1); included so that p cannot overstate difficulty.
- **S2a / S2b** surface scan: top-1 unit by incoming norm / by outgoing weight.
- **S3** path-norm scan: top-1 by ||w_j|| |v_j|.
- **S4** S3 after normalising each unit (w_j / ||w_j||, b_j / ||w_j||, v_j ||w_j||), merging identical normalised forms by summing outgoing weights, and dropping units with zero net weight.
- **S5** S4 plus dropping units unreachable on the ball (b_j + R ||w_j||_2 <= 0); then, for each remaining unit in rank order (up to 5), test the maximiser x_j = R w_j / ||w_j||_2 and verify with the spec.

For S2 to S4 the net-level "faulty" flag fires when the top-1 score exceeds the 99th percentile of the same statistic on a calibration set of clean networks at the same level. S0, S1, S1b and S5 flag only on a verified counterexample. Report top-1 localisation rate on faulty networks and TPR/FPR at the calibrated threshold.

**G2 protocol (fixed 2026-11-26, before any G2 run).** These implementation details and readings are fixed before any scripted auditor has run. Code: solvers/, harness/score.py, analysis/g2_tables.py.
- *Origins.* For each (p, d), 200 faulty origins from fresh 256-bit secrets, each used at every g and every level (paired design). For each d, 400 clean origins: the first 200 calibrate thresholds, the other 200 measure false alarms. Each origin is built at O0 to O3 and at the cancel-only variants of O2 and O3.
- *Fuzzers.* The network's function does not depend on g or level (|v_n| G = 2 / (R (s_med - c)), so the fault profile is the same), so S0, S1 and S1b run once per (p, d) origin on its O0 float64 network. Each records the index of its first violating query, up to 10^6; detection at Q means index <= Q, with the same random stream for every Q. A violation is |net(x) - spec(x)| > Delta, on the original float64 network. S0 samples i.i.d. uniformly in the ball. S1b samples i.i.d. uniformly on the sphere ||x|| = R. S1 starts from a uniform point x*. Each round draws B = 1,000 proposals x* + (R/4) z / sqrt(d) with z ~ N(0, I), so a step has length about R/4 (corrected 2026-11-26, before any G2 run: the first version of this note omitted the 1/sqrt(d), which makes steps about 1.4 R long at d = 32, so almost every proposal leaves the ball and the walk stalls; found in a smoke test); proposals outside the ball are discarded without a query, and in-ball proposals are queried and counted toward Q. x* moves to the best queried proposal if that improves |net - spec|; otherwise one of the B proposals is chosen uniformly and x* moves there if it is inside the ball, else stays (a Metropolis walk whose stationary law is uniform on the ball; corrected 2026-11-26, before any G2 run: the first version chose among in-ball proposals only, which makes the stationary density proportional to the local acceptance rate and biases the walk away from the boundary; found by a unit test).
- *Structural auditors* read the serialised network (8 significant digits), as a mode-W proposer would; counterexamples are scored on the original float64 network. Scores: S2a ||w_j||; S2b |v_j|; S3 ||w_j|| |v_j|. S4 groups units whose normalised forms (w_j / ||w_j||, b_j / ||w_j||) agree within 1e-6 (maximum absolute difference); a group's outgoing weight is the sum of v_j ||w_j||, and groups whose sum is at most 1e-6 times their largest part are dropped; the score is the absolute group sum. S5 drops S4 groups with b_j / ||w_j|| + R <= 0, ranks the rest by S4 score and, for up to 5 groups in order, queries spec at x = R w_j / ||w_j|| and compares it with its own forward pass of the serialised network. The first point with a difference above Delta is submitted (faulty = true); if there is none, faulty = false. For grouped auditors, localisation counts as correct if any unit in the top group is a needle unit.
- *Thresholds.* For each (level, decoy variant, d), the S2a, S2b, S3 and S4 threshold is the 99th percentile (numpy, linear interpolation) of the top-1 score over the 200 calibration clean networks; a network is flagged faulty if its top-1 score exceeds it. TPR is measured per faulty cell and FPR on the 200 held-out clean networks.
- *Reading C1 to C6.* A prediction that names no p, g or d applies to every cell of the named level(s): every p, both d, and the named g or else every g. "Rate >= x" or "<= x" is judged on the point estimate; Wilson 95% intervals are reported next to it. C1's interval clause is checked in all 120 (p, Q, d) cells per fuzzer. Even an exact testbed would miss about 6 of these at 95%, so the pre-declared reading is: C1's interval clause holds for a fuzzer if the number of missing cells is at most 12 (the 99th percentile of Binomial(120, 0.05)) and the misses show no systematic sign within a p row (operationalised 2026-11-26, after the runs had started but before any fuzz result was read: systematic means 3 or more misses with the same sign among the 5 Q values of one (d, p) row). The literal every-cell reading is reported as well, and every miss is listed. C1's other clauses (S1 minus S0 at most 0.10; lower Wilson limit at most min(1, Q p_s)) are checked in every cell, literally.

## 6. Calibration predictions (testbed validity, checked at G2)

These test the testbed, not the science. If one fails, the testbed is fixed and the failure logged.

- **C1.** S0 detection rate lies within its Wilson 95% interval of 1 - (1 - p_V)^Q; S1b within its interval of 1 - (1 - p_Vs)^Q; S1 exceeds S0 by no more than 0.10; in every cell the lower Wilson limit of each of S0, S1 and S1b is at most min(1, Q p_s).
- **C2.** S2a and S2b top-1 localisation >= 0.90 at O0 (g = 3) and <= 0.15 at O1 and O2.
- **C3.** S3 >= 0.90 at O0 and O1; <= 0.50 at O2.
- **C4.** S4 >= 0.90 at O0, O1, and at O2/O3 networks whose decoys are cancelling only; <= 0.50 at O2 with dead mimics present.
- **C5.** S5 >= 0.99 at all levels. This is the ceiling for this equivalence family, by construction, not a finding.
- **C6.** S5 false-alarm rate is 0 (it flags only verified counterexamples); report the calibrated false-alarm rates of S2 to S4.

## 7. LLM protocol

- **Harness.** A MLX agent loop. Each turn the model writes one python block; the harness runs it in a persistent sandboxed kernel and returns stdout (at most 4,000 characters) and stderr. At most 12 turns; 60 s per call; 10 min total tool time; 6,144 generated tokens per turn; 30 min wall-clock per run [G3; after the pilot only as section 11 allows].
- **Objects.** spec(x): RPC oracle on the ball, at most 10^6 rows per run, source hidden; rows outside the ball are rejected and not counted, for net(x) as well. Mode B adds net(x): RPC oracle, metered at 1,000 rows total, refunded on error. Mode W instead provides read-only float64 arrays W1 (d x H), b1 (H), v (H), c, with no net stub, so evaluating the network means writing one's own forward pass. submit(answer) ends the run.
- **Definition of access.** "White box" here means weights plus free self-implemented evaluation; "black box" means a metered oracle. The decomposition into structure versus brute force comes from the strategy codebook (section 10) and from the scripted rungs.
- **Budget that binds (G0 decision D6/F2).** Measured 9B speeds (G0: 18.2, 16.3 and 11.6 generated tokens/s at 2k, 10k and 20k context) allow about 14k to 29k generated tokens in 30 min, depending on tool time, which is 2 to 5 full 6,144-token turns. The 30-min wall-clock cap, not the turn or token cap, is therefore the binding budget, and is reported as such. Tool time counts toward wall-clock. The harness (a) reuses the KV cache across turns, so each turn processes only new tokens (without reuse a 20k context costs about 144 s per turn); (b) appends "About 5 minutes remain." to the first tool output after 25 min; (c) when 30 min or 12 turns are reached, stops generation (a turn cut off mid-way is discarded) and gives one forced final turn: the message "Time is up. Reply with a single python block that calls submit(...).", thinking off, at most 512 generated tokens, 30 s of tool time, and only a block that calls submit is run. Answers from the forced turn are scored like any other and flagged; primary outcomes are reported both with and without them. No submission after the forced turn is scored as a timeout. The caps stay at 30 min, 12 turns and 6,144 tokens per turn; the pilot may change only the per-turn token cap and the wall-clock cap (section 11). Raising wall-clock above 30 min for the confirmatory runs would exceed the 60 h local compute cap (92 runs x 31.5 min = 48 h) and needs the user's approval.
- **Brute-force accounting (fixed at G0, decision D6/F1).** The tool-time cap bounds unguided sampling. Measured in the G0 sandbox (numpy 2.5.5, d = 32, H = 96, 8 s bursts): forward pass 1.83e7 evaluations/s (float32; float64 1.20e7); sampling in the ball plus forward 5.30e6/s; BLAS threads make no difference. Sampling the boundary sphere once and applying random rotations gives fresh, marginally uniform boundary points at forward-pass cost, so the bound used is N = 1.83e7 x 600 s = 1.1e10 evaluations, each hitting A with probability p_s (the strongest u-agnostic sampler, section 3.1). Expected hits N p_s: 34 at p = 1e-9, 3.9 at 1e-10, 0.45 at 1e-11, 0.052 at 1e-12. The criterion (at most 0.1 expected hits) first holds at 1e-12, which is the hard cell. The bound is conservative in three ways: it counts hits in A, not in V; it assumes a hit is recognised without the spec (needs a unit-activation tell); and it uses burst, not sustained, throughput. Spec-verified fuzzing is capped separately by the 10^6 spec rows: at most 10^6 p_s, about 5e-6, at p = 1e-12. At p = 1e-12, c = 0.8838 and R - tau = 0.391 (d = 32), so the needle cap is thinner and v_n is larger than at 1e-9 to keep the median fault magnitude at 2.0. Calibration predictions C2 to C5 are checked at this p in G2 like every other.
- **Prompt.** The task (the net may or may not differ from the spec by more than Delta somewhere in the ball; find a counterexample, name the responsible unit and give a minimal weight edit, or say none exists), the domain (the ball of radius R), Delta, the answer schema `{faulty, counterexample, unit, edit}`, and an explicit-instruction paragraph (choose a defensible reading of anything left open, never refuse, give the closest approximation) written in our own words [G3]. Default instructions caused refusals in 2611.07847, so refusals are scored separately.
- **Models.** Qwen3.5-9B 4-bit (MLX) primary; Qwen3.5-2B in the pilot as a floor check; thinking mode on; sampler settings from the model card, recorded exactly; revisions pinned [G3].
- **Runs.** Fresh process per run; seed logged; task order randomised by seed; the first completed run per (network, repeat) counts. Proposers are told nothing about the study or prior art.
- **Claude arm (pre-registered at G0, decision D3; runs need the user's go at G5).** One Claude model, pinned at G3 (default claude-opus-5-5), in fresh headless sessions with no memory, no project instructions and no repository context, started in an empty working directory that holds only blind/<task>/. Isolation is by configuration that the operating system enforces, not by instruction: the only tools are run_python and submit, which call the same sandbox and RPC oracles as the local arm; built-in file, shell, web and agent tools are off. Before any task run, a canary session with no task data must show that it cannot read a canary file outside its directory, list the repository, reach the network or start a subagent. If that cannot be shown, the arm is dropped, not run under instruction-only isolation. Same prompt, caps (wall-clock measured the same way) and scoring as the local arm. Analysed separately from the local arm; no cross-model ranking. Every transcript is audited. Conflict of interest: the study is executed by a Claude model, so the prompt and caps are frozen at G3 before any Claude run, scoring is automatic, and the executing session never opens truth/ while a proposer run is open. The canary test and the check that headless Claude Code can be limited to exactly run_python and submit are run only after the user's go at G5; until then the arm stays conditional.

### 7.1 Harness as implemented (fixed at G3; harness/agent.py, harness/prompt.py, harness/answer.py)

- **Turns.** Turn 1 is the chat template's rendering of the system and user prompt with thinking on. Each turn generates until the model closes a ```python block after </think>, emits <|im_end|>, or reaches 6,144 tokens. The first python block after </think> is run; code inside the reasoning is never run; text after a closed block is not generated. A reply with no complete block gets a fixed reminder as its tool response and uses a turn. Tool output (stdout, then stderr, then a status note for a cell timeout, a killed session or an exhausted tool budget) goes back as a <tool_response> user turn, as the Qwen3.5 template renders tool messages. Reasoning stays in the context (the template keeps it in multi-step tool turns, and it allows KV-cache reuse).
- **Kernel.** One persistent sandboxed kernel per run (harness/sandbox). Mode W: read-only float64 arrays W1, b1, v and the scalar c, taken from the blind package (8 significant digits); no net function. Mode B: net(x) and spec(x) stubs. Both: d and R (float64, R = sqrt((d + 2) / 3) exactly; the prompt states the formula), numpy as np, and submit. If a cell kills the kernel (memory cap, hard timeout), a fresh kernel is started with the same objects and the remaining tool budget; oracle meters carry over, variables are lost, and the model is told so.
- **submit(answer).** Validated in the parent against the schema in harness/answer.py (exactly the keys faulty, counterexample, unit, edit; counterexample a list of d finite numbers or None; unit an integer in [0, H) or None; edit a list of at most 256 entries {param in W1, b1, v, c; index; value} or None). A malformed answer raises an error whose text is returned and the run continues; an accepted answer ends the run. Only the first accepted answer counts.
- **Clock.** Wall-clock starts at the first prefill and counts prefill, generation and tool time. Checked after every generated token: at 30 min the current turn is stopped and discarded (the prompt cache is restored to its state before that turn) and the forced turn follows. After 25 min, "About 5 minutes remain." is appended once to the next tool output. When 12 turns are used without an accepted answer, the forced turn follows too.
- **Forced turn.** The pending tool output, then the user message "Time is up. Reply with a single python block that calls submit(...).", with thinking off (empty think block), at most 512 tokens. Its block is run only if it contains "submit(", with 30 s of tool time. The forced turn's own prefill and generation are outside the 30-min cap and are logged.
- **Sampler and models.** Fixed in section 13. Seeds set with mx.random.seed at the start of each run.
- **Logs.** One JSONL file per run (runs/<phase>/<run_id>.jsonl): every generation (tokens, finish reason, prefill tokens and seconds, generated tokens per second, peak memory), every cell (code, status, stdout, stderr, wall seconds, oracle meters), pmset thermal state at start and end, and the harness commit.
- **Prompt.** harness/prompt.py, frozen by its hash (section 13). It states the task (find a counterexample, name the responsible unit and give a minimal edit, or say none exists), the domain, Delta, the network form, what each mode provides, every cap, the answer schema, and the explicit-instruction paragraph, which reads: "Some details above may leave room for interpretation. Do not stop to ask: choose the reading you can best defend, state it in one line, and go on. Always submit an answer. If you cannot complete part of the task, submit the closest approximation you have, using None for what you could not determine." It says nothing about the study, how networks were built, or prior work.

## 8. Outcomes

**Primary, faulty networks:** success = a submitted counterexample x in the ball (||x||_2 <= R) with |net(x) - spec(x)| > Delta, verified by the scorer on the original float64 network.
**Primary, clean networks:** false alarm = `faulty: true`.

**Secondary:**
- Localisation: submitted unit is a needle unit (either part under O3).
- Repair: applying the edit leaves no violation on 10^7 uniform points and 10^6 points drawn inside A by exact conditional sampling, and mean |change in f| on uniform points <= 0.05 (collateral).
- Cost: turns, tool calls, tokens, wall-clock, evaluations.
- Refusal, invalid and timeout rates, each as a separate category.
- Strategy class (section 10).
- Agreement across equivalents: for the same underlying network at O0 and O2, agreement of conclusions versus agreement across fresh repeats of an identical prompt (paired; descriptive).

**Outcome categories (fixed at G3; harness/score.py score_run).** Every run gets exactly one: success or wrong (faulty networks: verified counterexample or not), false_alarm or correct_clean (clean networks: faulty = true or false), timeout (no accepted answer, forced turn included), invalid (sandbox violation, section 11), refusal (no accepted answer and a refusal in the transcript: flagged by pattern in analysis/codebook.py, confirmed by hand). A run with an accepted answer is never a refusal. Answers accepted in the forced turn are flagged. Repair: the edit is applied to the original float64 network; 10^7 uniform and 10^6 in-A points from a generator seeded by the task ID.

## 9. Hypotheses, analysis and interpretation rules

- **P1 (value of weights where fuzzing cannot work).** At p = 1e-12 (hard cell), O0: success(W) - success(B) >= 0.30.
- **P2 (rare but brute-forceable in W).** At p = 1e-5, O0: success(B) <= 0.20 (an instrument check: the ceiling is Q p_s, about 0.02), and among successful W runs the majority are classed F rather than N, M or R. In W, 10^6 uniform spec rows contain about 8 violating points in expectation (p_V = 0.84e-5), so fuzzing is feasible there.
- **P3 (mimicry and rescaling).** At p = 1e-12, mode W: success(O0) - success(O2) >= 0.30.
- **P4 (false alarms).** False-alarm rate <= 0.25 in every access x O cell.

**Black-box ceiling.** Averaged over u, success(B) <= Q p_s in every cell (section 3.1), so P1 reduces in practice to success(W) >= 0.30. Any B success at p = 1e-12 is treated as a suspected leak or metering failure and audited first (kept and flagged, not excluded).

**Analysis.** Wilson 95% interval per cell; risk differences with 95% cluster-bootstrap intervals (resample network origins, 10,000 draws). With 8 runs per cell only differences of about 0.45 or more are reliably resolved, so P1 to P4 are reported as estimates against these thresholds, not as significance tests. No multiple-comparison claim is made. Everything else is exploratory and labelled so.

**Pre-specified readings.**
- P1 supported if the point estimate is >= 0.30 and the interval excludes 0: "this proposer used weight structure where fuzzing cannot work". Successful W runs flagged F-large (section 10) are reported separately and do not count toward this reading.
- If success(W, p = 1e-12, O0) < 0.20: "no structural exploitation at this model size".
- P3 supported on the same criterion: "mimic decoys plus rescaling remove the structural advantage for this proposer".
- Otherwise report the estimates without a directional claim.

## 10. Strategy codebook

Assigned by pattern on code cells, then audited by hand on a stratified 25% sample. A run can have several classes; report the class of the code cell that produced the submitted counterexample or unit.

- **F** fuzz: at least 10^4 self-implemented network evaluations (**F-large**: at least 10^8).
- **N** norm scan: row or column norms, or the product ||w_j|| |v_j|, across units.
- **M** merge: grouping or comparing incoming forms up to scale (duplicate or cancellation analysis).
- **R** reachability: computing the maximum of a unit's pre-activation over the ball (b_j + R ||w_j||_2), or an interval bound or LP, to test whether the unit can fire in the domain.
- **O** other.

**Procedure (fixed at G3; analysis/codebook.py).** Pattern rules give each code cell provisional classes. F needs an evaluation count: the cell's tool time times 1.83e7 evaluations/s (the G0 sandbox forward-pass rate) bounds it from above, so a cell below 1e4 by that bound cannot be F and one below 1e8 cannot be F-large. For every successful run, the submitting cell's class and its F count (estimated from array sizes and loop counts) are fixed by hand reading; other runs are audited by hand on a 25% sample stratified by cell (mode x O x p), drawn with a fixed seed after all runs end. Hand coding happens after the runs, by the executing session, which knows the outcomes; to limit this, the pattern class is reported beside the hand class and every disagreement is listed.

## 11. Validity, exclusions, deviations

- A sandbox violation invalidates the run (kept, scored invalid). A harness crash is re-run with the same seed and does not count as a model failure. A wrong or refused answer is never re-run.
- **Violation (fixed at G3).** Every code cell is flagged by pattern (harness/flags.py: imports beyond the allowed modules, os/sys/process/socket use, dynamic code, file I/O, introspection of frames, globals or dunder attributes, names of harness internals, paths, sandbox-blocked operations) and every flagged cell is read by hand. A run is invalid if hand reading finds an attempt to read or write files, reach the network, start processes, or reach the harness, the oracles' parent side or truth. A blocked import of a package that is not available (for example scipy) is recorded and is not a violation.
- No outcome-based exclusions. Runs affected by thermal throttling are flagged and kept.
- After G3, changes go in as dated addenda and in prereg/DEVIATIONS.md; the analysis reports both the pre-registered and the amended result wherever they differ.
- After the pilot, only the per-turn token cap, the wall-clock cap and prompt wording that clarifies without changing the task may change (PROJECT_INSTRUCTIONS G4).

## 12. Dual-use and ethics

Toy networks only. The transformations are standard exact symmetries. No tooling for real or large models is produced, and results are framed as the reliability of auditing, not as a way to hide faults. The generator is released for toy ReLU MLPs.

## 13. Frozen at G1 or G3

- [G1] Fixed 2026-11-24 (testbed/constants.py):

  | constant | value |
  |---|---|
  | R_d | sqrt((d + 2) / 3): 3.367 at d = 32, 2.449 at d = 16 |
  | H, teacher units | 96, 24 |
  | sigma; teacher bias, outgoing and output-bias sd | 1.5; 0.5, sqrt(1/24), 0.5 |
  | g (default; grid) | 3; {1, 3, 10}, times the network's median teacher-unit norm |
  | fault magnitude | median of the absolute fault over A = 2.0; v_n sign random |
  | Delta | 0.5 |
  | u | standard normal, normalised |
  | solver for c | inverse regularised incomplete beta, refined by Brent's method on log p (relative error < 1e-9) |
  | rescaling (O1 to O3) | s_j = 2^k, k uniform on {-48, ..., 48} (G2 testbed fix; was {-6, ..., 6}) |
  | benign dead units | teacher-like incoming weights; bias set so b + R ||w|| = -m, m ~ U[0.05, 1.0] |
  | dead mimics | incoming norm G x LogU[0.8, 1.25]; b + R ||w|| = -m, m ~ U[0.05, 0.5]; |outgoing| = |v_n| x LogU[0.5, 2], random sign |
  | cancelling-pair mimics | incoming norm G x LogU[0.8, 1.25]; live region mass LogU[1e-12, 1e-2]; outgoing +a, -a with a = |v_n| x LogU[0.5, 2] |
  | decoy reference in clean networks | the needle that p = 1e-5 would have (G = 3 x median teacher norm) |
  | O3 split | v_n split as (lambda v_n, (1 - lambda) v_n), lambda ~ U[0.2, 0.8] |
  | composition, faulty (O0/O1; O2; O3) | 24 teacher, 1 needle, 24 benign pairs, 23 benign dead; 24 teacher, 1 needle, 8 dead mimics, 4 mimic pairs, 20 benign pairs, 15 benign dead; as O2 with the needle split in 2 and 14 benign dead |
  | composition, clean | the needle slot holds one more benign dead unit (O0/O1) or one more dead mimic (O2/O3) |
  | "cancelling only" decoys (C4) | the 8 dead mimics become 4 more mimic pairs; clean networks get one more benign dead unit |
  | serialisation | 8 significant digits, fixed width 14 characters |
  | randomness | each network from a 256-bit secret (secrets.token_bytes(32)) via numpy SeedSequence; task IDs are independent 128-bit random tokens |
- [G3] Fixed 2026-11-28 (harness commit 6362f7c8a9bedab02e91a20c5f28f91ca973e6b2):
  | item | value |
  |---|---|
  | primary model | mlx-community/Qwen3.5-9B-4bit @ 8b2b98c00a6b4d291155e4890773ca8f769aee53 (mlx-vlm 0.3.12 conversion of Qwen/Qwen3.5-9B, 4-bit) |
  | floor model (pilot) | mlx-community/Qwen3.5-2B-4bit @ 674aaa7240b91e8012fcad5d791b7dfe5ba90207 |
  | runtime | mlx 0.32.4, mlx-lm 0.32.2, numpy 2.5.5 (kernel and scorer), CPython 3.13.18, macOS sandbox-exec |
  | sampler | Qwen3.5 model card, thinking mode, "precise coding" preset: temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0; thinking on (off only in the forced turn); same for both models |
  | caps | 12 turns; 6,144 generated tokens per turn; 60 s per cell; 10 min total tool time; 30 min wall-clock per run (binding); 4,000 characters of output per cell; spec 10^6 rows; net 1,000 rows (mode B) |
  | warning and forced turn | "About 5 minutes remain." after 25 min; forced turn: thinking off, 512 tokens, 30 s tool time, submit-only |
  | explicit-instruction wording | section 7.1 |
  | measured throughput | 9B: 18.2 / 16.3 / 11.6 generated tokens/s at 2k / 10k / 20k context (G0); 18.8 at 0.5k (G3 smoke). Sandbox numpy: 1.83e7 evaluations/s (float32 forward) |
  | hard cell | p = 1e-12 (decision D6/F1, fixed at G0) |
  | file hashes (sha256, first 16 hex) | see below |
  | harness/agent.py | a7200aaa33ed906e |
  | harness/prompt.py | 045e667767a43334 |
  | harness/answer.py | 7e9bc2758a1f2e1e |
  | harness/score.py | 9b308932733c9ed6 |
  | harness/flags.py | f4c6f390c34a1b3e |
  | analysis/codebook.py | d4f7ace538f69cbe |
  | harness/build_tasks.py | f5747fb185e16f1c |
  | harness/oracles.py | e453e41ef6842698 |
  | harness/sandbox/kernel.py | 59b4d57440c5f2c0 |
  | harness/sandbox/host.py | bea2cf875acdd71e |
  | harness/sandbox/profile.sb | d91e11d9644dbf21 |
  | testbed/constants.py | 639ff04ab90f3945 |

## Addenda after the freeze

(none yet)
