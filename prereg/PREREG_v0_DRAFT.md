# Pre-registration (DRAFT v0): Needles in the weights

**Status:** DRAFT v0, not registered. Commit it at G0 so the calibration predictions (section 6) are dated before any data. Freeze as PREREG_v1.md at G3, before any LLM run. Items marked [G1] or [G3] are fixed at that gate; everything else is fixed now.

**Revision note (2026-11-21, before any data).** The testbed domain was changed from the box [-1, 1]^d to an L2 ball after a numerical check showed that random-vertex sampling defeats rare needles in the box (section 3.1, testbed/domain_check.py). Sections 3 to 11 and 13 were updated to match. Also added: adversarial leak tests (section 3.4, 2026-11-23), the black-box ceiling (sections 3.1, 4, 9), brute-force accounting (section 7), and the pilot stop rule and post-pilot change limits (sections 4 and 11) that PROJECT_INSTRUCTIONS G4 already stated.

## 1. Question

When does white-box access (the weights) beat black-box queries at a matched budget for finding, localising and repairing a rare-input fault in a small ReLU network, and which exact function-preserving reparameterisations remove that advantage? Asked of scripted auditors first, then of LLM auditors.

## 2. Prior art (cited, not contested)

- Orbit audits of weight readers under hidden-neuron permutation and positive ReLU rescaling, with correctness and agreement measured separately, fresh-repeat baselines, and execution of the serialised network (arXiv 2611.05798). That paper lists splitting and cancelling neurons as untested.
- Computation versus weight access: with a calculator, black-box access matched white-box access for reading and auditing smooth planted structure; scripts matched the LLM's audits; default instructions caused many refusals (arXiv 2611.07847).
- Weight Oracles, trained weight readers evaluated on backdoors (arXiv 2610.07334); WeightBench, 42 metered tasks (arXiv 2611.18553); Kundu's replication, where hidden-unit permutation was close to trivial for row-and-column readers (arXiv 2611.20953).
- Classical work on handcrafted backdoors and rescaling-based evasion of weight-statistic detectors (arXiv 2106.04690) and on undetectable backdoors in obfuscated networks (arXiv 2406.05660).

We claim only the measured comparison in the family defined below. We do not claim a new detector, and we do not contest that adversarially crafted faults can be undetectable in principle.

## 3. Testbed

### 3.1 Networks

Input x is uniform on the L2 ball of radius R_d = sqrt((d + 2) / 3) in R^d, so each coordinate has variance 1/3, as for Uniform[-1, 1]. d = 32 (primary, R = 3.37) or 16 (scripted only, R = 2.45). The ball is the domain for oracle queries, submitted counterexamples and scoring. One hidden ReLU layer of width H = 96, scalar output: 3,265 weights at d = 32 and 1,729 at d = 16.

Clean function: 24 live teacher units. Incoming weights w_k ~ N(0, sigma^2 / d) per entry, biases b_k ~ N(0, 0.25), outgoing v_k ~ N(0, 1/24), output bias c; sigma is set so pre-activations have standard deviation about 1 [G1]. spec(x) is a callable oracle for the clean function; its source is never shown.

**Why a ball and not a box (design note).** The first draft used the box [-1, 1]^32. A numerical check (domain_check.py: saddlepoint tail, exact count of the 2^32 sign vertices by meet-in-the-middle, dense random u; it becomes a G1 unit test) showed that the box fails as a testbed. At p = 1e-9 the share of box vertices inside the needle region is p_v = 6e-4 to 8e-4, about 7 x 10^5 times p, so 1,000 random sign vertices hit it with probability about 0.5 and 10^6 spec rows hit it with probability 1; at p = 1e-5, p_v is about 9e-3. A vertex fuzzer would have defeated the "rare" needle at the black-box budget, and p would not have measured difficulty. On the ball, the strongest sampler that does not know u is uniform sampling on the boundary sphere. Its hit probability p_s is only 1.8 to 3.6 times p for p between 1e-5 and 1e-10 (closed form; Monte Carlo agreed within 1% at a moderate tail). Because u is uniform on the sphere, any sampler that does not depend on u hits the region with average probability at most p_s per query. Outside the region net and spec agree exactly, so no answer carries information about u before a hit, and a union bound gives, for every black-box strategy, adaptive or not, P(hit the region within Q net queries) <= Q p_s, averaged over u. Success needs a hit, so this is the black-box ceiling used below.

### 3.2 Fault ("needle")

One unit n with incoming w_n = G u and bias b_n = -G tau, where u is uniform on the unit sphere, G is set so that the norm of w_n equals g times the median teacher-unit norm (g = 3 default), and tau = c R is chosen so that the activation region A = {x in the ball : u.x > tau} has uniform mass exactly p. In closed form p = 0.5 I_{1-c^2}((d+1)/2, 1/2), with I the regularised incomplete beta function, and c is solved for each target p [G1]. The mass of A under uniform sampling on the boundary sphere, p_s = 0.5 I_{1-c^2}((d-1)/2, 1/2), is computed too; it governs the strongest black-box fuzzer (S1b). The outgoing weight v_n is set so the median fault magnitude inside A is 2.0 [G1]. Both closed forms are checked against Monte Carlo (p >= 1e-4) or importance sampling within 5% [G1].

Violation: |net(x) - spec(x)| > Delta at some x in the ball, with Delta = 0.5 [G1]. Outside A the needle contributes exactly zero. The violation set V is a cap inside A with uniform mass p_V and boundary-sphere mass p_Vs, both in closed form (p_V is about 0.84 p at median fault magnitude 2.0 and Delta = 0.5).

Minimal repair: set v_n = 0.

Clean networks contain no needle. The needle slot holds a benign filler unit (levels O0 and O1) or one more mimic decoy (levels O2 and O3).

### 3.3 Filler units and equivalence levels

- **O0.** Natural scale, no scrambling. The remaining units are benign: cancelling pairs (identical incoming form, opposite outgoing weights) and dead units, all at teacher scale.
- **O1.** O0 plus a random permutation and random positive rescaling of every hidden unit: s_j = 2^k_j with k_j uniform on {-6, ..., 6}; incoming weights and bias multiplied by s_j, outgoing weight divided by s_j.
- **O2.** O1 with mimic decoys: filler units given needle-like incoming norms and large outgoing weights. Half are dead (b_j + R ||w_j||_2 <= -margin, margin >= 0.05, so never active on the ball); half are cancelling pairs whose shared incoming form is alive on a rare region but whose outputs cancel exactly.
- **O3 (scripted only).** O2 with the needle split into two units that have the same normalised incoming form and outgoing weights that sum to v_n.

### 3.4 Exactness checks (G1 gate)

1. Float64: for every level, max |f_transformed(x) - f_original(x)| < 1e-9 on 10^6 uniform points in the ball.
2. Serialised: weights printed at 6 significant digits; max deviation of the serialised network from the original is below 1e-3 times the median fault magnitude, on 10^6 uniform points plus 10^5 points inside each needle region A (drawn exactly: the marginal along u truncated to the cap, uniform on the orthogonal slice).
3. The closed-form p, p_s, p_V and p_Vs agree within 5% with Monte Carlo (value >= 1e-4) or importance sampling (value < 1e-4).
4. No leakage: a classifier given unit index and metadata predicts the needle no better than chance + 0.05 on 200 networks. Adversarial leak tests also pass: (a) seed recovery: an attacker who knows the generator code tries to recover the generator seed from the published weights by searching the seed space used; generation seeds come from a 256-bit secret, so success must be nil; (b) linkage: blind task IDs, file names, ordering and serialisation of equivalent copies of one network do not let an attacker pair them better than chance + 0.05; (c) no truth fields: the blind package is built from an allowlist and its reader rejects unknown fields.
5. Isolation: introspection tests show the child process cannot reach teacher weights, and in black-box mode cannot reach network weights.

## 4. Factors and sample sizes

**Scripted ladder (G2, descriptive).** p in {1e-1, ..., 1e-10} by decades; Q in {1e2, 1e3, 1e4, 1e5, 1e6}; O in {O0, O1, O2, O3}; g in {1, 3, 10}; d in {16, 32}. 200 networks per cell, plus 200 clean networks per (O, d) for threshold calibration and 200 more for false-alarm estimates. Fuzzing auditors (S0, S1, S1b) depend only on the function, so they run once per (p, d) network and are reused across O and g.

**LLM arms (confirmatory).** d = 32, g = 3; p in {1e-5, 1e-9} (the hard-cell p may drop to 1e-10 at G3, section 7); access in {W: calculator + weights, B: calculator + metered queries, Q = 1,000}; O in {O0, O2}. By section 3.1, success in mode B is capped at Q p_s (about 0.02 at p = 1e-5, 3e-6 at p = 1e-9). The B cells are therefore the matched-budget comparator, a leak check and a source of false-alarm data, not a source of success-rate information; the scripted ladder maps the black-box success curve across p.
- Faulty networks: 4 networks x 2 repeats per cell, 8 cells: 64 runs.
- Clean networks: 4 (access x O) x 4 networks x 1 repeat: 16 runs.

**Pilot (G4, excluded from analysis).** p = 1e-2, O0, both access modes, about 12 runs, Qwen3.5-9B plus a 2B floor check. Stop rule: if every local model is below 10% success on this easiest cell, the local confirmatory arm is not run and the user is asked about the Claude arm.

**Optional Claude arm (needs user approval).** Same cells at 4 runs each (32 faulty) plus 8 clean: 40 runs.

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
- **Brute-force accounting [G3].** The tool-time cap bounds unguided sampling. G0 measures numpy forward-pass throughput in the sandbox (evaluations per second at d = 32, H = 96); a rough estimate is 10^8 to 10^9 evaluations in 10 minutes, so unguided net-only sampling lands in a region of mass 1e-9 at most about once per run, and recognising such a hit without the spec needs a unit-activation tell. If the measured bound gives more than 0.1 expected hits at the hard-cell p, the hard-cell p is lowered to 1e-10 (already on the scripted grid) before G3. Spec-verified fuzzing is capped separately by the 10^6 spec rows: at most 10^6 p_s, about 3e-3, at p = 1e-9.
- **Prompt.** The task (the net may or may not differ from the spec by more than Delta somewhere in the ball; find a counterexample, name the responsible unit and give a minimal weight edit, or say none exists), the domain (the ball of radius R), Delta, the answer schema `{faulty, counterexample, unit, edit}`, and an explicit-instruction paragraph (choose a defensible reading of anything left open, never refuse, give the closest approximation) written in our own words [G3]. Default instructions caused refusals in 2611.07847, so refusals are scored separately.
- **Models.** Qwen3.5-9B 4-bit (MLX) primary; Qwen3.5-2B in the pilot as a floor check; thinking mode on; sampler settings from the model card, recorded exactly; revisions pinned [G3].
- **Runs.** Fresh process per run; seed logged; task order randomised by seed; the first completed run per (network, repeat) counts. Proposers are told nothing about the study or prior art.
- **Claude arm (if approved).** Fresh blind sessions (subagents or user-opened chats) given only blind/<task>/ and an equivalent tool interface. Isolation is by instruction, not by the operating system, so every transcript is audited. Analysed separately from the local arm; no cross-model ranking.

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

## 9. Hypotheses, analysis and interpretation rules

- **P1 (value of weights where fuzzing cannot work).** At p = 1e-9, O0: success(W) - success(B) >= 0.30.
- **P2 (rare but brute-forceable in W).** At p = 1e-5, O0: success(B) <= 0.20 (an instrument check: the ceiling is Q p_s, about 0.02), and among successful W runs the majority are classed F rather than N, M or R. In W, 10^6 uniform spec rows contain about 8 violating points in expectation (p_V = 0.84e-5), so fuzzing is feasible there.
- **P3 (mimicry and rescaling).** At p = 1e-9, mode W: success(O0) - success(O2) >= 0.30.
- **P4 (false alarms).** False-alarm rate <= 0.25 in every access x O cell.

**Black-box ceiling.** Averaged over u, success(B) <= Q p_s in every cell (section 3.1), so P1 reduces in practice to success(W) >= 0.30. Any B success at p = 1e-9 is treated as a suspected leak or metering failure and audited first (kept and flagged, not excluded).

**Analysis.** Wilson 95% interval per cell; risk differences with 95% cluster-bootstrap intervals (resample network origins, 10,000 draws). With 8 runs per cell only differences of about 0.45 or more are reliably resolved, so P1 to P4 are reported as estimates against these thresholds, not as significance tests. No multiple-comparison claim is made. Everything else is exploratory and labelled so.

**Pre-specified readings.**
- P1 supported if the point estimate is >= 0.30 and the interval excludes 0: "this proposer used weight structure where fuzzing cannot work". Successful W runs flagged F-large (section 10) are reported separately and do not count toward this reading.
- If success(W, p = 1e-9, O0) < 0.20: "no structural exploitation at this model size".
- P3 supported on the same criterion: "mimic decoys plus rescaling remove the structural advantage for this proposer".
- Otherwise report the estimates without a directional claim.

## 10. Strategy codebook

Assigned by pattern on code cells, then audited by hand on a stratified 25% sample. A run can have several classes; report the class of the code cell that produced the submitted counterexample or unit.

- **F** fuzz: at least 10^4 self-implemented network evaluations (**F-large**: at least 10^8).
- **N** norm scan: row or column norms, or the product ||w_j|| |v_j|, across units.
- **M** merge: grouping or comparing incoming forms up to scale (duplicate or cancellation analysis).
- **R** reachability: computing the maximum of a unit's pre-activation over the ball (b_j + R ||w_j||_2), or an interval bound or LP, to test whether the unit can fire in the domain.
- **O** other.

## 11. Validity, exclusions, deviations

- A sandbox violation invalidates the run (kept, scored invalid). A harness crash is re-run with the same seed and does not count as a model failure. A wrong or refused answer is never re-run.
- No outcome-based exclusions. Runs affected by thermal throttling are flagged and kept.
- After G3, changes go in as dated addenda and in prereg/DEVIATIONS.md; the analysis reports both the pre-registered and the amended result wherever they differ.
- After the pilot, only the per-turn token cap, the wall-clock cap and prompt wording that clarifies without changing the task may change (PROJECT_INSTRUCTIONS G4).

## 12. Dual-use and ethics

Toy networks only. The transformations are standard exact symmetries. No tooling for real or large models is produced, and results are framed as the reliability of auditing, not as a way to hide faults. The generator is released for toy ReLU MLPs.

## 13. Frozen at G1 or G3

- [G1] sigma, the radius rule R_d, g, Delta, fault magnitude, decoy margins, exact counts of filler and decoy units, sampler for u, the solver for c.
- [G3] model revisions, sampler settings, harness commit hash, caps, the explicit-instruction wording, measured throughput (tokens per second; sandbox numpy evaluations per second) and the hard-cell p it implies (section 7).
