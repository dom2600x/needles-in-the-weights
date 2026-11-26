# Needles in the weights

A small, pre-registered study: when does access to a network's weights help an auditor find, localise and repair a rare-input fault, compared with black-box queries at a matched budget, and which exact function-preserving reparameterisations remove that help? Scripted auditors first, then local open-weight LLM auditors.

Scope is toy one-hidden-layer ReLU networks built by our own generator. No claims about real or large models.

- `PROJECT_INSTRUCTIONS.md`: rules, gates G0 to G5 and session protocol.
- `prereg/PREREG_v0_DRAFT.md`: pre-registration draft (frozen as v1 at G3).
- `testbed/`: generator, geometry, serialiser, verifier (`.venv/bin/python -m testbed.verify`) and tests; `testbed/domain_check.py` is the numerical check behind the ball domain.
- `harness/`: proposer sandbox (`harness/sandbox/`), metered oracles, G0 benchmarks.
- Tests: `.venv/bin/python -m pytest`.
- `STATUS.md`: living log.

Status: G1 (testbed) complete; see STATUS.md. No study data collected.
