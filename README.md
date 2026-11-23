# Needles in the weights

A small, pre-registered study: when does access to a network's weights help an auditor find, localise and repair a rare-input fault, compared with black-box queries at a matched budget, and which exact function-preserving reparameterisations remove that help? Scripted auditors first, then local open-weight LLM auditors.

Scope is toy one-hidden-layer ReLU networks built by our own generator. No claims about real or large models.

- `PROJECT_INSTRUCTIONS.md`: rules, gates G0 to G5 and session protocol.
- `prereg/PREREG_v0_DRAFT.md`: pre-registration draft (frozen as v1 at G3).
- `testbed/domain_check.py`: the numerical check behind the ball domain (`python3 testbed/domain_check.py`, needs numpy and scipy).
- `STATUS.md`: living log.

Status: G0 (preflight) not started. No data collected.
