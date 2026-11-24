# Make the repository root importable for tests (testbed, harness).
import os, sys

ROOT = os.path.dirname(os.path.realpath(__file__))
for p in (ROOT, os.path.join(ROOT, "harness", "sandbox")):
    if p not in sys.path:
        sys.path.insert(0, p)
