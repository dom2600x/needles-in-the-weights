"""PREREG 3.1: testbed/domain_check.py (box versus ball) as a unit test. Runs the unmodified script."""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.realpath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "domain_check.py")


def test_domain_check():
    out = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, timeout=600, check=True).stdout
    box, ball = out.split("=== BALL")
    # Box: the saddlepoint agrees with Monte Carlo, and vertex sampling overstates rarity at p = 1e-9.
    mc, sp = map(float, re.search(r"MC p=([\d.e+-]+)\s+saddle p=([\d.e+-]+)", box).groups())
    assert abs(sp / mc - 1) < 0.05
    ratios = [float(m) for m in re.findall(r"^\s*1e-09\s+\S+\s+\S+\s+\S+\s+(\S+)", box, re.M)]
    assert len(ratios) == 3 and min(ratios) > 1e5
    # Ball: closed forms agree with Monte Carlo, and the sphere sampler gains only a small factor.
    m = re.search(r"MC ball p=([\d.e+-]+)\s+closed form=([\d.e+-]+);\s+MC sphere p=([\d.e+-]+)\s+closed form=([\d.e+-]+)", ball)
    bmc, bcf, smc, scf = map(float, m.groups())
    assert abs(bmc / bcf - 1) < 0.05 and abs(smc / scf - 1) < 0.05
    gains = [float(x) for x in re.findall(r"^\s*1e-\d+\s+\S+\s+\S+\s+(\S+)", ball, re.M)]
    assert len(gains) == 4 and all(1.5 < x < 4 for x in gains)
