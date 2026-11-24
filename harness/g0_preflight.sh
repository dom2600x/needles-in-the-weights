#!/bin/bash
# G0 preflight, read-only: installs nothing, downloads nothing, writes only to stdout.
# Run on the Mac in Terminal:  bash harness/g0_preflight.sh | tee g0_preflight.txt
echo "== date (UTC)";      date -u +%Y-%m-%dT%H:%M:%SZ
echo "== macOS";           sw_vers 2>/dev/null
echo "== hardware";        sysctl -n hw.model machdep.cpu.brand_string hw.memsize hw.ncpu 2>/dev/null
echo "== disk free (home)"; df -h "$HOME" | tail -1
echo "== thermal";         pmset -g therm 2>/dev/null | sed -n '1,6p'
echo "== tools";           for t in python3 python3.11 python3.12 python3.13 uv pip3 git sandbox-exec brew; do printf '%-14s %s\n' "$t" "$(command -v $t || echo MISSING)"; done
echo "== python versions"; for p in python3 python3.11 python3.12 python3.13; do command -v $p >/dev/null && printf '%-12s %s %s\n' "$p" "$($p -c 'import sys,platform;print(sys.version.split()[0], platform.machine())')"; done
echo "== packages (python3)"
python3 - <<'PY'
import importlib.metadata as m
for p in ["numpy","scipy","pytest","mlx","mlx-lm","mlx-metal","transformers","huggingface-hub"]:
    try: print(f"{p:16s} {m.version(p)}")
    except m.PackageNotFoundError: print(f"{p:16s} MISSING")
PY
echo "== network (HEAD requests, 8 s timeout)"
for u in https://pypi.org/simple/ https://files.pythonhosted.org https://huggingface.co https://github.com; do
  printf '%-34s %s\n' "$u" "$(curl -sI -m 8 -o /dev/null -w '%{http_code}' "$u" || echo FAIL)"; done
echo "== HF cache";        ls -d "$HOME/.cache/huggingface/hub/"models--* 2>/dev/null || echo "none"
echo "== done"
