"""G0 sandbox escape and control tests.  Run:  .venv/bin/python -m pytest harness/sandbox/test_sandbox.py -v

Every escape test runs twice: with the audit hook (full sandbox) and without it (OS layer
alone), so each attack must be stopped by sandbox-exec by itself. A cell prints ESCAPED only
if the attack worked.
"""
import os, secrets, socket, sys, threading, time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from host import REPO, SandboxKernel  # noqa: E402

HOME = os.path.realpath(os.path.expanduser("~"))
SCRATCH_ROOT = os.environ.get("SANDBOX_TEST_SCRATCH") or os.path.join(REPO, "runs", "g0", "sandbox_scratch")


@pytest.fixture(params=[True, False], ids=["hook+os", "os-only"])
def kernel(request, tmp_path_factory):
    scratch = os.path.join(SCRATCH_ROOT, secrets.token_hex(8))
    k = SandboxKernel(scratch, oracles={"spec": lambda x: np.tanh(x).sum(axis=1)},
                      audit_hook=request.param, cell_timeout=10, rss_cap_gb=1.0)
    k.start()
    yield k
    k.close()


def run(k, code):
    r = k.exec(code)
    assert "ESCAPED" not in (r["stdout"] or ""), r
    return r


def attempt(body):
    """Wrap an attack so it prints ESCAPED on success and BLOCKED with the error otherwise."""
    lines = ["try:"] + ["    " + l for l in body.strip().splitlines()] + [
        "    print('ESCAPED')", "except BaseException as e:", "    print('BLOCKED', type(e).__name__, e)"]
    return "\n".join(lines)


# ---------------- file system ----------------
READ_TARGETS = [
    f"open({os.path.join(REPO, 'STATUS.md')!r}).read()",
    f"open({os.path.join(REPO, 'PROJECT_INSTRUCTIONS.md')!r}, 'rb').read()",
    f"open({os.path.join(HOME, '.zshrc')!r}).read()",
    f"__import__('os').listdir({HOME!r})",
    f"__import__('os').listdir({REPO!r})",
    f"__import__('os').listdir({os.path.join(HOME, '.ssh')!r})",
    "__import__('os').listdir('/Users')",
    f"__import__('os').stat({os.path.join(REPO, 'STATUS.md')!r})",
    "open('/etc/passwd').read()",
    f"__import__('os').open({os.path.join(REPO, 'README.md')!r}, 0)",
    f"__import__('numpy').loadtxt({os.path.join(REPO, 'README.md')!r}, dtype=str, delimiter='\\x00')",
]


@pytest.mark.parametrize("expr", READ_TARGETS)
def test_cannot_read_outside_scratch(kernel, expr):
    r = run(kernel, attempt(f"_ = {expr}"))
    assert "BLOCKED" in r["stdout"], r


WRITE_TARGETS = [os.path.join(HOME, "sbx_escape_probe.txt"), os.path.join(REPO, "sbx_escape_probe.txt"),
                 "/tmp/sbx_escape_probe.txt", os.path.join(REPO, ".venv", "sbx_escape_probe.txt"),
                 os.path.join(REPO, "harness", "sandbox", "kernel.py")]


@pytest.mark.parametrize("path", WRITE_TARGETS)
def test_cannot_write_outside_scratch(kernel, path):
    before = os.path.getmtime(path) if os.path.exists(path) else None
    r = run(kernel, attempt(f"open({path!r}, 'a').write('x')"))
    assert "BLOCKED" in r["stdout"], r
    after = os.path.getmtime(path) if os.path.exists(path) else None
    assert before == after, f"{path} changed"


def test_cannot_escape_scratch_via_symlink(kernel):
    r = run(kernel, attempt(f"import os\nos.symlink({os.path.join(REPO, 'STATUS.md')!r}, 'link')\nopen('link').read()"))
    assert "BLOCKED" in r["stdout"], r


# ---------------- network ----------------
@pytest.fixture
def local_listener():
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1); srv.settimeout(15)
    hits = []
    def accept():
        try:
            c, _ = srv.accept(); hits.append(1); c.close()
        except OSError:
            pass
    t = threading.Thread(target=accept, daemon=True); t.start()
    yield srv.getsockname()[1], hits
    srv.close()


NET_ATTACKS = [
    "import socket\ns = socket.create_connection(('1.1.1.1', 443), timeout=3)",
    "import socket\nsocket.getaddrinfo('github.com', 443)",
    "import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\ns.sendto(b'x', ('8.8.8.8', 53))",
    "import socket\ns = socket.socket(socket.AF_UNIX)\ns.connect('/var/run/mDNSResponder')",
    "import socket\ns = socket.socket()\ns.bind(('127.0.0.1', 0))\ns.listen(1)",
    "import urllib.request\nurllib.request.urlopen('https://huggingface.co', timeout=3)",
    "import urllib.request\nurllib.request.urlopen('http://1.1.1.1', timeout=3)",
    "import socket\ns = socket.socket(socket.AF_INET6)\ns.connect(('2606:4700:4700::1111', 443))",
]


@pytest.mark.parametrize("code", NET_ATTACKS)
def test_no_network(kernel, code):
    r = run(kernel, attempt(code))
    assert "BLOCKED" in r["stdout"], r


def test_no_loopback_to_host(kernel, local_listener):
    port, hits = local_listener
    r = run(kernel, attempt(f"import socket\nsocket.create_connection(('127.0.0.1', {port}), timeout=3)"))
    assert "BLOCKED" in r["stdout"] and not hits, r


# ---------------- processes and native code ----------------
PROC_ATTACKS = [
    "import os\nassert os.system('echo hi > /dev/null') == 0",
    "import subprocess\nsubprocess.run(['/bin/echo', 'hi'], check=True)",
    "import os\npid = os.fork()\nif pid == 0: os._exit(0)\nos.waitpid(pid, 0)",
    "import os\nos.posix_spawn('/bin/echo', ['/bin/echo', 'hi'], {})",
    "import os\nos.execv('/bin/sh', ['/bin/sh', '-c', 'true'])",
    "import ctypes\nlibc = ctypes.CDLL(None)\nfd = libc.open(%r.encode(), 0)\nassert fd >= 0" % os.path.join(REPO, "STATUS.md"),
    "import multiprocessing\np = multiprocessing.Process(target=print)\np.start(); p.join()",
]


@pytest.mark.parametrize("code", PROC_ATTACKS)
def test_no_processes_or_native_escape(kernel, code):
    r = run(kernel, attempt(code))
    assert "BLOCKED" in r["stdout"] or r["status"] == "killed", r
    if r["status"] != "killed":
        assert run(kernel, "print('alive')")["stdout"] == "alive\n"


# ---------------- process hygiene ----------------
def test_clean_environment(kernel):
    r = run(kernel, "import os\nprint(sorted(os.environ))")
    keys = eval(r["stdout"])  # noqa: S307  (our own test output)
    assert set(keys) <= {"HOME", "TMPDIR", "PATH", "LANG", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                         "__CF_USER_TEXT_ENCODING", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"}, keys


def test_no_inherited_file_descriptors(kernel):
    r = run(kernel, "import os\nok = []\nfor fd in range(3, 256):\n    try:\n        os.fstat(fd); ok.append(fd)\n"
                    "    except OSError:\n        pass\nprint(ok)")
    fds = eval(r["stdout"].strip())  # noqa: S307  (our own test output)
    assert len(fds) <= 2, fds  # only the two protocol pipes


# ---------------- resource caps ----------------
def test_soft_timeout_keeps_state(kernel):
    run(kernel, "keep = 7")
    r = run(kernel, "while True: pass")
    assert r["status"] == "timeout"
    assert run(kernel, "print(keep)")["stdout"] == "7\n"


def test_hard_timeout_when_alarm_ignored(kernel):
    t0 = time.monotonic()
    r = run(kernel, "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True: pass")
    assert r["status"] == "killed" and time.monotonic() - t0 < 25, r


def test_timeout_not_swallowed_by_except(kernel):
    r = run(kernel, "while True:\n    try:\n        while True: pass\n    except BaseException:\n        pass")
    assert r["status"] == "killed", r


def test_memory_cap(kernel):
    r = run(kernel, "a = np.ones(int(3e9) // 8)\nimport time\ntime.sleep(5)")
    assert r["status"] in ("killed", "error"), r


def test_output_cap(kernel):
    r = run(kernel, "print('x' * 100000)")
    assert len(r["stdout"]) < 4200 and "truncated" in r["stdout"]


def test_file_size_cap(kernel):
    r = run(kernel, attempt("open('big.bin', 'wb').write(b'\\0' * (100 << 20))"))
    assert "BLOCKED" in r["stdout"], r


def test_total_tool_budget():
    k = SandboxKernel(os.path.join(SCRATCH_ROOT, f"budget_{time.time_ns()}"), cell_timeout=2, tool_budget=3)
    k.start()
    try:
        assert k.exec("while True: pass")["status"] == "timeout"
        r = k.exec("while True: pass")
        assert r["status"] == "timeout" and r["wall_s"] < 1.6, r
        assert k.exec("print(1)")["status"] == "budget_exhausted"
    finally:
        k.close()


# ---------------- controls: legitimate work must succeed ----------------
def test_control_numpy_and_scratch(kernel):
    r = run(kernel, "x = np.random.default_rng(0).normal(size=(64, 32))\n"
                    "np.save('x.npy', x)\nprint(np.allclose(np.load('x.npy'), x), np.linalg.svd(x)[1].shape)\n"
                    "import numpy.fft\nprint(np.fft.fft(np.ones(4)).real.tolist())")
    assert r["status"] == "ok" and r["stdout"].startswith("True (32,)"), r


def test_control_spec_rpc(kernel):
    r = run(kernel, "x = np.zeros((5, 32)); x[0, 0] = 1\ny = spec(x)\nprint(y.shape, round(float(y[0]), 6))")
    assert r["stdout"] == "(5,) 0.761594\n", r


def test_control_oracle_error_is_reported(kernel):
    # A malformed call (1-D input to an oracle that sums over axis 1) must raise inside the cell, not crash the host.
    r = run(kernel, "try:\n    spec(np.zeros(3))\nexcept RuntimeError as e:\n    print('raised', e)")
    assert r["stdout"].startswith("raised") and "AxisError" in r["stdout"], r
    assert run(kernel, "print(spec(np.zeros((1, 2))).tolist())")["stdout"] == "[0.0]\n"


def test_hook_blocks_non_numpy_imports():
    k = SandboxKernel(os.path.join(SCRATCH_ROOT, f"imports_{time.time_ns()}"))
    k.start()
    try:
        # json, pickle, ctypes etc. are already loaded by the kernel or numpy, so the hook lets
        # them import; their dangerous operations are audited separately. tarfile is not loaded.
        r = k.exec(attempt("import tarfile"))
        assert "BLOCKED" in r["stdout"] and "ImportError" in r["stdout"], r
    finally:
        k.close()
