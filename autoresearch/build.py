"""Force-clean build of a SealBot bot dir, platform-aware.

Usage:  python build.py <bot_dir>            # e.g. python build.py current
        python build.py <bot_dir> --smoke    # + import & 1-move sanity check

Why this exists (instead of `cd <bot_dir> && python setup.py build_ext`):
- setup.py does NOT track header deps: an edit to engine/*.h with a stale
  build/ silently ships the old engine (cost one false 0W/91L gate on the
  cluster). This script always removes build/ and the old extension first.
- setup.py hardcodes GCC/icpc flags (-O3 -march=native); MSVC needs
  /O2 /arch:AVX2 /fp:fast (/fp:fast ~ icpc's default fp model).

Run with the hexo-strix venv python (needs pybind11 + setuptools).
"""

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MSVC_FLAGS = ["/O2", "/arch:AVX2", "/DNDEBUG", "/fp:fast"]
GCC_FLAGS = ["-O3", "-march=native", "-DNDEBUG"]


def build(bot_dir: Path) -> Path:
    """Clean + rebuild minimax_cpp inside bot_dir. Returns the ext path."""
    if not (bot_dir / "minimax_bot.cpp").exists():
        sys.exit(f"not a bot dir (no minimax_bot.cpp): {bot_dir}")

    # Force-clean: stale objects must never survive a header edit.
    shutil.rmtree(bot_dir / "build", ignore_errors=True)
    for pat in ("*.pyd", "*.so"):
        for f in bot_dir.glob(pat):
            f.unlink()

    is_msvc = sys.platform == "win32"
    flags = MSVC_FLAGS if is_msvc else GCC_FLAGS

    setup_stub = f"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
setup(
    name="minimax_cpp",
    ext_modules=[Pybind11Extension(
        "minimax_cpp", [r"{bot_dir.as_posix()}/minimax_bot.cpp"],
        cxx_std=17, extra_compile_args={flags!r},
        include_dirs=[r"{bot_dir.as_posix()}"])],
    cmdclass={{"build_ext": build_ext}},
)
"""
    stub = bot_dir / "_build_stub.py"
    stub.write_text(setup_stub)
    try:
        subprocess.run(
            [sys.executable, str(stub), "build_ext", "--inplace"],
            cwd=bot_dir, check=True)
    finally:
        stub.unlink(missing_ok=True)

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    ext = bot_dir / f"minimax_cpp{ext_suffix}"
    if not ext.exists():
        sys.exit(f"build produced no extension at {ext}")
    print(f"built: {ext}")
    return ext


def smoke(bot_dir: Path):
    """Import the freshly built extension and play one move."""
    code = f"""
import sys, time
sys.path.insert(0, r"{REPO.as_posix()}")
sys.path.insert(0, r"{bot_dir.as_posix()}")
from game import HexGame
from minimax_cpp import MinimaxBot
g = HexGame(win_length=6); g.make_move(0, 0)
b = MinimaxBot(time_limit=0.1)
t0 = time.perf_counter(); mv = b.get_move(g)
print(f"smoke ok: {{mv}} in {{time.perf_counter()-t0:.2f}}s")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    d = Path(args[0])
    if not d.is_absolute():
        d = REPO / d
    build(d)
    if "--smoke" in sys.argv:
        smoke(d)
