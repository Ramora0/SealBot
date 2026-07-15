"""Scaling battery: data fraction x model size, steps-matched, shared
cache, identical val split (seed 0). Non-mirror (clean architecture
science; the shipping net adds --mirror separately).

Axes:
  data:  25% / 50% / 100% at M64/C32 (epochs scaled so total steps match)
  size:  C16 / C32(anchor) / M128-C32 / M128-C64 at 100% data
         (M is train-time only — baked table cost depends on C alone)

Run (hexo venv, GPU):  python mixnet_battery.py
"""

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "output_battery", "cache_shared.npz")

CONFIGS = [
    ("anchor",  ["--epochs", "12"]),
    ("d25",     ["--data-frac", "0.25", "--epochs", "48"]),
    ("d50",     ["--data-frac", "0.5",  "--epochs", "24"]),
    ("c16",     ["--C", "16", "--epochs", "12"]),
    ("m128",    ["--M", "128", "--epochs", "12"]),
    ("m128c64", ["--M", "128", "--C", "64", "--epochs", "12"]),
]


def main():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    results = {}
    for name, flags in CONFIGS:
        out = f"output_batt_{name}"
        t0 = time.time()
        print(f"=== [{name}] starting ({' '.join(flags)}) ===", flush=True)
        p = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "mixnet_train.py"),
             "--cache", CACHE, "--out", out, "--device", "cuda"] + flags,
            cwd=SCRIPT_DIR, capture_output=True, text=True)
        tail = "\n".join((p.stdout or "").strip().splitlines()[-3:])
        print(tail, flush=True)
        if p.returncode != 0:
            print(f"=== [{name}] FAILED ===\n{(p.stderr or '')[-2000:]}",
                  flush=True)
            results[name] = "FAILED"
            continue
        last = next((l for l in reversed(p.stdout.splitlines())
                     if l.startswith("epoch")), "?")
        results[name] = last
        print(f"=== [{name}] done in {(time.time()-t0)/60:.0f} min ===",
              flush=True)
    print("\n===== BATTERY SUMMARY =====")
    for name, _ in CONFIGS:
        print(f"{name:8s}  {results.get(name, 'MISSING')}")


if __name__ == "__main__":
    main()
