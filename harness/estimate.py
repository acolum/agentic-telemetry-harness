"""Project the four-arm study cost from a smoke run's measured per-session costs.

    python -m harness.run --out runs/smoke --smoke 4 --cap 3      # 12 sessions, then prints the projection
    python -m harness.estimate --smoke runs/smoke [--eval 300 --calib 60]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from .config import ARCHITECTURES, PRICES_PER_M

ARMS = {  # arm: (calibration per arch, eval per arch, injected-only?, model override or None)
    "A main": ("calib", "eval", False, None),
    "B observe": (0, "injected", False, None),
    "C adaptive": (0, "eval_c", False, None),
    "D replicate": ("calib", "eval", False, "gpt-4o-mini"),
}


def report(smoke_dir, eval_n=500, calib_n=80, attack_fraction=0.6, eval_c=400, rep_model="gpt-4o-mini"):
    sessions = [json.load(open(f)) for f in glob.glob(os.path.join(smoke_dir, "sessions", "*.json"))]
    if not sessions:
        print("no sessions found"); return
    model = sessions[0]["meta"]["model"]
    per = {}
    for arch in ARCHITECTURES:
        S = [s for s in sessions if s["meta"]["architecture"] == arch]
        if not S:
            continue
        per[arch] = {"usd": sum(s["meta"].get("cost_usd", 0) for s in S) / len(S),
                     "tin": sum(s["meta"].get("tokens_in", 0) for s in S) / len(S),
                     "tout": sum(s["meta"].get("tokens_out", 0) for s in S) / len(S),
                     "events": sum(s["meta"]["n_events"] for s in S) / len(S), "n": len(S)}
    print(f"\nMeasured per-session cost on {model} (smoke run, {len(sessions)} sessions):")
    for arch, d in per.items():
        print(f"  {arch:11s} ${d['usd']:.4f}  ({d['tin']:.0f} in / {d['tout']:.0f} out tokens, {d['events']:.0f} events, n={d['n']})")
    # scale factor for the replicate model: ratio of (input+output) prices, applied to the agent share (~90% of cost)
    pm, pr = PRICES_PER_M.get(model, (0.4, 1.6)), PRICES_PER_M.get(rep_model, (0.15, 0.6))
    denom = pm[0] * 0.9 + pm[1] * 0.1
    rep_scale = 0.1 + 0.9 * ((pr[0] * 0.9 + pr[1] * 0.1) / denom) if denom > 0 else 1.0
    n_inj = int(round(eval_n * attack_fraction))
    sizes = {"A main": (calib_n + eval_n, 1.0), "B observe": (n_inj, 1.0), "C adaptive": (eval_c, 1.0),
             "D replicate": (calib_n + eval_n, rep_scale)}
    total = 0.0
    print("\nProjected arm cost (central; add ~40% for a conservative figure):")
    for arm, (n_per_arch, scale) in sizes.items():
        cost = sum(per.get(a, {"usd": 0})["usd"] * n_per_arch * scale for a in ARCHITECTURES)
        total += cost
        print(f"  {arm:12s} {3 * n_per_arch:5d} sessions  ${cost:6.2f}")
    print(f"  {'TOTAL':12s} {'':5s}           ${total:6.2f}   (conservative ${1.4 * total:.2f})")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True); ap.add_argument("--eval", type=int, default=500); ap.add_argument("--calib", type=int, default=80)
    a = ap.parse_args(argv)
    report(a.smoke, eval_n=a.eval, calib_n=a.calib)


if __name__ == "__main__":
    main()
