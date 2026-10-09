#!/usr/bin/env python3
"""compare.py RESULTS.jsonl [BASELINE.jsonl]: per-stage instruction counts (millions) for each variant and workload
of a bench.sh run, ranked by cost, and with a baseline the change per stage. After bynds/needle-rs perfvm/compare.py
(dcb8f79), with Oído's stages (tasr_nemo.c TASR_PROFILE spans) and workloads."""
import json, sys


def load(p):
    return {(r["variant"], r["workload"]): r for r in map(json.loads, open(p))}


cur = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else {}
for wl in ["utterance", "stream", "stream_setup"]:
    for var in ["plain", "neon"]:
        r = cur.get((var, wl))
        if not r:
            continue
        b = base.get((var, wl))
        ops = {k: v[0] for k, v in r["ops"].items()}
        ops["other"] = r["other"]
        bops = ({k: v[0] for k, v in b["ops"].items()} | {"other": b["other"]}) if b else {}
        print(f"\n{wl} / {var}: {r['total'] / 1e6:,.1f} M instructions"
              + (f" (baseline {b['total'] / 1e6:,.1f} M, {(r['total'] - b['total']) / b['total'] * 100:+.2f}%)" if b else ""))
        print(f"  {'stage':12}{'M instr':>10}{'share':>8}{'calls':>9}" + (f"{'baseline':>10}{'change':>9}" if b else ""))
        for k, v in sorted(ops.items(), key=lambda kv: -kv[1]):
            if not v and not bops.get(k):
                continue
            calls = r["ops"][k][1] if k in r["ops"] else ""
            line = f"  {k:12}{v / 1e6:10.2f}{v / r['total'] * 100:7.1f}%{calls:>9}"
            if b:
                bv = bops.get(k, 0)
                line += f"{bv / 1e6:10.2f}{(v - bv) / bv * 100 if bv else 0:+8.1f}%"
            print(line)
