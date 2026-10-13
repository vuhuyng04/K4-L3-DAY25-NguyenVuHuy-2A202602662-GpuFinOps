"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py

Your Turn #2: right-size memory-bound inference fleets by $/GB-VRAM and peak
HBM bandwidth instead of $/GPU-hr.
"""
from __future__ import annotations
import math
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
MEMORY_BOUND_WORKLOADS = ("infer", "embed")
TARGET_MBU = 0.60   # sustainable decode MBU before latency degrades (deck §5)


def catalog_efficiency(cat) -> list[dict]:
    """$/GB-VRAM and $/(TB/s) per GPU type — the unit prices memory-bound work pays."""
    out = []
    for gtype, c in cat.items():
        od = num(c["on_demand_hr"])
        out.append({"gpu_type": gtype, "on_demand_hr": od, "hbm_gb": num(c["hbm_gb"]),
                    "peak_bw_tbs": num(c["peak_bw_tbs"]),
                    "usd_per_gb": od / num(c["hbm_gb"]), "usd_per_tbs": od / num(c["peak_bw_tbs"])})
    return sorted(out, key=lambda x: x["usd_per_gb"])


def size_option(need_gb: float, need_bw_tbs: float, c: dict) -> int:
    """GPUs of one type needed to hold `need_gb` of VRAM and serve `need_bw_tbs` of HBM BW."""
    return max(1, math.ceil(need_gb / num(c["hbm_gb"])), math.ceil(need_bw_tbs / num(c["peak_bw_tbs"])))


def rightsize_memory_bound(tel, cat, target_mbu: float = TARGET_MBU) -> dict:
    """Re-plan each memory-bound replica group (same GPU type + workload) on the cheapest fleet.

    Requirement per group = sum of each GPU's peak VRAM in use and peak HBM BW
    divided by `target_mbu`. Every catalog type is priced as n x GPU, so a cheap
    $/GPU-hr part that lacks bandwidth (L4: 0.3 TB/s) needs more units and loses.
    """
    peak = defaultdict(lambda: {"bw": 0.0, "mem": 0.0, "type": None, "workload": None})
    for r in tel:
        if r["workload"] not in MEMORY_BOUND_WORKLOADS:
            continue
        p = peak[r["gpu_id"]]
        p["type"], p["workload"] = r["gpu_type"], r["workload"]
        p["bw"] = max(p["bw"], num(r["achieved_bw_tbs"]))
        p["mem"] = max(p["mem"], num(r["mem_used_gb"]))

    groups = defaultdict(list)
    for gid, p in peak.items():
        groups[(p["type"], p["workload"])].append((gid, p))

    plans = []
    for (gtype, workload), members in sorted(groups.items()):
        need_gb = sum(p["mem"] for _, p in members)
        need_bw = sum(p["bw"] for _, p in members) / target_mbu
        cur_n = len(members)
        cur_hr = cur_n * num(cat[gtype]["on_demand_hr"])
        options = []
        for t, c in cat.items():
            n = size_option(need_gb, need_bw, c)
            options.append({"gpu_type": t, "n": n, "usd_hr": n * num(c["on_demand_hr"])})
        best = min(options, key=lambda o: (o["usd_hr"], o["n"]))
        if best["usd_hr"] >= cur_hr:
            best = {"gpu_type": gtype, "n": cur_n, "usd_hr": cur_hr}
        cheapest_hr = min(cat, key=lambda t: num(cat[t]["on_demand_hr"]))
        naive = next(o for o in options if o["gpu_type"] == cheapest_hr)
        plans.append({
            "gpus": [g for g, _ in members], "workload": workload,
            "current": f"{cur_n}x {gtype}", "current_usd_hr": round(cur_hr, 2),
            "need_gb": round(need_gb, 1), "need_bw_tbs": round(need_bw, 2),
            "proposed": f"{best['n']}x {best['gpu_type']}", "proposed_usd_hr": round(best["usd_hr"], 2),
            "naive_cheapest": f"{naive['n']}x {naive['gpu_type']}", "naive_usd_hr": round(naive["usd_hr"], 2),
            "monthly_savings": round((cur_hr - best["usd_hr"]) * 24 * DAYS),
        })
    return {"target_mbu": target_mbu, "plans": plans,
            "monthly_savings": sum(p["monthly_savings"] for p in plans)}


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

    efficiency = catalog_efficiency(cat)
    rightsizing = rightsize_memory_bound(tel, cat)
    if verbose:
        print("\n-- Right-sizing memory-bound inference (Your Turn #2) --")
        print(f"{'gpu':7}{'$/hr':>7}{'HBM GB':>8}{'TB/s':>6}{'$/GB-VRAM':>11}{'$/(TB/s)':>10}")
        for e in efficiency:
            print(f"{e['gpu_type']:7}{e['on_demand_hr']:>7.2f}{e['hbm_gb']:>8.0f}{e['peak_bw_tbs']:>6.2f}"
                  f"{e['usd_per_gb']:>11.4f}{e['usd_per_tbs']:>10.3f}")
        print(f"\nneed = peak VRAM in use, peak HBM BW / target MBU {rightsizing['target_mbu']:.0%}")
        print(f"{'group':24}{'current':>10}{'$/hr':>7}{'need GB':>9}{'need TB/s':>10}"
              f"{'proposed':>12}{'$/hr':>7}{'cheapest$/hr':>16}{'$/month saved':>15}")
        for p in rightsizing["plans"]:
            naive = f"{p['naive_cheapest']} ${p['naive_usd_hr']:.2f}"
            print(f"{'+'.join(g.replace('gpu-', '') for g in p['gpus']):24}{p['current']:>10}{p['current_usd_hr']:>7.2f}"
                  f"{p['need_gb']:>9}{p['need_bw_tbs']:>10}{p['proposed']:>12}{p['proposed_usd_hr']:>7.2f}"
                  f"{naive:>16}  ${p['monthly_savings']:>12,}")
        print(f"right-size all memory-bound groups: ${rightsizing['monthly_savings']:,}/month")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "catalog_efficiency": efficiency, "rightsizing": rightsizing}


if __name__ == "__main__":
    run()
