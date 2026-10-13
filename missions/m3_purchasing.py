"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

Your Turn #1: compares the simple v1 tier policy with a cost-based v2 policy that
adds per-GPU spot interruption rates, honest 24h/day reserved billing and a
1yr-vs-3yr choice over a planning horizon.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30
# How long we are confident a persistent service keeps its GPU type. 36 months
# lets a 3yr commitment amortize fully; shorter horizons favor 1yr / on-demand.
HORIZON_MONTHS = 36
MATRIX_HOURS = (3, 8, 14, 20, 24)


def _v1_cost(tier: str, gpu_hours: float, c: dict) -> float:
    """Original M3 costing: reserved billed only for hours actually run."""
    od = num(c["on_demand_hr"])
    if tier == "spot":
        return pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
    if tier == "reserved":
        return gpu_hours * num(c["reserved_3yr_hr"])
    return gpu_hours * od


def plan_jobs(jobs, cat, horizon_months: float = HORIZON_MONTHS) -> list[dict]:
    """Price every job under the v1 and v2 policies."""
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        job_days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        on_demand_cost = gpu_hours * num(c["on_demand_hr"])

        v1_tier = pricing.recommend_tier(hpd, interruptible)
        v1_cost = _v1_cost(v1_tier, gpu_hours, c)

        costs = pricing.tier_costs(hpd, interruptible, c, gpu_type=gtype, job_days=job_days,
                                   horizon_months=horizon_months, num_gpus=ngpu, days=DAYS)
        v2_choice = min(costs, key=costs.get)
        tier, _, term = v2_choice.partition("_")
        if tier == "on":
            tier, term = "on_demand", ""
        recs.append({
            "job_id": j["job_id"], "gpu_type": gtype, "hours_per_day": hpd,
            "interruptible": interruptible, "job_days": job_days,
            "tier": tier, "term": term,
            "on_demand": round(on_demand_cost), "optimized": round(costs[v2_choice]),
            "v1_tier": v1_tier, "v1_optimized": round(v1_cost),
            "interrupt_rate": pricing.SPOT_INTERRUPT_RATE.get(gtype, pricing.DEFAULT_SPOT_INTERRUPT_RATE),
        })
    return recs


def tier_matrix(cat, horizon_months: float = HORIZON_MONTHS) -> dict:
    """v2 recommendation for GPU type x duty cycle x interruptible (persistent job)."""
    out = {}
    for gtype, c in cat.items():
        for intr in (False, True):
            out[(gtype, intr)] = {h: pricing.recommend_tier(h, intr, gpu_type=gtype, job_days=30,
                                                            prices=c, horizon_months=horizon_months)
                                  for h in MATRIX_HOURS}
    return out


def _totals(recs, key):
    od = sum(r["on_demand"] for r in recs)
    opt = sum(r[key] for r in recs)
    return od, opt, ((od - opt) / od * 100 if od else 0.0)


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    recs = plan_jobs(jobs, cat)
    on_demand_monthly, optimized_monthly, savings_pct = _totals(recs, "optimized")
    _, v1_optimized, v1_pct = _totals(recs, "v1_optimized")

    horizons = {}
    for h in (6, 12, 24, 36):
        _, opt_h, pct_h = _totals(plan_jobs(jobs, cat, horizon_months=h), "optimized")
        horizons[h] = {"optimized_monthly": round(opt_h), "savings_pct": round(pct_h, 1)}
    matrix = tier_matrix(cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':6}{'h/d':>4}{'irate':>7}  {'v1 tier':10}{'v1 $':>9}  {'v2 tier':16}{'v2 $':>9}{'on-demand':>11}")
        for r in recs:
            v2 = r["tier"] + (f" ({r['term']})" if r["term"] else "")
            print(f"{r['job_id']:18}{r['gpu_type']:6}{r['hours_per_day']:>4.0f}{r['interrupt_rate']:>7.2f}  "
                  f"{r['v1_tier']:10}${r['v1_optimized']:>8,}  {v2:16}${r['optimized']:>8,}${r['on_demand']:>10,}")
        print(f"\nmonthly on-demand ${on_demand_monthly:,.0f}")
        print(f"  v1 policy -> ${v1_optimized:,.0f}  ({v1_pct:.1f}% saved; reserved billed only for hours run)")
        print(f"  v2 policy -> ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved; honest 24h reserved, "
              f"per-GPU spot risk, {HORIZON_MONTHS}-month horizon)")
        print("\nv2 savings vs planning horizon:  " +
              "  ".join(f"{h}mo={v['savings_pct']}%" for h, v in horizons.items()))
        print(f"\nv2 tier matrix (persistent job, {HORIZON_MONTHS}-month horizon); columns = hours/day")
        print(f"{'gpu':7}{'intr':>5}  " + "".join(f"{h:>14}" for h in MATRIX_HOURS))
        for (gtype, intr), row in matrix.items():
            print(f"{gtype:7}{int(intr):>5}  " + "".join(f"{row[h]:>14}" for h in MATRIX_HOURS))

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "policy_v1_optimized_monthly": round(v1_optimized), "policy_v1_savings_pct": round(v1_pct, 1),
            "horizon_sensitivity": horizons, "matrix": matrix, "horizon_months": HORIZON_MONTHS}


if __name__ == "__main__":
    run()
