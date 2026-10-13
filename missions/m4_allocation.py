"""M4 — Cost Allocation: tag -> showback -> chargeback + FOCUS export (deck §10).

Run: python missions/m4_allocation.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import csv
import os
from missions._common import load_csv, ROOT
from finops import allocation
from missions import m2_inference_levers as m2

TAGS = ["team", "project"]


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    parsed = [m2._parse(r) for r in rows]
    cache_teams = m2.cache_economics(parsed)["cache_teams"]  # same optimized pricing as M2
    priced = []
    for r, p in zip(rows, parsed):
        cost = m2.request_cost(p, cache_teams=cache_teams)
        priced.append({"team": r["team"], "project": r["project"],
                       "service": "gpu-inference", "cost": cost})

    by_team = allocation.cost_by_tag(priced, "team")
    by_project = allocation.cost_by_tag(priced, "project")
    coverage = allocation.tag_coverage(priced, TAGS)
    ready = allocation.chargeback_ready(coverage)

    # write a FOCUS-style export (first 50 rows as a sample)
    focus = allocation.to_focus_rows(priced[:50])
    out_path = os.path.join(ROOT, "outputs", "focus_export.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["BillingAccountId", "ChargePeriodStart", "ServiceCategory",
                    "ServiceName", "ResourceId", "BilledCost", "BillingCurrency", "team", "project"])
        for fr in focus:
            w.writerow([fr["BillingAccountId"], fr["ChargePeriodStart"], fr["ServiceCategory"],
                        fr["ServiceName"], fr["ResourceId"], fr["BilledCost"], fr["BillingCurrency"],
                        fr["Tags"]["team"], fr["Tags"]["project"]])

    if verbose:
        print("== M4 Cost Allocation ==")
        print("cost by team ($/day):")
        for k, v in sorted(by_team.items(), key=lambda x: -x[1]):
            print(f"  {k:12} ${v:8.2f}")
        print(f"tag coverage: {coverage:.0%}  ->  chargeback ready? {ready}")
        print(f"FOCUS export -> outputs/focus_export.csv ({len(focus)} rows)")

    return {"by_team": {k: round(v, 2) for k, v in by_team.items()},
            "by_project": {k: round(v, 2) for k, v in by_project.items()},
            "tag_coverage": round(coverage, 3), "chargeback_ready": ready}


if __name__ == "__main__":
    run()
