"""M6 — Carbon-aware Scheduling (Your Turn #5, deck §11).

Run: python missions/m6_carbon_scheduling.py

Interruptible jobs (training, fine-tune, sandbox, batch eval) are not latency
bound, so they can run wherever power is cheapest and cleanest. Compare all
regions on electricity $ and gCO2e for those jobs, starting from us-east-1.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

HOME_REGION = "us-east-1"
PUE = 1.2                       # datacenter overhead (cooling, power conversion)
CARBON_PRICE_USD_PER_T = 100.0  # internal carbon price used to put CO2 on the P&L
# Rough distance from NimbusAI's US-east users (illustrative): matters for serving,
# not for interruptible batch work.
LATENCY_TIER = {"us-east-1": "local", "us-east-wa": "near", "us-west-2": "cross-US",
                "europe-north1": "transatlantic", "europe-central2": "transatlantic"}


def job_energy_wh(job, cat) -> float:
    """Wall-plug Wh for a job's whole run: TDP x GPUs x hours/day x days x PUE."""
    watts = num(cat[job["gpu_type"]]["watts"])
    return watts * num(job["num_gpus"]) * num(job["hours_per_day"]) * num(job["days"]) * PUE


def region_table(wh: float) -> list[dict]:
    """Electricity $, gCO2e and carbon-priced $ of `wh` in every region."""
    rows = []
    for region in sustainability.REGION_CARBON:
        usd = sustainability.energy_cost_usd(wh, region)
        g = sustainability.carbon_g(wh, region)
        rows.append({"region": region, "usd_per_kwh": sustainability.REGION_PRICE_KWH[region],
                     "gco2_per_kwh": sustainability.REGION_CARBON[region],
                     "energy_usd": usd, "carbon_kg": g / 1000.0,
                     "carbon_priced_usd": usd + g / 1e6 * CARBON_PRICE_USD_PER_T,
                     "latency": LATENCY_TIER.get(region, "?")})
    return rows


def pick_regions(table) -> dict:
    """Best region by $, by CO2, and balanced (50/50 min-max normalized score)."""
    def norm(key):
        lo, hi = min(r[key] for r in table), max(r[key] for r in table)
        return {r["region"]: (r[key] - lo) / (hi - lo) if hi > lo else 0.0 for r in table}
    n_usd, n_co2 = norm("energy_usd"), norm("carbon_kg")
    return {
        "cheapest": min(table, key=lambda r: r["energy_usd"])["region"],
        "cleanest": min(table, key=lambda r: r["carbon_kg"])["region"],
        "balanced": min(table, key=lambda r: 0.5 * n_usd[r["region"]] + 0.5 * n_co2[r["region"]])["region"],
        "carbon_priced": min(table, key=lambda r: r["carbon_priced_usd"])["region"],
    }


def run(verbose: bool = True) -> dict:
    cat = catalog_by_type()
    jobs = [j for j in load_csv("workloads.csv") if int(num(j["interruptible"]))]
    per_job = []
    for j in jobs:
        wh = job_energy_wh(j, cat)
        per_job.append({"job_id": j["job_id"], "kwh": wh / 1000.0,
                        "home_kg": sustainability.carbon_g(wh, HOME_REGION) / 1000.0})
    total_wh = sum(p["kwh"] for p in per_job) * 1000.0
    table = region_table(total_wh)
    best = pick_regions(table)
    by_region = {r["region"]: r for r in table}
    for p in per_job:
        p["clean_kg"] = sustainability.carbon_g(p["kwh"] * 1000.0, best["cleanest"]) / 1000.0
    home, clean = by_region[HOME_REGION], by_region[best["cleanest"]]
    cheap = by_region[best["cheapest"]]

    result = {
        "jobs": per_job, "total_kwh": total_wh / 1000.0, "regions": table, "best": best,
        "home_region": HOME_REGION,
        "co2_saved_kg": home["carbon_kg"] - clean["carbon_kg"],
        "co2_saved_pct": (1 - clean["carbon_kg"] / home["carbon_kg"]) * 100 if home["carbon_kg"] else 0.0,
        "energy_usd_saved_clean": home["energy_usd"] - clean["energy_usd"],
        "energy_usd_saved_cheap": home["energy_usd"] - cheap["energy_usd"],
        "carbon_price": CARBON_PRICE_USD_PER_T,
    }

    if verbose:
        print("== M6 Carbon-aware Scheduling ==")
        print(f"interruptible jobs, energy = TDP x GPUs x h/day x days x PUE {PUE}")
        print(f"{'job':18}{'kWh':>9}{HOME_REGION + ' kg':>15}{best['cleanest'] + ' kg':>19}")
        for p in per_job:
            print(f"{p['job_id']:18}{p['kwh']:>9,.0f}{p['home_kg']:>15,.1f}{p['clean_kg']:>19,.1f}")
        print(f"\ntotal {result['total_kwh']:,.0f} kWh in all regions:")
        print(f"{'region':17}{'$/kWh':>7}{'gCO2/kWh':>10}{'energy $':>10}{'CO2 kg':>9}"
              f"{'$ + CO2@$' + str(int(CARBON_PRICE_USD_PER_T)) + '/t':>16}  latency")
        for r in sorted(table, key=lambda r: r["carbon_kg"]):
            print(f"{r['region']:17}{r['usd_per_kwh']:>7.3f}{r['gco2_per_kwh']:>10}{r['energy_usd']:>10,.0f}"
                  f"{r['carbon_kg']:>9,.0f}{r['carbon_priced_usd']:>16,.0f}  {r['latency']}")
        print(f"\ncheapest $: {best['cheapest']}   cleanest CO2: {best['cleanest']}   "
              f"balanced 50/50: {best['balanced']}   with carbon price: {best['carbon_priced']}")
        print(f"{HOME_REGION} -> {best['cleanest']}: save {result['co2_saved_kg']:,.0f} kg CO2e "
              f"({result['co2_saved_pct']:.0f}%) and ${result['energy_usd_saved_clean']:,.0f} electricity")

    return result


if __name__ == "__main__":
    run()
