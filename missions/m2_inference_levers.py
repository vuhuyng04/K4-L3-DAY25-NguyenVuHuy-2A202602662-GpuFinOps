"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

Your Turn #3: prompt caching is only applied where `cache_is_worth_it()` says the
re-read rate beats the cache-write cost.
Your Turn #4: reasoning budget — $ and Wh of is_reasoning traffic, and what a
traffic cap on reasoning would save.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
WRITE_PREMIUM = pricing.CACHE_WRITE_PREMIUM   # Anthropic-style cache pricing
# Gemini-style alternative: flat storage fee per cached token-hour.
STORAGE_USD_PER_M_HR = 1.00
CACHE_TTL_HR = 1.0
REASONING_OUTPUT_TAX = 6          # generator: reasoning multiplies output tokens ~6x
REASONING_CAPS = (0.10, 0.05, 0.02, 0.0)


def _parse(r) -> dict:
    return {"inp": int(num(r["input_tokens"])), "out": int(num(r["output_tokens"])),
            "cached": int(num(r["cached_input_tokens"])), "batch": bool(int(num(r["is_batch"]))),
            "reasoning": bool(int(num(r["is_reasoning"]))), "tier": r["route_tier"], "team": r["team"]}


def cache_economics(rows) -> dict:
    """Per-team cache re-read rate vs break-even, plus break-even per model tier.

    The log has no prefix ids, so re-reads are estimated from the hit ratio h:
    with every miss writing the prefix once, reads per write = h / (1 - h).
    """
    tot = defaultdict(lambda: [0, 0])
    for p in rows:
        tot[p["team"]][0] += p["inp"]
        tot[p["team"]][1] += p["cached"]
    be = pricing.cache_break_even_reads(WRITE_PREMIUM)
    teams = {}
    for team, (inp, cached) in sorted(tot.items()):
        h = cached / inp if inp else 0.0
        reads = h / (1 - h) if h < 1 else float("inf")
        teams[team] = {"hit_ratio": round(h, 3), "avg_reads": round(reads, 2),
                       "break_even": round(be, 2), "worth_it": pricing.cache_is_worth_it(reads, WRITE_PREMIUM)}
    tiers = {}
    for tier, (pin, _) in MODEL_PRICES.items():
        frac = pricing.cache_storage_frac(STORAGE_USD_PER_M_HR, CACHE_TTL_HR, pin)
        tiers[tier] = {"anthropic_style": round(be, 2),
                       "gemini_style": round(pricing.cache_break_even_reads(0.0, 0.10, frac), 2)}
    return {"teams": teams, "tiers": tiers,
            "cache_teams": {t for t, v in teams.items() if v["worth_it"]}}


def request_cost(p, cascade=True, cache_teams=(), batch=True, write_premium=WRITE_PREMIUM) -> float:
    """Cost of one parsed request with the chosen levers switched on."""
    pin, pout = MODEL_PRICES[p["tier"] if cascade else "large"]
    cache_on = p["team"] in cache_teams
    return pricing.request_cost(p["inp"], p["out"], pin, pout,
                                cached_in=p["cached"] if cache_on else 0,
                                batch=p["batch"] and batch,
                                write_premium=write_premium if cache_on else 0.0)


def _wh(p) -> float:
    return sustainability.wh_per_query(p["inp"] + p["out"], is_reasoning=p["reasoning"])


def reasoning_budget(rows, cache_teams, caps=REASONING_CAPS) -> dict:
    """Split optimized $ and Wh by is_reasoning, then simulate reasoning traffic caps.

    Under a cap of c x traffic, the reasoning requests with the longest outputs
    (proxy for the hardest tasks) keep reasoning; the rest are demoted to a normal
    answer: is_reasoning=0 and output / REASONING_OUTPUT_TAX.
    """
    split = {}
    for flag in (True, False):
        sub = [p for p in rows if p["reasoning"] == flag]
        split["reasoning" if flag else "standard"] = {
            "requests": len(sub),
            "cost": sum(request_cost(p, cache_teams=cache_teams) for p in sub),
            "wh": sum(_wh(p) for p in sub),
        }
    tot = {k: sum(v[k] for v in split.values()) for k in ("requests", "cost", "wh")}
    for v in split.values():
        v["share_traffic"] = v["requests"] / tot["requests"]
        v["share_cost"] = v["cost"] / tot["cost"]
        v["share_wh"] = v["wh"] / tot["wh"]
        v["wh_per_query"] = v["wh"] / v["requests"] if v["requests"] else 0.0
    std, rsn = split["standard"], split["reasoning"]
    ratio = rsn["wh_per_query"] / std["wh_per_query"] if std["wh_per_query"] else 0.0

    hard_first = sorted((p for p in rows if p["reasoning"]), key=lambda p: -p["out"])
    scenarios = []
    for cap in caps:
        keep = int(cap * len(rows))
        demoted = [dict(p, reasoning=False, out=p["out"] // REASONING_OUTPUT_TAX) for p in hard_first[keep:]]
        cost = std["cost"] + sum(request_cost(p, cache_teams=cache_teams) for p in hard_first[:keep])             + sum(request_cost(p, cache_teams=cache_teams) for p in demoted)
        wh = std["wh"] + sum(_wh(p) for p in hard_first[:keep]) + sum(_wh(p) for p in demoted)
        scenarios.append({"cap": cap, "reasoning_requests": min(keep, len(hard_first)),
                          "binding": keep < len(hard_first),
                          "cost": cost, "wh": wh,
                          "saved_usd_day": tot["cost"] - cost, "saved_wh_day": tot["wh"] - wh})
    min_kept_out = hard_first[max(0, int(0.05 * len(rows)) - 1)]["out"] if hard_first else 0
    return {"split": split, "total": tot, "wh_ratio": ratio, "scenarios": scenarios,
            "cap5_min_output": min_kept_out}


def run(verbose: bool = True) -> dict:
    rows = [_parse(r) for r in load_csv("token_usage.csv")]
    total_tokens = sum(p["inp"] + p["out"] for p in rows)
    econ = cache_economics(rows)
    cache_teams = econ["cache_teams"]
    all_teams = set(econ["teams"])

    def total(**kw):
        return sum(request_cost(p, **kw) for p in rows)

    # $/1M-token ladder: switch levers on one at a time
    ladder = [
        ("Baseline (all large, no cache, no batch)", total(cascade=False, cache_teams=(), batch=False)),
        ("+ Cascade (route_tier)", total(cascade=True, cache_teams=(), batch=False)),
        ("+ Prompt caching (gated)", total(cascade=True, cache_teams=cache_teams, batch=False)),
        ("+ Batch API", total(cascade=True, cache_teams=cache_teams, batch=True)),
    ]
    base_cost, opt_cost = ladder[0][1], ladder[-1][1]
    ladder = [{"step": s, "daily": round(c, 2), "per_m": round(pricing.dollars_per_million(c, total_tokens), 3)}
              for s, c in ladder]
    naive_cache = total(cascade=True, cache_teams=all_teams, batch=True, write_premium=0.0)
    cache_all = total(cascade=True, cache_teams=all_teams, batch=True)

    budget = reasoning_budget(rows, cache_teams)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("\n$/1M-token ladder (levers switched on one at a time):")
        for s in ladder:
            print(f"  {s['step']:42}${s['daily']:>7.2f}/day  ${s['per_m']:.3f}/1M")

        print("\n-- cache_is_worth_it() (Your Turn #3) --")
        print(f"{'team':11}{'hit ratio':>10}{'reads/write':>12}{'break-even':>11}  cache?")
        for t, v in econ["teams"].items():
            print(f"{t:11}{v['hit_ratio']:>10.1%}{v['avg_reads']:>12}{v['break_even']:>11}  {'ON' if v['worth_it'] else 'off'}")
        print("break-even reads per model tier:")
        for tier, v in econ["tiers"].items():
            print(f"  {tier:6} write-premium 1.25x: {v['anthropic_style']:>5}   "
                  f"storage ${STORAGE_USD_PER_M_HR}/1M-tok-hr x {CACHE_TTL_HR:g}h: {v['gemini_style']:>6}")
        print(f"cache ON for every team (pays write premium): ${cache_all:.2f}/day vs gated ${opt_cost:.2f}/day")
        print(f"original M2 accounting (cache everywhere, writes free): ${naive_cache:.2f}/day")

        print("\n-- Reasoning budget (Your Turn #4) --")
        print(f"{'traffic':10}{'requests':>9}{'% traffic':>10}{'$/day':>8}{'% cost':>8}{'Wh/day':>9}{'% Wh':>7}{'Wh/query':>10}")
        for k, v in budget["split"].items():
            print(f"{k:10}{v['requests']:>9}{v['share_traffic']:>10.1%}{v['cost']:>8.2f}{v['share_cost']:>8.1%}"
                  f"{v['wh']:>9.0f}{v['share_wh']:>7.1%}{v['wh_per_query']:>10.2f}")
        print(f"a reasoning query uses {budget['wh_ratio']:.0f}x the energy of a standard one")
        print("cap reasoning to x% of traffic (longest outputs keep reasoning):")
        for sc in budget["scenarios"]:
            note = "" if sc["binding"] else "  (not binding: already below cap)"
            print(f"  cap {sc['cap']:>4.0%}: save ${sc['saved_usd_day']:.2f}/day (${sc['saved_usd_day']*30:.0f}/mo), "
                  f"{sc['saved_wh_day']/1000:.1f} kWh/day ({sc['saved_wh_day']*30/1000:.0f} kWh/mo){note}")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "ladder": ladder, "cache": econ,
        "cache_all_daily": round(cache_all, 2), "naive_cache_daily": round(naive_cache, 2),
        "reasoning": budget,
        "small_share": sum(p["tier"] == "small" for p in rows) / len(rows),
    }


if __name__ == "__main__":
    run()
