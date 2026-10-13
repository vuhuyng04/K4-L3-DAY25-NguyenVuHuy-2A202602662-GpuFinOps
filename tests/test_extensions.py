"""Tests for the five "Your Turn" extensions."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data import generate
from finops import pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m5_report, m6_carbon_scheduling

H100 = {"on_demand_hr": 2.5, "spot_hr": 1.5, "reserved_1yr_hr": 2.0, "reserved_3yr_hr": 1.4}


# --- Ext 1: tier policy ---------------------------------------------------
def test_recommend_tier_v1_unchanged_without_prices():
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_reserved_term_follows_horizon():
    assert pricing.recommend_tier(24, False, gpu_type="H100", prices=H100, horizon_months=36) == "reserved_3yr"
    assert pricing.recommend_tier(24, False, gpu_type="H100", prices=H100, horizon_months=12) == "reserved_1yr"
    # a 3yr commitment used for 12 months costs 3x its sticker rate
    assert abs(pricing.reserved_effective_hr(1.4, 36, 12) - 4.2) < 1e-9


def test_short_jobs_never_commit_and_reserved_bills_24h():
    assert pricing.recommend_tier(20, False, gpu_type="H100", job_days=14, prices=H100) == "on_demand"
    costs = pricing.tier_costs(12, False, H100, gpu_type="H100", job_days=30)
    assert abs(costs["reserved_3yr"] - 24 * 30 * 1.4) < 1e-6   # idle hours still billed
    assert pricing.recommend_tier(6, False, gpu_type="H100", job_days=30, prices=H100) == "on_demand"


def test_high_interruption_rate_kills_thin_spot_discount():
    thin = dict(H100, spot_hr=2.2)   # only 12% off on-demand
    assert pricing.recommend_tier(20, True, gpu_type="H100", prices=thin, job_days=5) == "spot"
    assert pricing.recommend_tier(20, True, gpu_type="H100", prices=thin, job_days=5,
                                  spot_interrupt_rate=0.30) == "on_demand"


# --- Ext 2: right-sizing --------------------------------------------------
def test_rightsizing_respects_bandwidth_and_vram():
    generate.main()
    r1 = m1_efficiency_audit.run(verbose=False)
    cat = {c["gpu_type"]: c for c in r1["catalog_efficiency"]}
    for p in r1["rightsizing"]["plans"]:
        n, gtype = p["proposed"].split("x ")
        assert int(n) * cat[gtype]["peak_bw_tbs"] >= p["need_bw_tbs"]
        assert int(n) * cat[gtype]["hbm_gb"] >= p["need_gb"]
        assert p["proposed_usd_hr"] <= p["current_usd_hr"]
    assert r1["rightsizing"]["monthly_savings"] > 0


# --- Ext 3: cache economics -----------------------------------------------
def test_cache_break_even_and_worth_it():
    assert abs(pricing.cache_break_even_reads(0.25, 0.10) - 0.25 / 0.9) < 1e-9
    assert pricing.cache_is_worth_it(1.0)
    assert not pricing.cache_is_worth_it(0.1)
    # flat storage fee hurts cheap models more
    small = pricing.cache_storage_frac(1.0, 1.0, 0.20)
    large = pricing.cache_storage_frac(1.0, 1.0, 3.00)
    assert pricing.cache_break_even_reads(0, 0.1, small) > pricing.cache_break_even_reads(0, 0.1, large)


def test_request_cost_write_premium_default_is_zero():
    base = pricing.request_cost(1000, 100, 3.0, 15.0, cached_in=500)
    assert pricing.request_cost(1000, 100, 3.0, 15.0, cached_in=500, write_premium=0.25) > base


def test_m2_gates_cache_and_ladder_is_monotonic():
    generate.main()
    r2 = m2_inference_levers.run(verbose=False)
    teams = r2["cache"]["teams"]
    assert teams["assistant"]["worth_it"] and not teams["search"]["worth_it"]
    per_m = [s["per_m"] for s in r2["ladder"]]
    assert per_m == sorted(per_m, reverse=True)
    assert r2["optimized_daily"] <= r2["cache_all_daily"]


# --- Ext 4: reasoning budget ----------------------------------------------
def test_reasoning_split_and_caps():
    generate.main()
    b = m2_inference_levers.run(verbose=False)["reasoning"]
    s = b["split"]
    assert s["reasoning"]["requests"] + s["standard"]["requests"] == 2400
    assert s["reasoning"]["share_wh"] > s["reasoning"]["share_traffic"]
    saved = [sc["saved_usd_day"] for sc in b["scenarios"]]   # caps 10% -> 0%
    assert saved == sorted(saved) and saved[-1] > 0


# --- Ext 5: carbon-aware scheduling ---------------------------------------
def test_carbon_scheduling_regions():
    generate.main()
    r6 = m6_carbon_scheduling.run(verbose=False)
    assert r6["best"]["cleanest"] == "europe-north1"
    assert r6["best"]["cheapest"] == "us-east-wa"
    assert r6["co2_saved_kg"] > 0 and len(r6["regions"]) == 5


# --- Report ---------------------------------------------------------------
def test_m5_report_has_analysis_sections():
    generate.main()
    r5 = m5_report.run(verbose=False)
    assert len(r5["levers"]) == 4
    md = open(os.path.join(ROOT, "outputs", "report.md"), encoding="utf-8").read()
    for heading in ("$/1M-token", "GPU-Util lie", "Hành động đề xuất", "Sustainability", "Your Turn"):
        assert heading in md
