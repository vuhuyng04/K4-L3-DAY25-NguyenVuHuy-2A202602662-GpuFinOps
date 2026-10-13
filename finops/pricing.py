"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
    write_premium: float = 0.0,     # cache-write surcharge on uncached input (Anthropic 5-min: 0.25)
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price.

    With caching enabled, every uncached input token is also written to the cache
    and billed at (1 + write_premium) x price; leave it 0 to ignore write cost.
    """
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m * (1.0 + write_premium)
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


CACHE_WRITE_PREMIUM = 0.25   # Anthropic 5-min cache write = 1.25x base input price


def cache_storage_frac(storage_usd_per_m_hr: float, ttl_hours: float, price_in_per_m: float) -> float:
    """Storage-billed caches (Gemini-style) as a fraction of the prefix's input price.

    Storage is a flat $/1M-token-hour, so on a cheap model it is a big fraction of
    the input price — caching a small model needs far more reads to pay off.
    """
    if price_in_per_m <= 0:
        return float("inf")
    return storage_usd_per_m_hr * ttl_hours / price_in_per_m


def cache_break_even_reads(
    write_premium: float = CACHE_WRITE_PREMIUM,
    read_discount: float = 0.10,
    storage_cost_frac: float = 0.0,
) -> float:
    """Reads per cached prefix at which caching breaks even.

    Writing a prefix costs (write_premium + storage_cost_frac) x its input price
    once; each read saves (1 - read_discount) x that price. Anthropic defaults:
    0.25 / 0.90 = 0.28 reads.
    """
    saving_per_read = 1.0 - read_discount
    if saving_per_read <= 0:
        return float("inf")
    return (max(0.0, write_premium) + max(0.0, storage_cost_frac)) / saving_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_premium: float = CACHE_WRITE_PREMIUM,
    read_discount: float = 0.10,
    storage_cost_frac: float = 0.0,
) -> bool:
    """Caching saves money only if a prefix is re-read more often than break-even."""
    return avg_cache_reads > cache_break_even_reads(write_premium, read_discount, storage_cost_frac)


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Per-hour spot preemption probability by GPU type — illustrative 2026 snapshot.
# Flagship training parts (H100/H200) sit in large, deep pools and are reclaimed
# less often; small inference parts (A10G/L4) are oversubscribed and churn more.
SPOT_INTERRUPT_RATE = {
    "H100": 0.03, "H200": 0.03, "B200": 0.04, "A100": 0.05,
    "MI300X": 0.06, "A10G": 0.08, "L4": 0.08,
}
DEFAULT_SPOT_INTERRUPT_RATE = 0.05


def reserved_effective_hr(rate_hr: float, term_months: int, horizon_months: float) -> float:
    """Effective $/hr of a commitment when you only need it for `horizon_months`.

    A reservation is billed for its whole term whether you use it or not, so a
    3yr commitment for a service that lives 12 months costs 3x its sticker rate.
    """
    used = min(float(term_months), max(0.0, horizon_months))
    return rate_hr * term_months / used if used > 0 else float("inf")


def tier_costs(
    hours_per_day: float,
    interruptible: bool,
    prices: dict,
    gpu_type: str | None = None,
    job_days: float | None = None,
    horizon_months: float = 36,
    num_gpus: int = 1,
    days: int = 30,
    spot_interrupt_rate: float | None = None,
) -> dict:
    """Monthly cost of every *eligible* tier for one workload.

    - on_demand : pay only for the hours you run.
    - spot      : only if interruptible; priced with the checkpoint/rework sim at
                  the GPU type's preemption rate.
    - reserved  : only if the workload is persistent (job_days >= 30 or unknown);
                  billed 24h/day (the commitment does not pause when you idle) at
                  the horizon-amortized 1yr / 3yr rate.
    """
    hpd = max(0.0, min(24.0, hours_per_day))
    run_hours = hpd * days * num_gpus
    commit_hours = 24.0 * days * num_gpus
    od = float(prices["on_demand_hr"])
    costs = {"on_demand": run_hours * od}
    if interruptible and run_hours > 0:
        rate = spot_interrupt_rate
        if rate is None:
            rate = SPOT_INTERRUPT_RATE.get(gpu_type, DEFAULT_SPOT_INTERRUPT_RATE)
        sim = spot_checkpoint_cost(run_hours, float(prices["spot_hr"]), od, interrupt_rate=rate)
        costs["spot"] = sim["spot_cost"]
    if job_days is None or job_days >= 30:
        for term, key in ((12, "reserved_1yr_hr"), (36, "reserved_3yr_hr")):
            if key in prices:
                eff = reserved_effective_hr(float(prices[key]), term, horizon_months)
                costs[f"reserved_{term // 12}yr"] = commit_hours * eff
    return costs


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    prices: dict | None = None,
    horizon_months: float = 36,
    spot_interrupt_rate: float | None = None,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Without `prices` this is the DOCUMENTED simple policy (v1):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    With a catalog row in `prices` it is the cost-based policy (v2): price every
    eligible tier via `tier_costs` (per-GPU spot interruption rate, 24h/day
    reserved billing, 1yr vs 3yr amortized over `horizon_months`) and return the
    cheapest — 'on_demand', 'spot', 'reserved_1yr' or 'reserved_3yr'. Ties go to
    on_demand, which keeps the option to walk away.
    """
    if prices is not None:
        costs = tier_costs(hours_per_day, interruptible, prices, gpu_type=gpu_type,
                           job_days=job_days, horizon_months=horizon_months,
                           spot_interrupt_rate=spot_interrupt_rate)
        return min(costs, key=costs.get)
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
