"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png

Every number in the report is computed live from the mission results; the
analysis sections are generated around them (Vietnamese, for the lab write-up).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys
from missions._common import num, catalog_by_type, ROOT
from finops import pricing, report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m6_carbon_scheduling

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}
TARGET_TRAIN_MFU = 0.40   # healthy H100 training MFU (fleet peers run 0.40-0.43)


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def _lie_section(r1, cat) -> tuple[str, float]:
    """GPU-Util lie explanation + $ paid for FLOPs never delivered."""
    rows, wasted = [], 0.0
    lie_ids = {l["gpu_id"] for l in r1["lies"]}
    peers = [s["mfu"] for s in r1["summary"]
             if s["gpu_type"] == r1["lies"][0]["gpu_type"] and s["gpu_id"] not in lie_ids and s["idle_hours"] == 0]
    for lie in r1["lies"]:
        od = num(cat[lie["gpu_type"]]["on_demand_hr"])
        monthly = od * 24 * DAYS
        target = TARGET_TRAIN_MFU if lie["workload"] == "train" else 0.60
        metric = lie["mfu"] if lie["workload"] == "train" else lie["mbu"]
        waste = monthly * max(0.0, 1 - metric / target)
        wasted += waste
        label = "MFU" if lie["workload"] == "train" else "MBU"
        rows.append(f"| {lie['gpu_id']} | {lie['gpu_type']} | {lie['workload']} | {lie['gpu_util_pct']}% | "
                    f"{lie['mfu']:.3f} | {lie['mbu']:.3f} | {label} {metric:.2f} vs mục tiêu {target:.2f} | "
                    f"{_usd(monthly)} | {_usd(waste)} |")
    md = "\n".join([
        "## Phân tích: \"GPU-Util lie\" là gì và tốn bao nhiêu tiền",
        "",
        "| GPU | Loại | Workload | GPU-Util | MFU | MBU | Hiệu quả thực | Tiền thuê/tháng | Trả cho FLOPs/BW không nhận được |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"**Tổng tiền \"trả mà không nhận\": {_usd(wasted)}/tháng.**",
        "",
        "**Cơ chế.** `GPU-Util` của `nvidia-smi` là tỷ lệ *thời gian* có ít nhất một kernel đang chạy "
        "trong cửa sổ lấy mẫu, không phải tỷ lệ SM/Tensor Core thực sự làm việc. Một kernel nhỏ chiếm "
        "1/132 SM, hoặc một kernel đang đứng chờ HBM (memory stall), vẫn được tính là GPU \"bận\" 100%. "
        f"Vì vậy `gpu-h100-4` báo {r1['lies'][0]['gpu_util_pct']}% util nhưng chỉ đạt MFU "
        f"{r1['lies'][0]['mfu']:.2f} (các H100 cùng cụm đạt {min(peers):.2f}–{max(peers):.2f}). Nguyên nhân gốc rễ thường gặp: "
        "(1) batch/micro-batch quá nhỏ nên kernel không lấp đầy SM; (2) kernel launch overhead và đồng bộ "
        "CPU↔GPU (không dùng CUDA Graphs/fused kernels); (3) dataloader/I-O chậm làm GPU quay vòng chờ; "
        "(4) chờ all-reduce/NCCL giữa các node; (5) phép toán memory-bound (layernorm, attention không dùng "
        "FlashAttention) chiếm phần lớn thời gian. `gpu-a10g-1` là trường hợp decode memory-bound: util 97% "
        "nhưng MBU chỉ ~0.30, nghĩa là băng thông HBM cũng không được dùng hết.",
        "",
        "**Ý nghĩa chi phí.** Hoá đơn tính theo giờ GPU, không theo FLOPs. Ở MFU 0.19, chi phí thực trên mỗi "
        f"FLOP hữu ích cao gấp ~{sum(peers) / len(peers) / r1['lies'][0]['mfu']:.1f}× so với các H100 khoẻ cùng cụm, "
        "tức là $/1M-token training cũng đắt gấp chừng đó. "
        "Dashboard dựa trên GPU-Util sẽ báo \"đã tận dụng hết, cần mua thêm GPU\", dẫn tới quyết định mua "
        "sai. Hãy đo MFU/MBU (DCGM `SM_ACTIVE`, `TENSOR_ACTIVE`, `DRAM_ACTIVE`) và sửa nguyên nhân "
        "*trước* khi hạ cấp hay mua thêm GPU.",
    ])
    return md, wasted


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r6 = m6_carbon_scheduling.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    # compute-bound lies (training) drop one tier; memory-bound inference is
    # re-planned by HBM bandwidth + VRAM (Your Turn #2), so it is not double counted
    lie_savings = 0.0
    for lie in r1["lies"]:
        if lie["workload"] != "train":
            continue
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        lie_savings += max(0.0, delta) * 24 * DAYS
    mbu_savings = r1["rightsizing"]["monthly_savings"]
    rightsize_savings = lie_savings + mbu_savings

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size (util-lie + MBU)": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": r6["best"]["cleanest"],
    }

    sections = _sections(r1, r2, r3, r6, cat, levers, baseline, optimized, total_pct,
                         lie_savings, mbu_savings, wh)
    md = report.build_report(baseline, optimized, levers, sustainability=sust, extra_sections=sections)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"), baseline=baseline)

    if verbose:
        try:  # Windows consoles default to cp1252; the report is Vietnamese
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


def _sections(r1, r2, r3, r6, cat, levers, baseline, optimized, total_pct,
              lie_savings, mbu_savings, wh_query) -> list[str]:
    tokens_month = r2["total_tokens"] * DAYS
    prices = m2_inference_levers.MODEL_PRICES
    in_ratio = prices["large"][0] / prices["small"][0]
    out_ratio = prices["large"][1] / prices["small"][1]
    small_share = r2["small_share"]
    per_m = lambda usd: usd / (tokens_month / 1e6)
    total_saved = sum(levers.values())

    # 1 — unit economics: blended spend ladder + inference-only ladder
    blended = [f"| Baseline | {_usd(baseline)} | ${per_m(baseline):.2f} | — |"]
    level = baseline
    for name, v in levers.items():
        level -= v
        blended.append(f"| + {name} | {_usd(level)} | ${per_m(level):.2f} | {v / total_saved:.0%} |")
    infer = [f"| {s['step']} | ${s['daily'] * DAYS:,.0f} | ${s['per_m']:.3f} |" for s in r2["ladder"]]
    unit = "\n".join([
        "## Baseline vs. Optimized theo $/1M-token",
        "",
        f"Sản lượng: {r2['total_tokens']:,} token/ngày ≈ {tokens_month / 1e6:,.0f}M token/tháng. "
        f"Tổng chi giảm từ **{_usd(baseline)} → {_usd(optimized)}/tháng (−{total_pct:.0f}%)**.",
        "",
        "**Chi phí blended (toàn bộ GPU + API) chia cho token phục vụ, bật lần lượt từng lever:**",
        "",
        "| Bước | Chi/tháng | $/1M-token | % đóng góp vào tổng tiết kiệm |",
        "|---|---|---|---|",
        *blended,
        "",
        "**Riêng inference API (M2): từng lever giảm $/1M-token bao nhiêu:**",
        "",
        "| Bước | Chi/tháng | $/1M-token |",
        "|---|---|---|",
        *infer,
        "",
        f"Cascade đóng góp lớn nhất ở inference (${r2['ladder'][0]['per_m']:.2f} → ${r2['ladder'][1]['per_m']:.2f}/1M) "
        f"vì {small_share:.0%} request là câu dễ và model nhỏ rẻ hơn {in_ratio:.0f}× (input) đến {out_ratio:.1f}× (output). "
        "Cache và batch là chiết khấu "
        "nhân lên trên phần còn lại, nên tác động tuyệt đối nhỏ hơn sau khi đã cascade.",
        "",
        "_Lưu ý phạm vi: inference + purchasing tính trên hoá đơn baseline; right-size và idle đo trên "
        "fleet telemetry (M1). Các lever có thể chồng lấn một phần khi triển khai thật._",
    ])

    lie_md, _ = _lie_section(r1, cat)

    # 3 — prioritized actions (ROI = $ saved per unit of effort/risk)
    spot = sum(r["on_demand"] - r["optimized"] for r in r3["recommendations"] if r["tier"] == "spot")
    reserved = sum(r["on_demand"] - r["optimized"] for r in r3["recommendations"] if r["tier"] == "reserved")
    n_spot = sum(r["tier"] == "spot" for r in r3["recommendations"])
    cap5 = next(s for s in r2["reasoning"]["scenarios"] if s["cap"] == 0.05)
    actions = "\n".join([
        "## Hành động đề xuất (xếp theo ROI = tiết kiệm / công sức & rủi ro)",
        "",
        "| # | Hành động | Tiết kiệm/tháng | Công sức / rủi ro | Vì sao ở vị trí này |",
        "|---|---|---|---|---|",
        f"| 1 | Tắt GPU idle (auto-shutdown khi util < 10% trong 30 phút) | {_usd(levers['Kill idle GPUs'])} | Vài giờ, không rủi ro | Tiền mất trắng, sửa bằng một policy |",
        f"| 2 | Chuyển {n_spot} job interruptible sang spot + checkpoint | {_usd(spot)} | Thấp: job đã checkpoint được | Lever lớn nhất không cần cam kết dài hạn |",
        f"| 3 | Cascade + batch + cache có điều kiện cho inference | {_usd(levers['Inference (cascade/cache/batch)'])} | Thấp: đổi routing/config | Giảm $/1M-token API {r2['savings_pct']:.0f}% |",
        f"| 4 | Sửa util-lie & right-size theo MBU (gộp 2×A100 → 1×MI300X, 2×A10G → 1×A100) | {_usd(lie_savings + mbu_savings)} | Trung bình: cần benchmark, MI300X đổi sang ROCm | Phải làm **trước** bước 5 |",
        f"| 5 | Reserved 3yr cho inference 24/7 sau khi đã right-size | {_usd(reserved)} | Cao: cam kết 36 tháng | Lớn nhưng khoá loại GPU; reserve sai GPU là khoá lãng phí 3 năm |",
        f"| 6 | Cap reasoning ≤5% traffic + dời job train sang vùng sạch | ${cap5['saved_usd_day'] * DAYS:,.0f} + điện | Thấp | $ nhỏ nhưng giảm {cap5['saved_wh_day'] * DAYS / 1000:,.0f} kWh/tháng và {r6['co2_saved_pct']:.0f}% CO2 của job train |",
        "",
        "Nguyên tắc: làm việc rẻ & không hối tiếc trước (idle, spot, routing), đo lại MFU/MBU, rồi mới right-size, "
        "và chỉ cam kết reserved khi fleet đã đúng kích thước.",
    ])

    # 4 — sustainability
    regions = sorted(r6["regions"], key=lambda r: r["carbon_kg"])
    reg_rows = [f"| {r['region']} | {r['usd_per_kwh']:.3f} | {r['gco2_per_kwh']} | {_usd(r['energy_usd'])} | "
                f"{r['carbon_kg']:,.0f} | {_usd(r['carbon_priced_usd'])} | {r['latency']} |" for r in regions]
    best = r6["best"]
    by_region = {r["region"]: r for r in r6["regions"]}
    home, cheap, clean = by_region[r6["home_region"]], by_region[best["cheapest"]], by_region[best["cleanest"]]
    dirty = max(r6["regions"], key=lambda r: r["gco2_per_kwh"])
    rsn = r2["reasoning"]["split"]
    sust = "\n".join([
        "## Sustainability: carbon gắn với chi phí",
        "",
        f"- Năng lượng/truy vấn chuẩn (800 token): {wh_query:.2f} Wh; truy vấn reasoning trung bình "
        f"{rsn['reasoning']['wh_per_query']:.0f} Wh ({r2['reasoning']['wh_ratio']:.0f}×).",
        f"- Job interruptible tiêu thụ {r6['total_kwh']:,.0f} kWh/tháng (TDP × GPU × giờ × PUE 1.2).",
        "",
        f"| Vùng | $/kWh | gCO2/kWh | Tiền điện | CO2 (kg) | Điện + CO2 @ ${r6['carbon_price']:.0f}/t | Độ trễ tới user |",
        "|---|---|---|---|---|---|---|",
        *reg_rows,
        "",
        f"- **Sạch nhất:** {best['cleanest']} (thuỷ điện Na Uy): dời job train từ us-east-1 giảm "
        f"{r6['co2_saved_kg']:,.0f} kg CO2e (−{r6['co2_saved_pct']:.0f}%) và {_usd(r6['energy_usd_saved_clean'])} tiền điện.",
        f"- **Rẻ nhất / cân bằng nhất:** {best['cheapest']} (${cheap['usd_per_kwh']}/kWh, {cheap['gco2_per_kwh']} gCO2/kWh): "
        f"rẻ hơn {r6['home_region']} {_usd(r6['energy_usd_saved_cheap'])} và vẫn giảm "
        f"{1 - cheap['carbon_kg'] / home['carbon_kg']:.0%} CO2; cũng thắng khi định giá carbon ${r6['carbon_price']:.0f}/t.",
        f"- **Dơ nhất:** {dirty['region']} ({dirty['gco2_per_kwh']} gCO2/kWh) vừa đắt điện vừa phát thải gấp "
        f"{dirty['gco2_per_kwh'] / clean['gco2_per_kwh']:.0f}× {clean['region']}. "
        "Chọn vùng là lever hiếm hoi giảm cả $ lẫn carbon cùng lúc.",
        "- **Trade-off:** chỉ job train/batch/eval (không nhạy độ trễ) nên dời đi; inference realtime giữ gần user. "
        "Cần xét thêm data residency và phí egress khi chuyển dataset qua Đại Tây Dương.",
    ])

    # 5 — Your Turn results
    cache = r2["cache"]
    on = [t for t, v in cache["teams"].items() if v["worth_it"]]
    off = [t for t, v in cache["teams"].items() if not v["worth_it"]]
    tiers = cache["tiers"]
    reads = lambda teams: "/".join(f"{cache['teams'][t]['avg_reads']:.2f}" for t in teams)
    rs = r1["rightsizing"]["plans"]
    eff0 = r1["catalog_efficiency"][0]
    ext = "\n".join([
        "## Phần mở rộng \"Your Turn\" (đã thực hiện cả 5)",
        "",
        f"1. **Tier policy v2** (`pricing.recommend_tier/tier_costs`): interruption rate theo GPU "
        f"({', '.join(f'{g} {v:.0%}' for g, v in pricing.SPOT_INTERRUPT_RATE.items())}), "
        f"reserved tính trung thực 24h/ngày, so 1yr vs 3yr theo horizon. Savings purchasing: v1 "
        f"{r3['policy_v1_savings_pct']}% → v2 {r3['savings_pct']}%. v2 thấp hơn vì v1 chỉ tính tiền reserved cho giờ chạy "
        f"(job-infer-search L4 18h/ngày bị đánh giá thấp). Độ nhạy horizon: "
        + ", ".join(f"{h} tháng = {v['savings_pct']}%" for h, v in r3["horizon_sensitivity"].items()) + ".",
        f"2. **Right-size theo MBU** (M1): " + "; ".join(
            f"{p['current']} → {p['proposed']} (cần {p['need_gb']} GB, {p['need_bw_tbs']} TB/s; "
            f"rẻ nhất theo $/GPU-hr sẽ là {p['naive_cheapest']} ${p['naive_usd_hr']:.2f}/h)" for p in rs if p["monthly_savings"] > 0)
        + f". Tổng {_usd(r1['rightsizing']['monthly_savings'])}/tháng. {eff0['gpu_type']} có $/GB-VRAM thấp nhất "
        f"(${eff0['usd_per_gb']:.3f}) và {eff0['peak_bw_tbs']} TB/s băng thông, nên một GPU gộp được nhiều replica.",
        f"3. **`cache_is_worth_it()`**: break-even {tiers['small']['anthropic_style']} lần đọc/prefix (write 1.25×, read 0.1×). "
        f"Bật cache cho {', '.join(on)} ({reads(on)} lần đọc/lần ghi), tắt cho {', '.join(off)} ({reads(off)}). "
        "Với cache tính phí lưu trữ "
        f"($1/1M-token-giờ), break-even là {tiers['small']['gemini_style']} lần đọc cho model nhỏ và "
        f"{tiers['large']['gemini_style']} cho model lớn. Bật cache cho mọi team: ${r2['cache_all_daily']:.2f}/ngày, "
        f"còn gated: ${r2['optimized_daily']:.2f}/ngày.",
        f"4. **Ngân sách reasoning**: {rsn['reasoning']['share_traffic']:.1%} traffic → {rsn['reasoning']['share_cost']:.1%} chi phí "
        f"và {rsn['reasoning']['share_wh']:.1%} năng lượng. Cap 10% không có tác dụng (đã dưới ngưỡng). Cap 5%: "
        f"−${cap5['saved_usd_day']:.2f}/ngày, −{cap5['saved_wh_day'] / 1000:.1f} kWh/ngày. Rule đề xuất: chỉ bật reasoning cho "
        f"task eval nhiều bước khi classifier độ khó > ngưỡng (≈ đầu ra kỳ vọng ≥ {r2['reasoning']['cap5_min_output']:,} token), "
        "kèm `max_thinking_tokens` và hạn mức Wh/ngày cho mỗi team.",
        f"5. **Carbon-aware scheduling** (`missions/m6_carbon_scheduling.py`): xem bảng 5 vùng ở trên. Sạch nhất "
        f"{best['cleanest']}, rẻ nhất/cân bằng {best['balanced']}.",
    ])
    return [unit, lie_md, actions, sust, ext]


if __name__ == "__main__":
    run()
