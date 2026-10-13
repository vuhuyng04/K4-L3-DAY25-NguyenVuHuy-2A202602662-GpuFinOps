"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 extra_sections: list[str] | None = None) -> str:
    """Return a markdown cost-optimization report (extra markdown sections appended)."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Best region (cleanest grid): {sustainability.get('best_region', 'n/a')}",
        ]
    for section in extra_sections or []:
        lines += ["", section.rstrip()]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline: float | None = None) -> str:
    """Write a savings chart PNG. Returns the path. No-op if matplotlib absent.

    With `baseline`, draws a true waterfall: baseline -> one step down per lever
    -> optimized. Without it, a plain bar chart of savings per lever.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    if baseline is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(names, vals, color="#2e548a")
        ax.set_ylabel("Savings (USD / month)")
        ax.set_title("GPU cost savings by FinOps lever")
        plt.xticks(rotation=20, ha="right")
    else:
        optimized = baseline - sum(vals)
        labels = ["Baseline"] + names + ["Optimized"]
        bottoms, heights, colors = [0.0], [baseline], ["#5b6770"]
        level = baseline
        for v in vals:
            level -= v
            bottoms.append(level)
            heights.append(v)
            colors.append("#2e8a57")
        bottoms.append(0.0)
        heights.append(optimized)
        colors.append("#2e548a")
        fig, ax = plt.subplots(figsize=(10, 5.2))
        bars = ax.bar(labels, heights, bottom=bottoms, color=colors)
        for i, (b, h) in enumerate(zip(bottoms, heights)):
            text = f"${h:,.0f}" if i in (0, len(labels) - 1) else f"-${h:,.0f}"
            ax.text(bars[i].get_x() + bars[i].get_width() / 2, b + h, text,
                    ha="center", va="bottom", fontsize=9)
        pct = sum(vals) / baseline * 100 if baseline else 0.0
        ax.set_ylabel("USD / month")
        # "\\$" keeps matplotlib from reading a $...$ pair as mathtext
        ax.set_title(f"GPU cost waterfall: \\${baseline:,.0f} -> \\${optimized:,.0f} per month (-{pct:.0f}%)")
        ax.set_ylim(0, baseline * 1.1)
        plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
