"""
FlexSys – Electricity Market Optimisation Web App
"""

from __future__ import annotations

import io
import base64
import json
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from flask import Flask, render_template, request, jsonify

from model import default_inputs, solve

app = Flask(__name__)

sns.set_theme(style="whitegrid", palette="muted")

BLOCK_COLORS = {"Winter Peak": "#3B82F6", "Shoulder": "#F59E0B", "Summer": "#10B981"}
SUB_ALPHA = {0: 0.55, 1: 0.90}  # lo sub-block darker / hi sub-block lighter


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    defaults = default_inputs()
    return render_template("index.html", defaults=json.dumps(defaults))


@app.route("/optimise", methods=["POST"])
def optimise():
    try:
        inputs = request.get_json(force=True)
        results = solve(inputs)

        if results.get("error"):
            return jsonify(results), 400

        cur = inputs.get("currency", "$")
        plots = generate_plots(inputs, results, cur)
        results["plots"] = plots
        return jsonify(results)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": True, "status": str(exc)}), 500


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def generate_plots(inputs: dict, results: dict, cur: str = "$") -> dict:
    plots = {}
    plots["ldc"] = _plot_ldc(inputs, results, cur)
    plots["capacity"] = _plot_capacity(results, cur)
    plots["dispatch"] = _plot_dispatch(inputs, results, cur)
    plots["storage"] = _plot_storage(inputs, results)
    return plots


def _plot_ldc(inputs: dict, results: dict, cur: str = "$") -> str:
    """LDC with sub-blocks sorted by price (descending) to form a proper curve."""
    subs = results["sub_blocks"]

    sorted_subs = sorted(subs, key=lambda s: s["price"], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    cum_hours = 0
    for sb in sorted_subs:
        h = sb["hours"]
        total = sb["total_served_gw"]
        b = sb["block"]
        idx = sb["sub_idx"]
        color = BLOCK_COLORS.get(b, "#6366F1")
        alpha = SUB_ALPHA.get(idx, 0.8)

        ax.barh(
            y=0, width=h, left=cum_hours, height=total,
            color=color, edgecolor="white", linewidth=0.8, alpha=alpha,
            align="edge",
        )

        mid_x = cum_hours + h / 2
        short_lbl = sb["label"].replace("ZVC", "").strip()
        ax.text(mid_x, total / 2,
                f"{b}\n({short_lbl})\n{total:.1f} GW\n{cur}{sb['price']:,.0f}/MWh",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.7))

        cum_hours += h

    ax.set_xlabel("Cumulative Hours", fontsize=11)
    ax.set_ylabel("Served Demand (GW)", fontsize=11)
    ax.set_title("Load Duration Curve with Sub-Block Clearing Prices",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, cum_hours)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    sns.despine(left=True, bottom=True)

    return _fig_to_base64(fig)


def _plot_capacity(results: dict, cur: str = "$") -> str:
    caps = results["capacities"]
    costs = results["annual_capital"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    techs = ["ZVC", "Gas", "Storage"]
    cap_vals = [caps["zvc"], caps["gas"], caps["storage_power"]]
    colors = ["#10B981", "#EF4444", "#3B82F6"]

    bars = ax1.bar(techs, cap_vals, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, cap_vals):
        if val > 0.01:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Capacity (GW)")
    ax1.set_title("Optimal Power Capacity", fontsize=12, fontweight="bold")

    cost_labels = ["ZVC", "Gas", "Storage"]
    cost_vals = [costs["zvc"], costs["gas"],
                 costs["storage_power"] + costs["storage_energy"]]
    nonzero = [(l, v, c) for l, v, c in zip(cost_labels, cost_vals, colors) if v > 0]
    if nonzero:
        labels_nz, vals_nz, colors_nz = zip(*nonzero)
        total_nz = sum(vals_nz)
        wedges, texts, autotexts = ax2.pie(
            vals_nz, labels=labels_nz, colors=colors_nz,
            autopct=lambda p: f"{cur}{p * total_nz / 100:,.0f}" if p > 3 else "",
            startangle=90, textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
    else:
        ax2.text(0.5, 0.5, "No capacity built", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="grey")
    ax2.set_title("Annual Capital Cost Split", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_dispatch(inputs: dict, results: dict, cur: str = "$") -> str:
    """2-row × 3-col grid: each block is a column, lo sub-block on top, hi on bottom."""
    blocks = inputs["blocks"]
    subs = results["sub_blocks"]

    sub_lookup = {}
    for sb in subs:
        sub_lookup[sb["block"], sb["sub_idx"]] = sb

    # Find global y-max across all sub-blocks
    y_max = 0
    for sb in subs:
        dem_total = sum(t["served"] for t in sb["demand_tiers"])
        dem_total += sb["shifted_in_total"] + sb["expandable"]["activated"]
        sup_total = sb["supply"]["zvc"] + sb["supply"]["gas"] + sb["supply"]["storage_discharge"]
        y_max = max(y_max, dem_total, sup_total)
    y_max = y_max * 1.1  # 10% headroom

    n_cols = len(blocks)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10), sharey=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, b in enumerate(blocks):
        for row in range(2):
            ax = axes[row, col]
            sb = sub_lookup[b, row]

            demand_labels, demand_vals = [], []
            for t in sb["demand_tiers"]:
                demand_labels.append(t["name"])
                demand_vals.append(t["served"])
            if sb["shifted_in_total"] > 1e-6:
                demand_labels.append("Shifted In")
                demand_vals.append(sb["shifted_in_total"])
            if sb["expandable"]["activated"] > 1e-6:
                demand_labels.append("Expandable")
                demand_vals.append(sb["expandable"]["activated"])

            supply_labels = ["ZVC", "Gas", "Storage"]
            supply_vals = [
                sb["supply"]["zvc"],
                sb["supply"]["gas"],
                sb["supply"]["storage_discharge"],
            ]

            demand_colors = ["#6366F1", "#8B5CF6", "#A78BFA", "#C4B5FD", "#DDD6FE"]
            supply_colors = ["#10B981", "#EF4444", "#3B82F6"]

            _stacked_bar(ax, 0, demand_vals, demand_labels,
                         demand_colors[:len(demand_vals)])
            _stacked_bar(ax, 1, supply_vals, supply_labels,
                         supply_colors[:len(supply_vals)])

            ax.set_ylim(0, y_max)
            ax.set_title(
                f"{b}\n{sb['label']} ({sb['hours']:.0f}h, {sb['zvc_availability']:.0f}% ZVC)"
                f"\n{cur}{sb['price']:,.0f}/MWh",
                fontsize=10, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Demand", "Supply"])
            if col == 0:
                ax.set_ylabel("GW")

    handles_all, labels_all = [], []
    for ax_flat in axes.flat:
        h, l = ax_flat.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels_all:
                handles_all.append(hi)
                labels_all.append(li)

    fig.legend(handles_all, labels_all, loc="lower center",
               ncol=min(len(labels_all), 6), fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Demand & Supply by Sub-Block",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _stacked_bar(ax, x_pos, vals, labels, colors):
    bottom = 0
    for val, label, color in zip(vals, labels, colors):
        if val < 1e-6:
            continue
        ax.bar(x_pos, val, bottom=bottom, width=0.6,
               color=color, edgecolor="white", linewidth=0.5,
               label=label)
        if val > 0.5:
            ax.text(x_pos, bottom + val / 2, f"{val:.1f}",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
        bottom += val


def _plot_storage(inputs: dict, results: dict) -> str:
    """Storage activity across all 6 sub-blocks in sequence."""
    subs = results["sub_blocks"]
    caps = results["capacities"]

    labels = [f"{sb['block'][:3]}\n{sb['label'][:3]}" for sb in subs]
    x = np.arange(len(subs))

    charge_vals = [sb["supply"]["storage_charge"] for sb in subs]
    discharge_vals = [sb["supply"]["storage_discharge"] for sb in subs]
    level_vals = [sb["storage"]["level"] for sb in subs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    bar_w = 0.35
    ax1.bar(x - bar_w / 2, charge_vals, bar_w, label="Charge (GW)",
            color="#3B82F6", edgecolor="white")
    ax1.bar(x + bar_w / 2, discharge_vals, bar_w, label="Discharge (GW)",
            color="#EF4444", edgecolor="white")
    ax1.set_ylabel("GW")
    ax1.set_title("Storage Power Activity by Sub-Block", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    sto_cap = caps["storage_energy"]
    ax2.plot(x, level_vals, "o-", color="#8B5CF6", linewidth=2, markersize=8)
    ax2.fill_between(x, level_vals, alpha=0.2, color="#8B5CF6")
    if sto_cap > 0.01:
        ax2.axhline(sto_cap, ls="--", color="grey", alpha=0.5,
                    label=f"Capacity ({sto_cap:,.0f} GWh)")
    ax2.set_ylabel("GWh")
    ax2.set_title("Storage Energy Level", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    if sto_cap > 0.01:
        ax2.legend(fontsize=9)

    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
