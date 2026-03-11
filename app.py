"""
Peaky Flexers – Electricity Market Optimisation Web App
"""

from __future__ import annotations

import io
import base64
import json
import traceback
import datetime
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from flask import Flask, render_template, request, jsonify

from pathlib import Path

from model import default_inputs, solve

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

_MODEL_SOURCE = Path(__file__).with_name("model.py").read_text(encoding="utf-8")


@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": True, "reply": "Request too large."}), 413


sns.set_theme(style="whitegrid", palette="muted")

BLOCK_COLORS = {"Winter Peak": "#3B82F6", "Shoulder": "#F59E0B", "Low Demand": "#10B981"}
SUB_ALPHA = {0: 0.55, 1: 0.90}  # lo sub-block darker / hi sub-block lighter


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    defaults = default_inputs()
    return render_template("index.html", defaults=json.dumps(defaults))


@app.route("/guide")
def guide():
    return render_template("guide.html")


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


CHAT_MODELS = {
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-6": "Claude Opus 4.6",
}

DEFAULT_CHAT_MODEL = "claude-opus-4-6"

_CHAT_LOG = Path(__file__).with_name("chat_log.txt")
_log_lock = threading.Lock()


def _log_chat(ip: str, model: str, message: str, history: list, reply: str):
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    is_new = len(history) == 0
    lines = [f"--- {'NEW CHAT' if is_new else 'CONTINUATION'} | {ts} | IP: {ip} | Model: {model} ---"]
    if is_new:
        lines.append("")
    lines.append(f"USER: {message}")
    lines.append(f"ASSISTANT: {reply}")
    lines.append("")
    with _log_lock:
        with open(_CHAT_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        import anthropic

        data = request.get_json(force=True)
        api_key = data.get("api_key", "").strip()
        if not api_key:
            return jsonify({"error": True, "reply": "Please enter your Anthropic API key."}), 400

        model_id = data.get("model", DEFAULT_CHAT_MODEL)
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": True, "reply": "Please enter a message."}), 400

        history = data.get("history", [])
        current_inputs = data.get("current_inputs")
        current_results = data.get("current_results")

        system_prompt = _build_system_prompt(current_inputs, current_results)

        messages = []
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )

        reply = resp.content[0].text if resp.content else "No response."

        ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
        _log_chat(ip, model_id, message, history, reply)

        return jsonify({"error": False, "reply": reply})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": True, "reply": f"Error: {exc}"}), 500


@app.route("/chat_models")
def chat_models():
    return jsonify({"models": CHAT_MODELS, "default": DEFAULT_CHAT_MODEL})


def _build_system_prompt(current_inputs, current_results):
    parts = [
        "You are an expert assistant embedded in the Peaky Flexers electricity market "
        "LP optimisation tool. You help users understand the model, its results, and "
        "the economics of electricity markets.",
        "",
        "## Model Source Code (model.py)",
        "```python",
        _MODEL_SOURCE,
        "```",
    ]

    if current_inputs:
        inputs_str = json.dumps(current_inputs, indent=2, default=str)
        parts += [
            "",
            "## Current Input Values",
            "```json",
            inputs_str,
            "```",
        ]

    if current_results:
        results_summary = {k: v for k, v in current_results.items()
                          if k != "plots"}
        results_str = json.dumps(results_summary, indent=2, default=str)
        parts += [
            "",
            "## Current Optimisation Results (plots excluded)",
            "```json",
            results_str,
            "```",
        ]

    parts += [
        "",
        "## Guidelines",
        "- Answer questions about the model mechanics, economics, and results.",
        "- Reference specific variables, constraints, and line numbers from the model code when relevant.",
        "- Use the current inputs and results to give specific, quantitative answers.",
        "- If the user asks what would happen with different parameters, explain the economic logic.",
        "- Be concise but thorough. Use markdown formatting.",
    ]

    return "\n".join(parts)


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
    plots["input_ldc"] = _plot_input_ldc(inputs)
    plots["ldc"] = _plot_ldc(inputs, results, cur)
    plots["price_duration"] = _plot_price_duration(inputs, results, cur)
    plots["capacity"] = _plot_capacity(results, cur)
    plots["dispatch"] = _plot_dispatch(inputs, results, cur)
    plots["dispatch_gwh"] = _plot_dispatch_gwh(inputs, results, cur)
    plots["storage"] = _plot_storage(inputs, results)
    flex = _plot_flexibility(inputs, results)
    if flex:
        plots["flexibility"] = flex
    eflow = _plot_energy_flows(inputs, results)
    if eflow:
        plots["energy_flows"] = eflow
    return plots


GEN_COLORS = {"Renewables": "#10B981", "Gas": "#EF4444", "Storage": "#3B82F6"}


INPUT_TIER_COLORS = {"High": "#4C1D95", "Mid": "#7C3AED", "Low": "#C4B5FD"}


def _plot_input_ldc(inputs: dict) -> str:
    """Input LDC — raw demand blocks with tier breakdown."""
    from matplotlib.patches import Patch
    blocks = inputs["blocks"]
    hours = inputs["hours"]
    demand_tiers = inputs["demand_tiers"]
    zvc_profile = inputs["zvc_profile"]

    subs = []
    for b in blocks:
        tiers = demand_tiers[b]
        tier_total = sum(t["quantity"] for t in tiers)
        for sp in zvc_profile[b]:
            sh = hours[b] * sp["pct_hours"] / 100.0
            subs.append({"block": b, "label": sp["label"],
                         "hours": sh, "load": tier_total, "tiers": tiers})

    subs.sort(key=lambda s: s["load"], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    cum = 0
    for sb in subs:
        h = sb["hours"]
        block_color = BLOCK_COLORS.get(sb["block"], "#6366F1")

        # Stack tiers bottom-up: Low at bottom, High at top
        reversed_tiers = list(reversed(sb["tiers"]))
        y_bot = 0.0
        for t in reversed_tiers:
            gw = t["quantity"]
            if gw < 0.01:
                continue
            tc = INPUT_TIER_COLORS.get(t["name"], "#6366F1")
            ax.barh(y=y_bot, width=h, left=cum, height=gw,
                    color=tc, edgecolor="white", linewidth=0.4,
                    align="edge")
            if gw > 1.5:
                ax.text(cum + h / 2, y_bot + gw / 2,
                        f"{t['name']}\n{gw:.0f} GW",
                        ha="center", va="center", fontsize=7,
                        color="white", alpha=0.9)
            y_bot += gw

        ax.text(cum + h / 2, y_bot + 0.8,
                f"{sb['block']}\n({sb['label']})",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=block_color)
        cum += h

    ax.set_xlabel("Cumulative Hours", fontsize=11)
    ax.set_ylabel("Load (GW)", fontsize=11)
    ax.set_title("Input Load Duration Curve (before optimisation)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, cum)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    legend_handles = [Patch(fc=INPUT_TIER_COLORS[t], ec="white", label=f"{t}-value")
                      for t in ("High", "Mid", "Low")]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    sns.despine(left=True, bottom=True)
    return _fig_to_base64(fig)


def _plot_ldc(inputs: dict, results: dict, cur: str = "$") -> str:
    """LDC showing total generation stack per sub-block."""
    from matplotlib.patches import Patch
    subs = results["sub_blocks"]

    def _gen_total(s):
        sup = s["supply"]
        return sup["zvc"] + sup["gas"] + sup["storage_discharge"]

    sorted_subs = sorted(subs, key=_gen_total, reverse=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    cum_hours = 0
    has_curtailed = False
    for sb in sorted_subs:
        h = sb["hours"]
        curtailed = sum(t["lost"] for t in sb["demand_tiers"])
        b = sb["block"]
        sup = sb["supply"]
        gen = _gen_total(sb)

        stack = [
            ("Renewables", sup["zvc"]),
            ("Gas", sup["gas"]),
            ("Storage", sup["storage_discharge"]),
        ]

        y_bot = 0.0
        for tech, gw in stack:
            if gw < 0.01:
                continue
            ax.barh(
                y=y_bot, width=h, left=cum_hours, height=gw,
                color=GEN_COLORS[tech], edgecolor="white", linewidth=0.4,
                align="edge",
            )
            y_bot += gw

        if curtailed > 0.01:
            has_curtailed = True
            ax.barh(
                y=gen, width=h, left=cum_hours, height=curtailed,
                facecolor="none", edgecolor="#6B7280", linewidth=0.8,
                hatch="///", alpha=0.6, align="edge",
            )

        cum_hours += h

    ax.set_xlabel("Cumulative Hours", fontsize=11)
    ax.set_ylabel("Generation (GW)", fontsize=11)
    ax.set_title("Generation Duration Curve with Sub-Block Clearing Prices",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, cum_hours)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    legend_handles = [Patch(fc=GEN_COLORS[t], ec="white", label=t)
                      for t in ("Renewables", "Gas", "Storage")]
    if has_curtailed:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="#6B7280", hatch="///",
                  label="Curtailed"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    sns.despine(left=True, bottom=True)

    return _fig_to_base64(fig)


def _plot_price_duration(inputs: dict, results: dict, cur: str = "$") -> str:
    """Price Duration Curve — sub-blocks sorted by clearing price, high to low."""
    from matplotlib.patches import Patch

    subs = results["sub_blocks"]
    sorted_subs = sorted(subs, key=lambda s: s["price"], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    cum_hours = 0
    for sb in sorted_subs:
        h = sb["hours"]
        price = sb["price"]
        b = sb["block"]
        block_color = BLOCK_COLORS.get(b, "#6366F1")

        ax.barh(y=0, width=h, left=cum_hours, height=price,
                color=block_color, edgecolor="white", linewidth=0.4,
                align="edge", alpha=0.85)

        served_gw = sum(t["served"] for t in sb["demand_tiers"])
        shifted_out_gw = sum(
            sum(v for v in t["shifted_out"].values())
            for t in sb["demand_tiers"]
        )
        shifted_in_gw = sb["shifted_in_total"]
        curtailed_gw = sum(t["lost"] for t in sb["demand_tiers"])
        sto_dis = sb["supply"]["storage_discharge"]
        sto_chg = sb["supply"]["storage_charge"]

        cum_hours += h

    ax.set_xlabel("Cumulative Hours", fontsize=11)
    ax.set_ylabel(f"Clearing Price ({cur}/MWh)", fontsize=11)
    ax.set_title("Price Duration Curve", fontsize=13, fontweight="bold")
    ax.set_xlim(0, cum_hours)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{cur}{x:,.0f}"))

    blocks_in_chart = []
    seen = set()
    for sb in sorted_subs:
        if sb["block"] not in seen:
            seen.add(sb["block"])
            blocks_in_chart.append(sb["block"])
    legend_handles = [Patch(fc=BLOCK_COLORS.get(b, "#6366F1"), ec="white", label=b)
                      for b in blocks_in_chart]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    sns.despine(left=True, bottom=True)
    return _fig_to_base64(fig)


def _plot_capacity(results: dict, cur: str = "$") -> str:
    caps = results["capacities"]
    costs = results["annual_capital"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    techs = ["Renewables", "Gas", "Storage", "T&D"]
    cap_vals = [caps["zvc"], caps["gas"], caps["storage_power"], caps.get("td", 0)]
    colors = ["#10B981", "#EF4444", "#3B82F6", "#8B5CF6"]

    bars = ax1.bar(techs, cap_vals, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, cap_vals):
        if val > 0.01:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Capacity (GW)")
    ax1.set_title("Optimal Power Capacity", fontsize=12, fontweight="bold")

    cost_labels = ["Renewables", "Gas", "Storage", "T&D"]
    cost_vals = [costs["zvc"], costs["gas"],
                 costs["storage_power"] + costs["storage_energy"],
                 costs.get("td", 0)]
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
        dem_total += sb["supply"]["storage_charge"]
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
            if sb["supply"]["storage_charge"] > 1e-6:
                demand_labels.append("Sto. Charge")
                demand_vals.append(sb["supply"]["storage_charge"])

            supply_labels = ["Renewables", "Gas", "Storage"]
            supply_vals = [
                sb["supply"]["zvc"],
                sb["supply"]["gas"],
                sb["supply"]["storage_discharge"],
            ]

            d_colors = [DEMAND_COLOR_MAP.get(l, "#DDD6FE") for l in demand_labels]
            s_colors = [SUPPLY_COLOR_MAP.get(l, "#6366F1") for l in supply_labels]

            _stacked_bar(ax, 0, demand_vals, demand_labels, d_colors)
            _stacked_bar(ax, 1, supply_vals, supply_labels, s_colors)

            ax.set_ylim(0, y_max)
            ax.set_title(
                f"{b}\n{sb['label']} ({sb['hours']:.0f}h, {sb['zvc_availability']:.0f}% avail)"
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


DEMAND_COLOR_MAP = {
    "High": "#6366F1", "Mid": "#8B5CF6", "Low": "#A78BFA",
    "Shifted In": "#F59E0B", "Expandable": "#C4B5FD", "Sto. Charge": "#3B82F6",
}
SUPPLY_COLOR_MAP = {"Renewables": "#10B981", "Gas": "#EF4444", "Storage": "#3B82F6"}


def _stacked_bar(ax, x_pos, vals, labels, colors, fmt=".1f"):
    bottom = 0
    total = sum(v for v in vals if v > 0)
    for val, label, color in zip(vals, labels, colors):
        if val < 1e-6:
            continue
        ax.bar(x_pos, val, bottom=bottom, width=0.6,
               color=color, edgecolor="white", linewidth=0.5,
               label=label)
        threshold = max(0.5, total * 0.03)
        if val > threshold:
            ax.text(x_pos, bottom + val / 2, f"{val:{fmt}}",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
        bottom += val


def _plot_dispatch_gwh(inputs: dict, results: dict, cur: str = "$") -> str:
    """GWh version of dispatch bar charts — energy by sub-block."""
    blocks = inputs["blocks"]
    subs = results["sub_blocks"]

    sub_lookup = {}
    for sb in subs:
        sub_lookup[sb["block"], sb["sub_idx"]] = sb

    n_cols = len(blocks)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10), sharey=False)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, b in enumerate(blocks):
        for row in range(2):
            ax = axes[row, col]
            sb = sub_lookup[b, row]
            h = sb["hours"]

            demand_labels, demand_vals = [], []
            for t in sb["demand_tiers"]:
                demand_labels.append(t["name"])
                demand_vals.append(t["served"] * h)
            if sb["shifted_in_total"] > 1e-6:
                demand_labels.append("Shifted In")
                demand_vals.append(sb["shifted_in_total"] * h)
            if sb["expandable"]["activated"] > 1e-6:
                demand_labels.append("Expandable")
                demand_vals.append(sb["expandable"]["activated"] * h)
            if sb["supply"]["storage_charge"] > 1e-6:
                demand_labels.append("Sto. Charge")
                demand_vals.append(sb["supply"]["storage_charge"] * h)

            supply_labels = ["Renewables", "Gas", "Storage"]
            supply_vals = [
                sb["supply"]["zvc"] * h,
                sb["supply"]["gas"] * h,
                sb["supply"]["storage_discharge"] * h,
            ]

            d_colors = [DEMAND_COLOR_MAP.get(l, "#DDD6FE") for l in demand_labels]
            s_colors = [SUPPLY_COLOR_MAP.get(l, "#6366F1") for l in supply_labels]

            _stacked_bar(ax, 0, demand_vals, demand_labels, d_colors, fmt=",.0f")
            _stacked_bar(ax, 1, supply_vals, supply_labels, s_colors, fmt=",.0f")

            local_max = max(sum(demand_vals), sum(supply_vals)) * 1.1
            ax.set_ylim(0, max(local_max, 1))
            ax.set_title(
                f"{b}\n{sb['label']} ({sb['hours']:.0f}h, {sb['zvc_availability']:.0f}% avail)"
                f"\n{cur}{sb['price']:,.0f}/MWh",
                fontsize=10, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Demand", "Supply"])
            if col == 0:
                ax.set_ylabel("GWh")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    handles_all, labels_all = [], []
    for ax_flat in axes.flat:
        h_ax, l_ax = ax_flat.get_legend_handles_labels()
        for hi, li in zip(h_ax, l_ax):
            if li not in labels_all:
                handles_all.append(hi)
                labels_all.append(li)

    fig.legend(handles_all, labels_all, loc="lower center",
               ncol=min(len(labels_all), 6), fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Energy (GWh) by Sub-Block",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# GWh energy-flow Sankey — inter-period storage & shifting transfers
# ---------------------------------------------------------------------------

SB_COLORS = {
    ("Winter Peak", 0): "#DC2626", ("Winter Peak", 1): "#F87171",
    ("Shoulder", 0): "#D97706", ("Shoulder", 1): "#FBBF24",
    ("Low Demand", 0): "#059669", ("Low Demand", 1): "#34D399",
}


def _plot_energy_flows(inputs: dict, results: dict) -> str | None:
    """Sankey of GWh transfers between sub-blocks via storage and shifting."""
    subs = results["sub_blocks"]
    eta = inputs["supply"]["storage_efficiency"] / 100.0

    # Collect per-sub-block charge/discharge GWh
    chargers = []
    dischargers = []
    for sb in subs:
        h = sb["hours"]
        chg = sb["supply"]["storage_charge"] * h
        dis = sb["supply"]["storage_discharge"] * h
        lbl = f"{sb['block']}\n{sb['label']}"
        color = SB_COLORS.get((sb["block"], sb["sub_idx"]), "#6366F1")
        if chg > 0.1:
            chargers.append({"label": lbl, "gwh": chg, "color": color})
        if dis > 0.1:
            dischargers.append({"label": lbl, "gwh": dis, "color": color})

    # Collect shift flows — per destination sub-block
    hours = inputs["hours"]
    shift_flows = []
    seen_src = set()
    for sb in subs:
        if sb["sub_idx"] != 0:
            continue
        for t in sb["demand_tiers"]:
            for dest, qty in t["shifted_out"].items():
                key = (sb["block"], t["name"], dest)
                if key not in seen_src and qty > 0.01:
                    seen_src.add(key)
                    dest_sbs = [s for s in subs if s["block"] == dest]
                    for dsb in dest_sbs:
                        si_gwh = 0.0
                        for si in dsb.get("shifted_in", []):
                            if si["from_block"] == sb["block"] and si["tier"] == t["name"]:
                                si_gwh = si["quantity"] * dsb["hours"]
                        if si_gwh > 0.1:
                            shift_flows.append({
                                "src": f"Shift from\n{sb['block']}\n({t['name']}-value)",
                                "dest": f"Shift into\n{dest}\n{dsb['label']}",
                                "gwh": si_gwh,
                                "src_block": sb["block"],
                                "dest_block": dest,
                            })

    total_charge = sum(c["gwh"] for c in chargers)
    total_discharge = sum(d["gwh"] for d in dischargers)
    loss_gwh = max(0, total_charge - total_discharge)

    total_curtail_check = sum(
        t["lost"] * sb["hours"]
        for sb in subs for t in sb["demand_tiers"] if t["lost"] > 0.01
    )
    if total_charge < 0.1 and not shift_flows and total_curtail_check < 0.1:
        return None

    # --- Build Sankey: LEFT → RIGHT, with "battery" vs "behaviour" distinction ---
    left_nodes = []   # (label, gwh, color, kind)
    right_nodes = []  # (label, gwh, color, kind)
    flows = []        # (left_label, right_label, gwh, color, kind)

    # Storage chargers on left
    for c in chargers:
        left_nodes.append((c["label"], c["gwh"], c["color"], "battery"))

    # Storage dischargers on right
    for d in dischargers:
        right_nodes.append((d["label"], d["gwh"], d["color"], "battery"))

    # Efficiency loss on right
    if loss_gwh > 0.1:
        right_nodes.append(("Eff. Loss", loss_gwh, "#9CA3AF", "battery"))

    # Proportional allocation: each charger → each discharger
    if total_charge > 0.1:
        for c in chargers:
            share = c["gwh"] / total_charge
            for d in dischargers:
                flow_gwh = share * d["gwh"]
                if flow_gwh > 0.1:
                    flows.append((c["label"], d["label"], flow_gwh, d["color"], "battery"))
            if loss_gwh > 0.1:
                flows.append((c["label"], "Eff. Loss", share * loss_gwh, "#9CA3AF", "battery"))

    # Shift flows (behavioural change)
    for sf in shift_flows:
        src_c = BLOCK_COLORS.get(sf["src_block"], "#F59E0B")
        dst_c = BLOCK_COLORS.get(sf["dest_block"], "#10B981")
        left_nodes.append((sf["src"], sf["gwh"], src_c, "behaviour"))
        right_nodes.append((sf["dest"], sf["gwh"], dst_c, "behaviour"))
        flows.append((sf["src"], sf["dest"], sf["gwh"], dst_c, "behaviour"))

    # Curtailment flows (behavioural — demand withdrawn)
    curtail_by_block = {}
    for sb in subs:
        for t in sb["demand_tiers"]:
            if t["lost"] > 0.01:
                key = (sb["block"], t["name"])
                curtail_by_block[key] = curtail_by_block.get(key, 0) + t["lost"] * sb["hours"]

    total_curtail = sum(curtail_by_block.values())
    if total_curtail > 0.1:
        right_nodes.append(("Not Served", total_curtail, "#9CA3AF", "behaviour"))
        for (blk, tier), gwh in curtail_by_block.items():
            if gwh > 0.1:
                src_lbl = f"Curtailed\n{blk}\n({tier}-value)"
                src_c = BLOCK_COLORS.get(blk, "#6366F1")
                left_nodes.append((src_lbl, gwh, src_c, "behaviour"))
                flows.append((src_lbl, "Not Served", gwh, "#9CA3AF", "behaviour"))

    if not left_nodes:
        return None

    # --- Layout ---
    total_left = sum(g for _, g, _, _ in left_nodes)
    total_right = sum(g for _, g, _, _ in right_nodes)
    max_gwh = max(total_left, total_right)
    gap = max_gwh * 0.025
    section_gap = max_gwh * 0.06

    lx0, lx1 = 0.0, 0.06
    rx0, rx1 = 0.94, 1.0

    has_battery = any(k == "battery" for _, _, _, k in left_nodes)
    has_behaviour = any(k == "behaviour" for _, _, _, k in left_nodes)

    def _layout_nodes(nodes):
        pos = {}
        y = max_gwh + gap * (len(nodes) - 1)
        if has_battery and has_behaviour:
            y += section_gap
        prev_kind = None
        for label, gwh, _, kind in nodes:
            if prev_kind == "battery" and kind == "behaviour":
                y -= section_gap
            top = y
            bot = y - gwh
            pos[label] = (bot, top)
            y = bot - gap
            prev_kind = kind
        return pos

    left_pos = _layout_nodes(left_nodes)
    right_pos = _layout_nodes(right_nodes)

    # --- Draw ---
    fig_w = max(12, 6 + 0.8 * (len(left_nodes) + len(right_nodes)))
    fig, ax = plt.subplots(figsize=(fig_w, max(5, 0.6 * max(len(left_nodes), len(right_nodes)) + 2)))

    # Left rectangles
    for label, gwh, c, kind in left_nodes:
        bot, top = left_pos[label]
        if kind == "behaviour":
            ax.add_patch(plt.Rectangle(
                (lx0, bot), lx1 - lx0, top - bot,
                fc="none", ec=c, lw=2, linestyle="--", hatch="///", zorder=3))
        else:
            ax.add_patch(plt.Rectangle(
                (lx0, bot), lx1 - lx0, top - bot,
                fc=c, ec="white", lw=1.5, zorder=3))
        ax.text(lx0 - 0.01, (top + bot) / 2,
                f"{label}\n{gwh:,.0f} GWh",
                ha="right", va="center", fontsize=8, fontweight="bold", color=c)

    # Right rectangles
    for label, gwh, c, kind in right_nodes:
        bot, top = right_pos[label]
        if kind == "behaviour":
            ax.add_patch(plt.Rectangle(
                (rx0, bot), rx1 - rx0, top - bot,
                fc="none", ec=c, lw=2, linestyle="--", hatch="///", zorder=3))
        else:
            ax.add_patch(plt.Rectangle(
                (rx0, bot), rx1 - rx0, top - bot,
                fc=c, ec="white", lw=1.5, zorder=3))
        ax.text(rx1 + 0.01, (top + bot) / 2,
                f"{label}\n{gwh:,.0f} GWh",
                ha="left", va="center", fontsize=8, fontweight="bold", color=c)

    # Section labels
    if has_battery and has_behaviour:
        bat_tops = [left_pos[l][1] for l, _, _, k in left_nodes if k == "battery"]
        beh_tops = [left_pos[l][1] for l, _, _, k in left_nodes if k == "behaviour"]
        if bat_tops:
            ax.text(0.5, max(bat_tops) + gap * 1.5,
                    "BATTERY STORAGE", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#6366F1", alpha=0.7)
        if beh_tops:
            ax.text(0.5, max(beh_tops) + gap * 1.5,
                    "BEHAVIOURAL DEMAND SHIFT", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#9CA3AF", alpha=0.7)

    # Flow bands
    left_cur = {label: left_pos[label][1] for label, _, _, _ in left_nodes}
    right_cur = {label: right_pos[label][1] for label, _, _, _ in right_nodes}

    for src, dest, gwh, c, kind in flows:
        if gwh < 0.1:
            continue
        y0_top = left_cur[src]
        y0_bot = y0_top - gwh
        left_cur[src] = y0_bot

        y1_top = right_cur[dest]
        y1_bot = y1_top - gwh
        right_cur[dest] = y1_bot

        if kind == "behaviour":
            _bezier_band(ax, lx1, y0_bot, y0_top, rx0, y1_bot, y1_top, c, 0.12)
        else:
            _bezier_band(ax, lx1, y0_bot, y0_top, rx0, y1_bot, y1_top, c, 0.25)

        if gwh > max_gwh * 0.04:
            mx = 0.5
            my = (y0_top + y0_bot + y1_top + y1_bot) / 4
            ax.text(mx, my, f"{gwh:,.0f}",
                    ha="center", va="center", fontsize=8,
                    color=c, fontweight="bold", alpha=0.85)

    all_positions = list(left_pos.values()) + list(right_pos.values())
    max_h = max(t for _, t in all_positions) if all_positions else max_gwh
    min_h = min(b for b, _ in all_positions) if all_positions else 0
    pad = gap * 2
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(min_h - pad, max_h + pad)
    ax.axis("off")
    ax.set_title("Energy Transfers (GWh) — Storage & Shifting Between Sub-Blocks",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    return _fig_to_base64(fig)


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
# Flexibility Sankey diagrams
# ---------------------------------------------------------------------------

TIER_COLORS = {"High": "#4C1D95", "Mid": "#7C3AED", "Low": "#A78BFA"}
STO_COLOR = "#3B82F6"
DEST_COLORS = {
    "Served by Gen": "#10B981",
    "Served by Storage": "#0EA5E9",
    "Not Served": "#EF4444",
    "Into Storage": "#3B82F6",
}
SHIFT_COLOR = "#F59E0B"


def _compute_flexibility(inputs: dict, results: dict) -> list[dict]:
    """Aggregate per-block flexibility with storage and shifting (hours-weighted avg)."""
    blocks = inputs["blocks"]
    subs = results["sub_blocks"]
    flex = []

    for b in blocks:
        b_subs = [s for s in subs if s["block"] == b]
        lo, hi = b_subs[0], b_subs[1]
        h_lo, h_hi = lo["hours"], hi["hours"]
        h_total = h_lo + h_hi
        if h_total == 0:
            continue

        # Storage averages
        sto_chg = (h_lo * lo["supply"]["storage_charge"]
                   + h_hi * hi["supply"]["storage_charge"]) / h_total
        sto_dis = (h_lo * lo["supply"]["storage_discharge"]
                   + h_hi * hi["supply"]["storage_discharge"]) / h_total

        # Total supply and storage fraction for pro-rating "served by storage"
        sup_lo = (lo["supply"]["zvc"] + lo["supply"]["gas"]
                  + lo["supply"]["storage_discharge"])
        sup_hi = (hi["supply"]["zvc"] + hi["supply"]["gas"]
                  + hi["supply"]["storage_discharge"])
        sto_frac_lo = lo["supply"]["storage_discharge"] / sup_lo if sup_lo > 0 else 0
        sto_frac_hi = hi["supply"]["storage_discharge"] / sup_hi if sup_hi > 0 else 0

        # Shifted-in demand (dest-GW, already energy-scaled in model)
        shifted_in_total = lo["shifted_in_total"]

        tiers = []
        has_activity = False
        for j, t_lo in enumerate(lo["demand_tiers"]):
            t_hi = hi["demand_tiers"][j]
            avail = t_lo["quantity"]
            served_avg = (h_lo * t_lo["served"] + h_hi * t_hi["served"]) / h_total

            served_by_sto = (h_lo * t_lo["served"] * sto_frac_lo
                             + h_hi * t_hi["served"] * sto_frac_hi) / h_total
            served_by_gen = served_avg - served_by_sto

            shifted = {}
            shifted_total = 0.0
            for dest, qty in t_lo["shifted_out"].items():
                if qty > 0.01:
                    shifted[dest] = round(qty, 2)
                    shifted_total += qty
                    has_activity = True

            lost = max(0.0, avail - served_avg - shifted_total)
            if lost > 0.01:
                has_activity = True

            tiers.append({
                "name": t_lo["name"],
                "available": avail,
                "served_by_gen": round(max(0, served_by_gen), 2),
                "served_by_sto": round(max(0, served_by_sto), 2),
                "shifted": shifted,
                "lost": round(lost, 2),
            })

        if sto_chg > 0.01 or sto_dis > 0.01:
            has_activity = True
        if shifted_in_total > 0.01:
            has_activity = True

        if has_activity:
            flex.append({
                "block": b,
                "tiers": tiers,
                "storage_charge": round(sto_chg, 2),
                "storage_discharge": round(sto_dis, 2),
                "shifted_in": round(shifted_in_total, 2),
                "shifted_in_detail": lo["shifted_in"],
            })

    return flex


def _bezier_band(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.35):
    """Draw a smooth curved band (Sankey flow) between two vertical spans."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    xm = (x0 + x1) / 2
    verts = [
        (x0, y0_bot),
        (xm, y0_bot), (xm, y1_bot), (x1, y1_bot),
        (x1, y1_top),
        (xm, y1_top), (xm, y0_top), (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(Path(verts, codes), fc=color, ec="none", alpha=alpha)
    ax.add_patch(patch)


SHIFTIN_COLOR = "#F97316"


def _draw_sankey(ax, block_data):
    """Draw one Sankey diagram with storage and shifting activity."""
    tiers = block_data["tiers"]
    block_name = block_data["block"]
    sto_chg = block_data["storage_charge"]
    shifted_in = block_data.get("shifted_in", 0)
    shifted_in_detail = block_data.get("shifted_in_detail", [])

    # --- Build left-side nodes: demand tiers + shifted in + storage charge ---
    left_nodes = []  # (label, gw, color)
    for t in tiers:
        left_nodes.append((t["name"], t["available"],
                           TIER_COLORS.get(t["name"], "#6366F1")))
    if shifted_in > 0.01:
        srcs = ", ".join(si["from_block"] for si in shifted_in_detail)
        lbl = f"Shifted In\n({srcs})" if srcs else "Shifted In"
        left_nodes.append((lbl, shifted_in, SHIFTIN_COLOR))
    if sto_chg > 0.01:
        left_nodes.append(("Sto. Charge", sto_chg, STO_COLOR))

    # --- Build flows (left_label → right_label, qty) ---
    flows = []
    for t in tiers:
        if t["served_by_gen"] > 0.01:
            flows.append((t["name"], "Served by Gen", t["served_by_gen"]))
        if t["served_by_sto"] > 0.01:
            flows.append((t["name"], "Served by Storage", t["served_by_sto"]))
        for dest in sorted(t["shifted"]):
            if t["shifted"][dest] > 0.01:
                flows.append((t["name"], f"→ {dest}", t["shifted"][dest]))
        if t["lost"] > 0.01:
            flows.append((t["name"], "Not Served", t["lost"]))
    if shifted_in > 0.01:
        lbl = left_nodes[-2 if sto_chg > 0.01 else -1][0]
        flows.append((lbl, "Served by Gen", shifted_in))
    if sto_chg > 0.01:
        flows.append(("Sto. Charge", "Into Storage", sto_chg))

    # --- Compute right-side node totals ---
    dest_totals: dict[str, float] = {}
    dest_seen: list[str] = []
    for _, dest, qty in flows:
        if dest not in dest_totals:
            dest_totals[dest] = 0.0
            dest_seen.append(dest)
        dest_totals[dest] += qty

    # Order: Served by Gen, Served by Storage, shifted…, Not Served, Into Storage
    ordered_dests: list[str] = []
    for key in ("Served by Gen", "Served by Storage"):
        if key in dest_totals:
            ordered_dests.append(key)
    for d in dest_seen:
        if d.startswith("→"):
            ordered_dests.append(d)
    for key in ("Not Served", "Into Storage"):
        if key in dest_totals:
            ordered_dests.append(key)

    # --- Layout ---
    total_left_gw = sum(gw for _, gw, _ in left_nodes)
    total_right_gw = sum(dest_totals[d] for d in ordered_dests)
    max_gw = max(total_left_gw, total_right_gw)
    gap = max_gw * 0.03

    lx0, lx1 = 0.0, 0.08
    rx0, rx1 = 0.92, 1.0

    # Left positions (top-down)
    left_pos = {}
    y = max_gw + gap * (len(left_nodes) - 1)
    for label, gw, _ in left_nodes:
        top = y
        bot = y - gw
        left_pos[label] = (bot, top)
        y = bot - gap

    # Right positions (top-down)
    right_pos = {}
    y = max_gw + gap * (len(ordered_dests) - 1)
    for d in ordered_dests:
        top = y
        bot = y - dest_totals[d]
        right_pos[d] = (bot, top)
        y = bot - gap

    # --- Draw left rectangles ---
    for label, gw, c in left_nodes:
        bot, top = left_pos[label]
        ax.add_patch(plt.Rectangle(
            (lx0, bot), lx1 - lx0, top - bot,
            fc=c, ec="white", lw=1.5, zorder=3))
        ax.text(lx0 - 0.02, (top + bot) / 2,
                f"{label}\n{gw:.1f} GW",
                ha="right", va="center", fontsize=9, fontweight="bold", color=c)

    # --- Draw right rectangles ---
    for d in ordered_dests:
        bot, top = right_pos[d]
        c = DEST_COLORS.get(d, SHIFT_COLOR)
        ax.add_patch(plt.Rectangle(
            (rx0, bot), rx1 - rx0, top - bot,
            fc=c, ec="white", lw=1.5, zorder=3))
        ax.text(rx1 + 0.02, (top + bot) / 2,
                f"{d}\n{dest_totals[d]:.1f} GW",
                ha="left", va="center", fontsize=9, fontweight="bold", color=c)

    # --- Draw flow bands ---
    left_cur = {label: left_pos[label][1] for label, _, _ in left_nodes}
    right_cur = {d: right_pos[d][1] for d in ordered_dests}

    for src, dest, qty in flows:
        if qty < 0.01:
            continue
        y0_top = left_cur[src]
        y0_bot = y0_top - qty
        left_cur[src] = y0_bot

        y1_top = right_cur[dest]
        y1_bot = y1_top - qty
        right_cur[dest] = y1_bot

        c = DEST_COLORS.get(dest, SHIFT_COLOR)
        _bezier_band(ax, lx1, y0_bot, y0_top, rx0, y1_bot, y1_top, c, 0.3)

        if qty > max_gw * 0.04:
            mx = 0.5
            my = (y0_top + y0_bot + y1_top + y1_bot) / 4
            ax.text(mx, my, f"{qty:.1f}",
                    ha="center", va="center", fontsize=8,
                    color=c, fontweight="bold", alpha=0.9)

    # --- Formatting ---
    n_left = len(left_nodes)
    n_right = len(ordered_dests)
    max_h = max_gw + gap * max(n_left, n_right)
    pad = gap * 2
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-pad, max_h + pad)
    ax.axis("off")
    ax.set_title(block_name, fontsize=12, fontweight="bold")


def _plot_flexibility(inputs: dict, results: dict) -> str | None:
    """Generate Sankey diagrams for blocks with flexibility or storage activity."""
    flex = _compute_flexibility(inputs, results)
    if not flex:
        return None

    n = len(flex)
    fig, axes = plt.subplots(1, n, figsize=(min(8 * n, 24), 6))
    if n == 1:
        axes = [axes]

    for ax, fb in zip(axes, flex):
        _draw_sankey(ax, fb)

    fig.suptitle("Demand Flexibility — Shifting, Curtailment & Storage",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
