"""
Electricity Market LP Model — with ZVC Intermittency
=====================================================
Each LDC block is split into two sub-blocks (Low Renewables and High Renewables)
to capture intermittency of zero-variable-cost renewables.  This enables
within-block storage arbitrage: charge when renewables are plentiful, discharge
when scarce.

The model simultaneously determines:
  • optimal capacity investment (ZVC, gas peaker, storage)
  • dispatch in every sub-block

Shadow prices on sub-block energy-balance constraints give clearing prices.
Storage round-trip efficiency is physical; the implied cycling cost is output.

Two-pass shifting logic:
  Pass 1 — sub-block-level shift_in (can target cheaper sub-blocks).
  If ZVC capacity = 0, pass 2 — block-level shift_in (uniform GW across
  sub-blocks) to eliminate spurious asymmetry from LP degeneracy.

Units:  Power GW · Energy GWh · Price $/MWh · Capital $/kW/yr or $/kWh/yr
"""

from __future__ import annotations

import pulp
from typing import Any


# ---------------------------------------------------------------------------
# Default inputs
# ---------------------------------------------------------------------------

def default_inputs() -> dict[str, Any]:
    blocks = ["Winter Peak", "Shoulder", "Low Demand"]
    hours = {"Winter Peak": 300, "Shoulder": 3000, "Low Demand": 5460}
    base_load = {"Winter Peak": 60, "Shoulder": 40, "Low Demand": 20}

    demand_tiers = {
        "Winter Peak": [
            {"name": "High", "quantity": 10, "voll": 15000, "shift_cost": 1000},
            {"name": "Mid",  "quantity": 20, "voll": 800,   "shift_cost": 500},
            {"name": "Low",  "quantity": 15, "voll": 100,   "shift_cost": 50},
        ],
        "Shoulder": [
            {"name": "High", "quantity": 8,  "voll": 12000, "shift_cost": 800},
            {"name": "Mid",  "quantity": 18, "voll": 600,   "shift_cost": 40},
            {"name": "Low",  "quantity": 10, "voll": 120,   "shift_cost": 8},
        ],
        "Low Demand": [
            {"name": "High", "quantity": 5,  "voll": 10000, "shift_cost": 0},
            {"name": "Mid",  "quantity": 10, "voll": 400,   "shift_cost": 0},
            {"name": "Low",  "quantity": 5,  "voll": 100,   "shift_cost": 0},
        ],
    }

    expandable = {
        "Winter Peak": {"quantity": 100,  "value": 40},
        "Shoulder":    {"quantity": 500,  "value": 30},
        "Low Demand":  {"quantity": 1000, "value": 20},
    }

    zvc_profile = {
        "Winter Peak": [
            {"label": "Low Renewables",  "pct_hours": 10, "availability": 15},
            {"label": "High Renewables", "pct_hours": 90, "availability": 80},
        ],
        "Shoulder": [
            {"label": "Low Renewables",  "pct_hours": 20, "availability": 20},
            {"label": "High Renewables", "pct_hours": 80, "availability": 70},
        ],
        "Low Demand": [
            {"label": "Low Renewables",  "pct_hours": 15, "availability": 10},
            {"label": "High Renewables", "pct_hours": 85, "availability": 90},
        ],
    }

    supply = {
        "zvc_capital_cost": 450,
        "gas_capital_cost": 40,
        "gas_variable_cost": 80,
        "carbon_price": 0,
        "gas_emission_factor": 0.5,
        "storage_power_cost": 10,
        "storage_energy_cost": 10,
        "storage_cycles": 365,
        "storage_efficiency": 95,
    }

    return {
        "blocks": blocks,
        "hours": hours,
        "base_load": base_load,
        "demand_tiers": demand_tiers,
        "expandable": expandable,
        "zvc_profile": zvc_profile,
        "supply": supply,
    }


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

CAPITAL_CONV = 1000  # $/kW/yr × GW → model-units/yr


def solve(inputs: dict[str, Any]) -> dict[str, Any]:
    """Two-pass entry point.

    Pass 1: sub-block-level shift_in (full flexibility).
    If the result has ZVC capacity ≈ 0, re-solve with block-level shift_in
    to avoid LP-degeneracy artefacts when sub-blocks are identical.
    """
    result = _solve_core(inputs, block_level_shift=False)
    if result.get("error"):
        return result

    if result["capacities"]["zvc"] < 0.01:
        result = _solve_core(inputs, block_level_shift=True)

    return result


def _solve_core(inputs: dict[str, Any], *,
                block_level_shift: bool = False) -> dict[str, Any]:
    blocks = inputs["blocks"]
    hours = inputs["hours"]
    demand_tiers = inputs["demand_tiers"]
    expandable = inputs["expandable"]
    zvc_profile = inputs["zvc_profile"]
    supply = inputs["supply"]

    carbon_adder = supply.get("carbon_price", 0) * supply.get("gas_emission_factor", 0.5)
    gas_vc = supply["gas_variable_cost"] + carbon_adder
    eta = supply["storage_efficiency"] / 100.0
    annual_cycles = max(supply.get("storage_cycles", 365), 1)
    eff_energy_cost = supply["storage_energy_cost"] / annual_cycles

    nb = len(blocks)
    prob = pulp.LpProblem("ElectricityMarket", pulp.LpMaximize)

    def _s(n: str) -> str:
        return n.replace(" ", "_").replace("-", "_")

    # ── Build sub-block list ─────────────────────────────────────────────
    sub_blocks: list[tuple[str, int, str, float, float]] = []
    for b in blocks:
        for i, sp in enumerate(zvc_profile[b]):
            sh = hours[b] * sp["pct_hours"] / 100.0
            av = sp["availability"] / 100.0
            sub_blocks.append((b, i, sp["label"], sh, av))

    def sb_key(b, i):
        return (b, i)

    # ── Capacity investment variables ────────────────────────────────────
    cap_zvc = pulp.LpVariable("cap_zvc", 0)
    cap_gas = pulp.LpVariable("cap_gas", 0)
    cap_sto_pw = pulp.LpVariable("cap_sto_pw", 0)
    cap_sto_en = pulp.LpVariable("cap_sto_en", 0)

    # ── Dispatch variables (per sub-block) ───────────────────────────────
    served = {}
    for b, i, lbl, sh, av in sub_blocks:
        for t in demand_tiers[b]:
            served[b, i, t["name"]] = pulp.LpVariable(
                f"srv_{_s(b)}_{i}_{t['name']}", 0, t["quantity"]
            )

    expand = {}
    for b, i, lbl, sh, av in sub_blocks:
        expand[b, i] = pulp.LpVariable(
            f"exp_{_s(b)}_{i}", 0, expandable[b]["quantity"]
        )

    zvc = {}
    gas = {}
    sto_dis = {}
    sto_chg = {}
    for b, i, lbl, sh, av in sub_blocks:
        k = sb_key(b, i)
        zvc[k] = pulp.LpVariable(f"zvc_{_s(b)}_{i}", 0)
        gas[k] = pulp.LpVariable(f"gas_{_s(b)}_{i}", 0)
        sto_dis[k] = pulp.LpVariable(f"sdis_{_s(b)}_{i}", 0)
        sto_chg[k] = pulp.LpVariable(f"schg_{_s(b)}_{i}", 0)

    # Shift-out (block-level, only downward, not from last block)
    shift_out = {}
    for idx_b, b in enumerate(blocks):
        if idx_b >= nb - 1:
            continue
        for t in demand_tiers[b]:
            for j in range(idx_b + 1, nb):
                b2 = blocks[j]
                shift_out[b, t["name"], b2] = pulp.LpVariable(
                    f"sho_{_s(b)}_{t['name']}_to_{_s(b2)}", 0
                )

    # Shift-in variables — mode depends on block_level_shift flag
    sub_hours = {(b, i): sh for b, i, _lbl, sh, _av in sub_blocks}
    shift_in = {}  # keyed (src_block, tier, dest_block, dest_sub_idx)

    if not block_level_shift:
        # Sub-block-level: separate variable per destination sub-block
        for idx_b, b in enumerate(blocks):
            if idx_b >= nb - 1:
                continue
            for t in demand_tiers[b]:
                for j in range(idx_b + 1, nb):
                    b2 = blocks[j]
                    for _, i2, *_ in [s for s in sub_blocks if s[0] == b2]:
                        shift_in[b, t["name"], b2, i2] = pulp.LpVariable(
                            f"shi_{_s(b)}_{t['name']}_to_{_s(b2)}_{i2}", 0
                        )
    else:
        # Block-level: one variable per destination block, shared across
        # sub-blocks (same GW in every sub-block of the destination).
        _shift_in_blk = {}
        for idx_b, b in enumerate(blocks):
            if idx_b >= nb - 1:
                continue
            for t in demand_tiers[b]:
                for j in range(idx_b + 1, nb):
                    b2 = blocks[j]
                    v = pulp.LpVariable(
                        f"shi_{_s(b)}_{t['name']}_to_{_s(b2)}", 0
                    )
                    _shift_in_blk[b, t["name"], b2] = v
                    for _, i2, *_ in [s for s in sub_blocks if s[0] == b2]:
                        shift_in[b, t["name"], b2, i2] = v

    # Storage levels
    sto_level = {}
    for b, i, *_ in sub_blocks:
        sto_level[b, i] = pulp.LpVariable(f"slev_{_s(b)}_{i}", 0)
    sto_level_start = pulp.LpVariable("slev_start", 0)

    # ── Objective ────────────────────────────────────────────────────────
    obj = []

    for b, i, lbl, sh, av in sub_blocks:
        k = sb_key(b, i)
        for t in demand_tiers[b]:
            obj.append(sh * t["voll"] * served[b, i, t["name"]])
        obj.append(sh * expandable[b]["value"] * expand[b, i])
        obj.append(-sh * gas_vc * gas[k])

    for idx_b, b in enumerate(blocks):
        for idx_src in range(idx_b):
            b_src = blocks[idx_src]
            for t in demand_tiers[b_src]:
                if (b_src, t["name"], b) in shift_out:
                    obj.append(
                        hours[b_src] * (t["voll"] - t["shift_cost"])
                        * shift_out[b_src, t["name"], b]
                    )

    obj.append(-CAPITAL_CONV * supply["zvc_capital_cost"] * cap_zvc)
    obj.append(-CAPITAL_CONV * supply["gas_capital_cost"] * cap_gas)
    obj.append(-CAPITAL_CONV * supply["storage_power_cost"] * cap_sto_pw)
    obj.append(-CAPITAL_CONV * eff_energy_cost * cap_sto_en)

    prob += pulp.lpSum(obj), "Welfare"

    # ── Constraints ──────────────────────────────────────────────────────

    # 1. Energy balance per sub-block
    eb_names = {}
    for b, i, lbl, sh, av in sub_blocks:
        k = sb_key(b, i)
        dem = pulp.lpSum(served[b, i, t["name"]] for t in demand_tiers[b])

        shifted_in_vars = []
        for idx_src in range(blocks.index(b)):
            b_src = blocks[idx_src]
            for t in demand_tiers[b_src]:
                if (b_src, t["name"], b, i) in shift_in:
                    shifted_in_vars.append(shift_in[b_src, t["name"], b, i])

        total_dem = dem + pulp.lpSum(shifted_in_vars) + expand[b, i]
        total_sup = zvc[k] + gas[k] + sto_dis[k] - sto_chg[k]

        cname = f"eb_{_s(b)}_{i}"
        prob.addConstraint(total_sup >= total_dem, name=cname)
        eb_names[b, i] = cname

    # 1b. Energy conservation for shifts: total GWh in = total GWh out
    _seen_shift_cons = set()
    for idx_b, b in enumerate(blocks):
        if idx_b >= nb - 1:
            continue
        for t in demand_tiers[b]:
            for j in range(idx_b + 1, nb):
                b2 = blocks[j]
                if (b, t["name"], b2) not in shift_out:
                    continue
                cons_key = (b, t["name"], b2)
                if cons_key in _seen_shift_cons:
                    continue
                _seen_shift_cons.add(cons_key)

                if not block_level_shift:
                    gwh_in = pulp.lpSum(
                        shift_in[b, t["name"], b2, i2] * sub_hours[b2, i2]
                        for _, i2, *_ in [s for s in sub_blocks if s[0] == b2]
                    )
                else:
                    gwh_in = _shift_in_blk[b, t["name"], b2] * hours[b2]

                gwh_out = shift_out[b, t["name"], b2] * hours[b]
                prob += gwh_in == gwh_out, \
                    f"shcons_{_s(b)}_{t['name']}_to_{_s(b2)}"

    # 2. Tier balance: served + shifted_out ≤ quantity
    for b, i, lbl, sh, av in sub_blocks:
        for t in demand_tiers[b]:
            shifts = []
            idx_b = blocks.index(b)
            if idx_b < nb - 1:
                for j in range(idx_b + 1, nb):
                    b2 = blocks[j]
                    if (b, t["name"], b2) in shift_out:
                        shifts.append(shift_out[b, t["name"], b2])
            prob += (
                served[b, i, t["name"]] + pulp.lpSum(shifts)
                <= t["quantity"]
            ), f"tb_{_s(b)}_{i}_{t['name']}"

    # 3. Capacity constraints
    for b, i, lbl, sh, av in sub_blocks:
        k = sb_key(b, i)
        prob += zvc[k] <= av * cap_zvc, f"czvc_{_s(b)}_{i}"
        prob += gas[k] <= cap_gas, f"cgas_{_s(b)}_{i}"
        prob += sto_dis[k] <= cap_sto_pw, f"cdis_{_s(b)}_{i}"
        prob += sto_chg[k] <= cap_sto_pw, f"cchg_{_s(b)}_{i}"

    # 4. Storage energy level limits
    prob += sto_level_start <= cap_sto_en, "cslev_start"
    for b, i, *_ in sub_blocks:
        prob += sto_level[b, i] <= cap_sto_en, f"cslev_{_s(b)}_{i}"

    # 5. Storage dynamics (sequential: lo then hi within each block)
    prev = sto_level_start
    for b, i, lbl, sh, av in sub_blocks:
        k = sb_key(b, i)
        prob += (
            sto_level[k] == prev + sh * (eta * sto_chg[k] - sto_dis[k])
        ), f"sdyn_{_s(b)}_{i}"
        prev = sto_level[k]

    # 6. Cyclical
    prob += sto_level[sub_blocks[-1][0], sub_blocks[-1][1]] == sto_level_start, "scyc"

    # ── Solve ────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        return {"status": pulp.LpStatus[status], "error": True}

    # ── Extract results ──────────────────────────────────────────────────
    def _v(var):
        return round(pulp.value(var), 4)

    sb_prices = {}
    for b, i, lbl, sh, av in sub_blocks:
        con = prob.constraints[eb_names[b, i]]
        shadow = -con.pi if con.pi is not None else 0.0
        sb_prices[b, i] = round(shadow / sh, 2) if sh > 0 else 0.0

    block_prices = {}
    for b in blocks:
        total_h = sum(sh for bb, ii, ll, sh, aa in sub_blocks if bb == b)
        if total_h > 0:
            block_prices[b] = round(
                sum(sb_prices[b, i] * sh
                    for bb, i, ll, sh, aa in sub_blocks if bb == b)
                / total_h, 2
            )
        else:
            block_prices[b] = 0.0

    capacities = {
        "zvc": _v(cap_zvc),
        "gas": _v(cap_gas),
        "storage_power": _v(cap_sto_pw),
        "storage_energy": _v(cap_sto_en),
    }

    annual_capital = {
        "zvc": round(capacities["zvc"] * supply["zvc_capital_cost"] * CAPITAL_CONV, 2),
        "gas": round(capacities["gas"] * supply["gas_capital_cost"] * CAPITAL_CONV, 2),
        "storage_power": round(capacities["storage_power"] * supply["storage_power_cost"] * CAPITAL_CONV, 2),
        "storage_energy": round(capacities["storage_energy"] * eff_energy_cost * CAPITAL_CONV, 2),
    }
    annual_capital["total"] = round(sum(annual_capital.values()), 2)

    # Sub-block details
    sb_details = {}
    for b, i, lbl, sh, av in sub_blocks:
        k = sb_key(b, i)
        d: dict[str, Any] = {}
        d["block"] = b
        d["sub_idx"] = i
        d["label"] = lbl
        d["hours"] = round(sh, 1)
        d["zvc_availability"] = round(av * 100, 1)
        d["price"] = sb_prices[b, i]

        d["demand_tiers"] = []
        for t in demand_tiers[b]:
            ti = {
                "name": t["name"],
                "quantity": t["quantity"],
                "voll": t["voll"],
                "served": _v(served[b, i, t["name"]]),
                "lost": 0.0,
                "shifted_out": {},
            }
            total_shift = 0.0
            idx_b = blocks.index(b)
            if idx_b < nb - 1:
                for j in range(idx_b + 1, nb):
                    b2 = blocks[j]
                    if (b, t["name"], b2) in shift_out:
                        val = _v(shift_out[b, t["name"], b2])
                        if val > 1e-6:
                            ti["shifted_out"][b2] = val
                            total_shift += val
            ti["lost"] = round(t["quantity"] - ti["served"] - total_shift, 4)
            d["demand_tiers"].append(ti)

        shifted_in_total = 0.0
        shifted_in_list = []
        for idx_src in range(blocks.index(b)):
            b_src = blocks[idx_src]
            for t in demand_tiers[b_src]:
                if (b_src, t["name"], b, i) in shift_in:
                    val_dest = _v(shift_in[b_src, t["name"], b, i])
                    val_src = _v(shift_out[b_src, t["name"], b])
                    if val_dest > 1e-6:
                        shifted_in_list.append({
                            "from_block": b_src, "tier": t["name"],
                            "quantity": round(val_dest, 4),
                            "quantity_src_gw": round(val_src, 4),
                        })
                        shifted_in_total += val_dest
        d["shifted_in"] = shifted_in_list
        d["shifted_in_total"] = round(shifted_in_total, 4)

        d["expandable"] = {
            "available": expandable[b]["quantity"],
            "activated": _v(expand[b, i]),
            "value": expandable[b]["value"],
        }

        d["supply"] = {
            "zvc": _v(zvc[k]),
            "gas": _v(gas[k]),
            "storage_discharge": _v(sto_dis[k]),
            "storage_charge": _v(sto_chg[k]),
        }
        d["storage"] = {"level": _v(sto_level[k])}

        d["total_served_gw"] = round(
            sum(ti["served"] for ti in d["demand_tiers"])
            + d["shifted_in_total"]
            + d["expandable"]["activated"], 4
        )

        sb_details[b, i] = d

    # Storage economics
    total_chg_gwh = sum(
        sb_details[b, i]["hours"] * sb_details[b, i]["supply"]["storage_charge"]
        for b, i, *_ in sub_blocks
    )
    total_dis_gwh = sum(
        sb_details[b, i]["hours"] * sb_details[b, i]["supply"]["storage_discharge"]
        for b, i, *_ in sub_blocks
    )
    eff_loss = round(eta * total_chg_gwh - total_dis_gwh, 2)

    avg_chg_p = 0.0
    avg_dis_p = 0.0
    if total_chg_gwh > 1e-6:
        avg_chg_p = sum(
            sb_prices[b, i] * sb_details[b, i]["hours"] * sb_details[b, i]["supply"]["storage_charge"]
            for b, i, *_ in sub_blocks
        ) / total_chg_gwh
    if total_dis_gwh > 1e-6:
        avg_dis_p = sum(
            sb_prices[b, i] * sb_details[b, i]["hours"] * sb_details[b, i]["supply"]["storage_discharge"]
            for b, i, *_ in sub_blocks
        ) / total_dis_gwh

    implied_cc = 0.0
    if total_dis_gwh > 1e-6 and eta > 0:
        implied_cc = round(avg_chg_p / eta - avg_dis_p, 2)

    storage_econ = {
        "total_charged_gwh": round(total_chg_gwh, 2),
        "total_discharged_gwh": round(total_dis_gwh, 2),
        "efficiency_loss_gwh": eff_loss,
        "avg_charge_price": round(avg_chg_p, 2),
        "avg_discharge_price": round(avg_dis_p, 2),
        "implied_cycling_cost": implied_cc,
        "round_trip_efficiency": supply["storage_efficiency"],
    }

    # Welfare decomposition
    total_cv = 0.0
    total_vc = 0.0
    for b, i, lbl, sh, av in sub_blocks:
        d = sb_details[b, i]
        for t in d["demand_tiers"]:
            total_cv += sh * t["voll"] * t["served"]
        total_cv += sh * d["expandable"]["value"] * d["expandable"]["activated"]
        total_vc += sh * gas_vc * d["supply"]["gas"]

    seen_shifts = set()
    for b, i, lbl, sh, av in sub_blocks:
        for si in sb_details[b, i]["shifted_in"]:
            key = (si["from_block"], si["tier"], b)
            if key not in seen_shifts:
                seen_shifts.add(key)
                src_tier = next(
                    x for x in demand_tiers[si["from_block"]]
                    if x["name"] == si["tier"]
                )
                qty_src = si.get("quantity_src_gw", si["quantity"])
                total_cv += hours[si["from_block"]] * (
                    src_tier["voll"] - src_tier["shift_cost"]
                ) * qty_src

    # ── Assemble results ─────────────────────────────────────────────────
    sb_list = []
    for b, i, *_ in sub_blocks:
        sb_list.append(sb_details[b, i])

    results: dict[str, Any] = {
        "status": "Optimal",
        "error": False,
        "objective": round(pulp.value(prob.objective), 2),
        "blocks": blocks,
        "hours": hours,
        "block_prices": block_prices,
        "sub_blocks": sb_list,
        "capacities": capacities,
        "annual_capital": annual_capital,
        "storage_economics": storage_econ,
        "total_consumer_value": round(total_cv, 2),
        "total_variable_cost": round(total_vc, 2),
        "total_capital_cost": annual_capital["total"],
        "net_welfare": round(total_cv - total_vc - annual_capital["total"], 2),
    }
    return results
