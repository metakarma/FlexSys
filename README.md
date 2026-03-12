# Peaky Flexers — Electricity Market LP Optimiser

A web application that solves a welfare-maximising linear program for electricity market price formation. The model simultaneously determines **optimal capacity investment** and **dispatch** for four supply-side elements — renewables (zero variable cost), gas peakers, battery storage, and transmission & distribution — across a three-block Load Duration Curve with **renewables intermittency sub-blocks**, flexible (shiftable) and expandable demand.

A currency selector (£/$/€) at the top of the UI updates all labels and plot annotations. Each input section has a hover-over **info button** (ⓘ) that shows the relevant portion of the user guide inline.

## Quick Start

### Optional Beta Access Gate

For a simple beta password layer, set the environment variable `PEAKY_BETA_PASSWORD`. When set, the model page is greyed out until the correct password is entered, and optimise/chat endpoints remain blocked until access is granted. The provided `docker-compose.yml` passes this environment variable through into the container.

### With Docker

```bash
docker compose up --build
```

Open [http://localhost:8180](http://localhost:8180).

### Without Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:8080](http://localhost:8080).

## Model

### Renewables Intermittency (Sub-Blocks)

Each LDC block is split into two sub-blocks — **Low Renewables** and **High Renewables** — to capture the intermittent nature of renewable generation. Parameters per block:

| Parameter | Description |
|-----------|-------------|
| **% Hours** | Fraction of block hours in each sub-block (must sum to 100) |
| **% Renewables Availability** | Fraction of installed renewables capacity available in each sub-block |

This creates intra-block price differentials: prices are higher in Low Renewables sub-blocks (scarcity) and lower in High Renewables sub-blocks (abundance). Storage can arbitrage these differences by charging when renewables are plentiful and discharging when they are scarce.

### Supply Technologies (endogenous capacity)

The model determines the **welfare-maximising capacity mix**. No capacities are pre-set.

| Technology | Cost Parameters |
|-----------|----------------|
| **Renewables** (wind, solar, nuclear) | Annualised capital cost: £450/kW/year. Zero variable cost. |
| **Gas peaker** | Capital: £40/kW/year. Variable: £80/MWh + carbon price (default £50/tCO₂). |
| **Battery storage** | Duration: 4h. Connection: £150,000/MW. Cells: £150,000/MWh. Life: 15 years. Discount rate: 10%. 365 cycles/year. 95% round-trip efficiency. |
| **Transmission & Distribution** | Capacity cost: £10/kW/year. Sized to peak system generation. |

### Storage

Storage inputs are specified as physical CapEx parameters: **duration** (hours), **connection cost** (£/MW for inverter/grid connection), **cell cost** (£/MWh for battery cells), **asset life** (years), and **discount rate** (%). The model annualises using the <strong>capital recovery factor</strong> (CRF):

- CRF = r(1+r)^n / ((1+r)^n − 1), where r = discount rate (decimal), n = asset life
- Power cost (£/kW/yr) = (connection cost / 1000) × CRF
- Energy cost (£/kWh/yr) = (cell cost / 1000) × CRF

When discount rate is zero, CRF = 1/n (straight-line). The **cycles per year** parameter scales the storage energy level constraints in the LP, giving it a realistic annual throughput budget (physical capacity × cycles) while keeping full annual capital costs in the objective. A duration constraint links energy and power capacity: energy (GWh) = power (GW) × duration (h).

Round-trip efficiency (default 95%) creates an implicit cycling cost that depends on charge/discharge prices — this is an **output**, not an input.

### Transmission & Distribution

T&D capacity is determined endogenously. The constraint binds at peak total generation dispatched (renewables + gas + storage discharge) across all sub-blocks. The annualised T&D capacity cost (default £10/kW/year) enters the welfare objective alongside generation investment costs.

### Demand

Default LDC blocks: Winter Peak (300h, 45 GW), Shoulder (3,000h, 35 GW), Low Demand (5,460h, 20 GW). Demand is split into tiers at 10% High, 60% Mid, 30% Low of total load.

Demand can shift **downward** (Winter Peak → Shoulder → Low Demand) at a specified shift cost per tier. Low Demand periods have zero shift cost (demand cannot shift further down).

Expandable demand activates when the price drops below its value:

| Block | Quantity | Activation Value |
|-------|----------|-----------------|
| Winter Peak | 100 GW | £40/MWh |
| Shoulder | 500 GW | £30/MWh |
| Low Demand | 1,000 GW | £20/MWh |

### Prices

Shadow prices on the energy-balance constraints give **market-clearing prices** for each sub-block (6 prices total). When renewables capacity is zero and sub-blocks are identical, the model uses the minimum shadow price to reflect competitive price formation.

### Monetary units

All welfare and cost outputs are reported in full annual currency units. Internally, the model converts:

- **GW × hours → MWh** using `× 1000` for all demand value, gas variable cost, and shift-value terms
- **GW/GWh → kW/kWh** using `× 1,000,000` for all annual capital-cost terms

This keeps the optimisation and reported totals consistent with the input units shown in the UI.

## AI Chat Assistant

An expandable chat panel (toggle via the **AI Chat** button, bottom-right) lets users interrogate model results conversationally using Anthropic's Claude. Features:

- **Model selector** — defaults to Claude Opus 4.6.
- **API key input** — your Anthropic key is stored locally in the browser and never sent to the server beyond proxying requests.
- **Full model context** — every message is sent alongside the complete model source code (`model.py`), the current input parameters, and the latest optimisation results, so the AI can reason about your specific scenario.
- **Markdown & LaTeX rendering** — responses are formatted with Markdown (via `marked.js`) and mathematical equations are rendered with KaTeX.
- **Clear Chat** button to reset the conversation history.
- **Copy Chat** button to copy the full conversation to clipboard.
- **Download buttons** on all output charts (PNG) and tables (CSV).
- **Hover pop-outs** on the generation duration curve and price duration curve to inspect each sub-block without cluttering the chart.
- **Summary money metrics in millions** (for example `£m`) for cleaner headline reporting.
- **Public comments section** at the bottom of the page with links for GitHub feature requests and direct email contact.

## User Guide

A comprehensive pedagogic guide is built into the application at [`/guide`](http://localhost:8180/guide). It explains:

- The logic behind each input section (LDC blocks, sub-blocks, demand tiers, supply costs)
- How the welfare-maximising LP works, in plain language and formal notation
- How to interpret each output chart and metric
- How to use the AI chat assistant effectively, including how to get an API key

## Project Structure

| File | Purpose |
|------|---------|
| `model.py` | LP formulation and solver (PuLP/CBC) |
| `app.py` | Flask web server and plot generation |
| `templates/index.html` | Single-page UI with currency selector |
| `templates/guide.html` | Pedagogic user guide ("Get Flexing") |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build |
| `docker-compose.yml` | Orchestration (port 8180) |
