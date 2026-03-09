# Peaky Flexers — Electricity Market LP Optimiser

A web application that solves a welfare-maximising linear program for electricity market price formation. The model simultaneously determines **optimal capacity investment** and **dispatch** for three supply technologies — zero-variable-cost renewables (ZVC), gas peakers, and battery storage — across a three-block Load Duration Curve with **ZVC intermittency sub-blocks**, flexible (shiftable) and expandable demand.

A currency selector (£/$/€) at the top of the UI updates all labels and plot annotations.

## Quick Start

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

### ZVC Intermittency (Sub-Blocks)

Each LDC block is split into two sub-blocks — **Low Renewables** and **High Renewables** — to capture the intermittent nature of renewable generation. Parameters per block:

| Parameter | Description |
|-----------|-------------|
| **% Hours** | Fraction of block hours in each sub-block (must sum to 100) |
| **% Renewables Availability** | Fraction of installed ZVC capacity available in each sub-block |

This creates intra-block price differentials: prices are higher in Low Renewables sub-blocks (scarcity) and lower in High Renewables sub-blocks (abundance). Storage can arbitrage these differences by charging when renewables are plentiful and discharging when they are scarce.

### Supply Technologies (endogenous capacity)

The model determines the **welfare-maximising capacity mix**. No capacities are pre-set. Each technology has:

| Technology | Default Capital Cost | Variable Cost |
|-----------|---------------------|---------------|
| **ZVC** (renewables/nuclear) | £450/kW/year | £0/MWh |
| **Gas peaker** | £40/kW/year | £80/MWh |
| **Storage** | £10/kW/year (power) + £0.04/kWh/year (energy) | None — cost comes from efficiency losses |

### Storage

Storage round-trip efficiency is modelled physically (default 95%). Energy losses during charging/discharging create an implicit cycling cost that depends on charging and discharging prices — this is an **output**, not an input. Storage dynamics are tracked sequentially through all 6 sub-blocks with cyclical operation (end-of-year level = start-of-year level).

### Demand

Demand is divided into High/Mid/Low value tiers per block, each with a Value of Lost Load (VoLL) and a shift cost. Demand can only shift **downward** (Winter Peak → Shoulder → Low Demand).

Default shift costs:

| Block | High | Mid | Low |
|-------|------|-----|-----|
| Winter Peak | £1,000/MWh | £500/MWh | £50/MWh |
| Shoulder | £800/MWh | £40/MWh | £8/MWh |
| Low Demand | £20/MWh | £10/MWh | £5/MWh |

Expandable demand activates when the price drops below its value:

| Block | Quantity | Activation Value |
|-------|----------|-----------------|
| Winter Peak | 100 GW | £40/MWh |
| Shoulder | 500 GW | £30/MWh |
| Low Demand | 1,000 GW | £20/MWh |

### Prices

Shadow prices on the energy-balance constraints give **market-clearing prices** for each sub-block (6 prices total).

## Default Scenario Results

With defaults (ZVC £450/kW/yr, Gas £40/kW/yr + £80/MWh, Storage £10/kW + £0.04/kWh, 95% eff):

- **ZVC**: 18.3 GW built
- **Gas**: 13.2 GW built (peaking in Low Renewables sub-blocks)
- **Storage**: 14.1 GW power / 6,499 GWh energy (charges in cheap High Renewables periods, discharges in expensive Low Renewables periods)
- **Clearing prices**: £441/MWh (WP Low Renewables), £107/MWh (WP High Renewables), £107/MWh (Shoulder Low Renewables), £82/MWh (Shoulder High Renewables), £80/MWh (Low Demand Low Renewables), £64/MWh (Low Demand High Renewables)

All three technologies coexist. Storage performs seasonal and within-block arbitrage — charging in Low Demand High Renewables (£64/MWh) and discharging in Winter Peak Low Renewables (£441/MWh).

## AI Chat Assistant

An expandable chat panel (toggle via the **AI Chat** button, top-right) lets users interrogate model results conversationally using Anthropic's Claude. Features:

- **Model selector** — defaults to Claude Opus 4.6.
- **API key input** — your Anthropic key is stored locally in the browser and never sent to the server beyond proxying requests.
- **Full model context** — every message is sent alongside the complete model source code (`model.py`), the current input parameters, and the latest optimisation results, so the AI can reason about your specific scenario.
- **Markdown & LaTeX rendering** — responses are formatted with Markdown (via `marked.js`) and mathematical equations are rendered with KaTeX.
- **Clear Chat** button to reset the conversation history.

This gives users a natural-language interface to ask questions such as *"Why is Winter Peak price so much higher than Shoulder?"* or *"What happens if I double ZVC capital cost?"* — similar to working with an analyst who has the model open in front of them.

## Project Structure

| File | Purpose |
|------|---------|
| `model.py` | LP formulation and solver (PuLP/CBC) |
| `app.py` | Flask web server and plot generation |
| `templates/index.html` | Single-page UI with currency selector |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build |
| `docker-compose.yml` | Orchestration (port 8180) |
