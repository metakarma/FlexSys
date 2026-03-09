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

Each LDC block is split into two sub-blocks — **Low ZVC** and **High ZVC** — to capture the intermittent nature of renewable generation. Parameters per block:

| Parameter | Description |
|-----------|-------------|
| **% Hours** | Fraction of block hours in each sub-block (must sum to 100) |
| **% ZVC Availability** | Fraction of installed ZVC capacity available in each sub-block |

This creates intra-block price differentials: prices are higher in Low-ZVC sub-blocks (scarcity) and lower in High-ZVC sub-blocks (abundance). Storage can arbitrage these differences by charging when ZVC is plentiful and discharging when it is scarce.

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
- **Gas**: 13.2 GW built (peaking in Low-ZVC sub-blocks)
- **Storage**: 14.1 GW power / 6,499 GWh energy (charges in cheap High-ZVC periods, discharges in expensive Low-ZVC periods)
- **Clearing prices**: £441/MWh (WP Low), £107/MWh (WP High), £107/MWh (Shoulder Low), £82/MWh (Shoulder High), £80/MWh (Low Demand Low), £64/MWh (Low Demand High)

All three technologies coexist. Storage performs seasonal and within-block arbitrage — charging in Low Demand High ZVC (£64/MWh) and discharging in Winter Peak Low ZVC (£441/MWh).

## Project Structure

| File | Purpose |
|------|---------|
| `model.py` | LP formulation and solver (PuLP/CBC) |
| `app.py` | Flask web server and plot generation |
| `templates/index.html` | Single-page UI with currency selector |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build |
| `docker-compose.yml` | Orchestration (port 8180) |
