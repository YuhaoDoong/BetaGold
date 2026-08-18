**English** · [中文](README.zh-CN.md)

# GoldDash

GoldDash is a research-oriented system for studying precious-metals signals across daily and intraday horizons. It combines market and macroeconomic features, machine-learning range estimates, volatility-aware filters, options strategy research, backtesting, and a Streamlit dashboard.

> **Research use only.** GoldDash is an experimental research codebase, not a trading service or investment recommendation. Backtested results do not represent live performance and can be materially affected by data quality, transaction costs, model selection, and market-regime changes.

## Project at a glance

| Area | Current scope |
|---|---|
| Instruments | Gold and silver research, primarily through GLD, SLV, GC=F, and SI=F |
| Forecasting | Multi-horizon range estimation using LSTM/Transformer-based experiments |
| Signal research | Trend, realized-volatility, macro, regime, intraday-confirmation, and options-derived filters |
| Strategy studies | Futures, long calls, put credit spreads, long straddles, and short-volatility structures |
| Evaluation | Walk-forward analysis, multi-window backtests, calibration checks, and per-asset configuration |
| Interface | Streamlit dashboard plus command-line research scripts |

## Research workflow

```text
Market and macro data
        ↓
Feature engineering and model estimates
        ↓
Daily candidate signals
        ↓
Regime, trend, volatility, and intraday confirmation
        ↓
Strategy candidates and risk parameters
        ↓
Backtest / paper-position ledger / dashboard
```

The repository separates the user interface from the research modules:

- `core/` contains data access, feature logic, models, signal generation, calibration, and position-ledger code.
- `scripts/` contains data preparation, training, backtesting, diagnostics, and parameter studies.
- `tests/` contains offline unit and regression tests.
- `docs/` contains the detailed architecture, model, strategy, and experiment notes.

## Main capabilities

### Multi-source features

GoldDash combines price, volatility, macroeconomic, cross-asset, and market-structure inputs. The current research pipeline includes GLD/SLV and COMEX futures data, realized-volatility measures, GVZ and related volatility indicators, rates and dollar factors, technical indicators, and selected options/open-interest features.

### Multi-horizon modelling

The project includes daily and hourly modelling experiments. Range estimates are treated as inputs to a broader decision pipeline rather than as standalone price forecasts. Candidate signals pass through trend, volatility, regime, freshness, and intraday-confirmation checks.

### Strategy and risk research

The strategy layer maps filtered signals to instrument-specific research configurations. Position sizing, holding periods, exits, and option structures are evaluated separately by asset and regime. These modules are simulations; they do not place live orders.

### Auditability

Signal histories, calibration records, dashboard snapshots, and position-ledger logic are designed to make a result traceable from source data to the displayed decision state.

## Quick start

### 1. Create an environment

```bash
git clone https://github.com/YuhaoDoong/BetaGold.git
cd BetaGold

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data

The setup script downloads public market data, optionally retrieves FRED series, builds features, and can train the range model:

```bash
# Market data and features only
python scripts/setup_data.py --no-train

# Include FRED data; obtain a key from https://fred.stlouisfed.org/
python scripts/setup_data.py --fred-key YOUR_FRED_API_KEY --no-train

# Train after data preparation
python scripts/setup_data.py --train-only
```

The setup script writes to the repository-local `data/` directory. Set `data_root: "data"` in `config.yaml` for this layout. The tracked configuration currently reflects the maintainer's local research environment, so a fresh clone must update this value before launching the dashboard.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Some views require trained model artifacts, option-history files, or paper-position ledgers that are not distributed with the repository.

## Validation and tests

Run the isolated test directory with:

```bash
pytest -q tests
```

The suite mixes self-contained unit tests with regression checks tied to the maintainer's historical market snapshots. A fresh clone can therefore report missing-data failures until the corresponding local datasets are prepared. Experimental files under `scripts/` are intentionally excluded from automatic test discovery through `pytest.ini`.

For research results, prefer walk-forward and multi-window validation over a single optimized backtest. Useful entry points include:

```bash
python scripts/backtest_pipeline.py all
python scripts/monthly_retune.py --dry-run
python scripts/multi_window_validate.py
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — modules, data flow, and deployment model
- [Models](docs/MODELS.md) — range models, calibration, and regime classification
- [Strategies](docs/STRATEGIES.md) — strategy definitions and risk assumptions
- [Experiments](docs/EXPERIMENTS.md) — research log and historical comparisons

## Data and reproducibility notes

- Public sources can change schemas, revise observations, impose rate limits, or return delayed data.
- Model artifacts, large datasets, private brokerage exports, and credentials are not committed.
- Several legacy research scripts still assume the maintainer's local data layout. Treat them as experiment records unless their paths have been configured for your environment.
- Reported win rates or returns are conditional on the documented sample, execution model, and costs; they must not be interpreted as expected future performance.
- Any use with real capital requires independent validation, realistic slippage/fee modelling, position limits, and operational safeguards.

## Security

Do not commit credentials. `.env`, key files, and common secret-file names are excluded by `.gitignore`. Use environment variables or a local secret manager for brokerage credentials. The optional FRED key saved by `setup_data.py` is stored in `.fred_key`, which is also ignored.

## Project status

GoldDash is an active personal research project. Interfaces and experiment scripts may change as datasets and validation protocols evolve. Contributions and reproducibility reports are welcome through GitHub issues.
