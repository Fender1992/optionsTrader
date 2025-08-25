# Options Trading Platform

A fully automated options trading platform with machine learning signals, news sentiment analysis, and real-money trading capabilities through Tradier API.

## Features

- **Data Integration**: Alpha Vantage API for historical prices and news sentiment
- **Machine Learning**: LightGBM/XGBoost models for signal generation
- **Options Strategies**: Delta-based option selection with spreads for high IV
- **Risk Management**: Position limits, concentration limits, stop-loss/take-profit
- **Live Trading**: Tradier broker integration with paper/live modes
- **Mobile UI**: Progressive Web App with authentication and 2FA
- **Safety Controls**: Kill switch, trading hours, loss limits

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with:
- Alpha Vantage API key (already included)
- Tradier API credentials (get from https://developer.tradier.com)
- App credentials (generate with helper script)

### 3. Generate Authentication Credentials

```bash
# Generate password hash
python scripts/generate_password.py --password

# Generate TOTP secret for 2FA
python scripts/generate_password.py --totp
```

Add the generated values to your `.env` file.

### 4. Build Features & Train Model

```bash
# Fetch data and build features
python scripts/build_feature_store.py

# Train model and run backtest
python scripts/run_backtest.py
```

### 5. Run the Application

```bash
# Start web app
uvicorn app:app --host 0.0.0.0 --port 8080

# Or with auto-reload for development
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Access the app at http://localhost:8080

### 6. Run Live Trading Cycle (Optional)

```bash
# Manual execution
python scripts/run_live_cycle.py

# Or use the scheduler (runs automatically)
# The app starts the scheduler on startup
```

## Project Structure

```
OptionsTrading/
├── src/
│   ├── fetchers/        # Data fetching (Alpha Vantage)
│   ├── features/        # Feature engineering
│   ├── models/          # ML models
│   ├── options/         # Options data & strategies
│   ├── execution/       # Broker integration
│   └── jobs/            # Scheduler
├── scripts/             # CLI scripts
├── templates/           # HTML templates
├── static/              # Static files
├── data/                # Data storage
├── artifacts/           # Model & logs
├── app.py               # FastAPI application
├── config.yaml          # Configuration
└── requirements.txt     # Dependencies
```

## Configuration

Edit `config.yaml` to adjust:
- Universe of symbols
- Trading parameters (position limits, sizing)
- Options strategy settings
- Execution windows and safety limits

## Trading Modes

1. **Paper Mode** (default): Simulated trading with Tradier sandbox
2. **Live Mode**: Real money trading (requires explicit activation)

## Safety Features

- **Kill Switch**: Instantly blocks all order placement
- **Trading Hours**: Only trades during configured market hours
- **Position Limits**: Max 10 positions, 15% per name
- **Loss Limits**: Daily -2%, absolute -10% drawdown
- **Cash Buffer**: Maintains 3% cash reserve

## Mobile Access

The app is a Progressive Web App (PWA) that can be installed on mobile:

1. Open the app in mobile browser
2. Click "Add to Home Screen"
3. Access like a native app

## API Endpoints

- `POST /api/toggle-kill-switch` - Enable/disable trading
- `POST /api/set-capital` - Update trading capital
- `POST /api/toggle-mode` - Switch paper/live mode
- `GET /api/positions` - Current positions
- `GET /api/logs` - Trading logs
- `GET /api/pnl` - P&L history
- `POST /api/run-job` - Manually trigger jobs

## Security

- Single-user authentication with email/password
- Optional TOTP 2FA
- Session-based auth with HttpOnly cookies
- Optional IP allowlist
- All sensitive data in environment variables

## Disclaimer

**IMPORTANT**: This software is for educational purposes. Options trading involves substantial risk of loss. Past performance does not guarantee future results. You are responsible for:
- Understanding options trading risks
- Compliance with regulations
- Tax obligations
- Broker terms of service

Always test thoroughly in paper mode before considering live trading.

## Support

For issues or questions, please review the code documentation or consult the configuration files.