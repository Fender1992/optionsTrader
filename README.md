# 🚀 Maximum Profit Options Trading

**High-frequency day trading system optimized for explosive account growth**

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Alpha Vantage API key in .env file
# (API key already configured: C1RG6K7KKOHWS5TG)

# 3. Run the app
python app.py
```

Open http://localhost:8080 in your browser and click "Start Maximum Profit Trading"

## 🎯 What This Does

This system executes high-frequency options day trading with maximum aggression for explosive account growth:

- **8-25 trades per day** depending on account size
- **30% position sizing** for small accounts, scaling down as account grows  
- **40% profit targets** for maximum returns
- **0-7 DTE options only** for maximum gamma exposure
- **Commission-free execution** for modern brokers
- **100% profit reinvestment** for compound growth

## 📊 Performance Targets

| Account Size | Daily Trades | Position Size | Profit Target | Daily Profit Potential |
|--------------|--------------|---------------|---------------|------------------------|
| $10-$50      | 8            | 30%           | 40%           | $9.60 - $48.00        |
| $50-$500     | 10           | 15%           | 28%           | $21.00 - $210.00      |
| $500-$5K     | 18           | 10%           | 22%           | $198.00 - $1,980.00   |
| $5K+         | 25           | 8%            | 20%           | $2,000.00+            |

## 🕐 Trading Windows

The system automatically trades during peak opportunities:

- **09:30-10:30**: Opening volatility (4 trades, 1.3x profit multiplier)
- **12:00-13:00**: Lunch reversal (2 trades, 1.1x multiplier)
- **14:30-15:30**: Power hour (4 trades, 1.2x multiplier)
- **15:30-16:00**: Close scalping (2 trades, 1.5x multiplier)

## 🎯 Target Instruments

**Primary:** SPY, QQQ (most liquid)
**Secondary:** AAPL, MSFT, NVDA, AMZN, TSLA

## 📈 Growth Projections

**Example: $10 starting capital**
- **1 week**: $31.51 (215% gain)
- **1 month**: $1,559 (15,495% gain)
- **2 months**: $243,191 (2.4M% gain)

*Based on 8 trades/day, 30% position size, 55% win rate*

## ⚙️ Configuration

All parameters are automatically configured in `config/max_profit_config.json`:

```json
{
  "account_size_tiers": {
    "tiny_account": {
      "range": [10, 50],
      "optimal_trades_per_day": 8,
      "position_size_pct": 0.30,
      "profit_target_pct": 0.40,
      "stop_loss_pct": 0.25
    }
  }
}
```

## 🏗️ Architecture

```
├── app.py                              # Web interface
├── src/
│   ├── strategies/
│   │   └── max_profit_day_trading.py   # Core trading strategy  
│   ├── execution/
│   │   └── profit_maximizer.py         # Trading execution engine
│   ├── data/
│   │   └── alpha_vantage_fetcher.py    # Market data
│   ├── features/
│   │   └── advanced_technical.py       # 60+ technical indicators
│   ├── models/
│   │   └── advanced_ml.py              # ML signal generation
│   └── options/
│       └── advanced_strategy.py        # Options strategies
├── config/
│   └── max_profit_config.json          # Trading parameters
└── scripts/
    ├── run_max_profit_trading.py       # Command-line runner
    └── test_max_profit_strategy.py     # Strategy testing
```

## 🚨 Risk Warnings

⚠️ **This system uses AGGRESSIVE position sizing for maximum profit potential**

⚠️ **High frequency trading may result in substantial losses**

⚠️ **Only use money you can afford to lose**

⚠️ **Past performance does not guarantee future results**

## 🔧 Environment Variables

```bash
INITIAL_CAPITAL=10.0                    # Starting capital
ALPHA_VANTAGE_API_KEY=C1RG6K7KKOHWS5TG # Real Alpha Vantage API key (required)
```

## 📱 Usage

### Web Interface (Recommended)
```bash
python app.py
# Navigate to http://localhost:8080
```

### Command Line
```bash
python scripts/run_max_profit_trading.py
```

### Testing
```bash
python scripts/test_max_profit_strategy.py
```

## 🎯 Key Features

- **Maximum Profit Focus**: All conservative logic removed
- **Dynamic Position Sizing**: Scales with account growth
- **High-Frequency Execution**: Up to 25 trades/day
- **Smart Risk Management**: Controlled losses, explosive gains
- **Real-Time Web Interface**: Monitor and control trading
- **Automatic Compounding**: 100% profit reinvestment

## 📝 License

MIT License - Trade at your own risk!