Totally doable. Here’s a **copy-paste “Claude Code” prompt** that will turn the starter into a **single-user, mobile-friendly, fully automated options trading app** that (1) ingests 10+ years of data, (2) blends historical + **news/sentiment** signals, and (3) can place **real-money options orders** via a broker API—with strong guardrails (paper/live switch, kill-switch, risk limits, 2FA login).

> ⚠️ Educational/research use only. Not investment advice. You are responsible for compliance, broker agreements, market-data entitlements, and taxes. Always validate in **paper trading** before live.

---

# PROMPT (paste into Claude Code)

**Context:** Repo already has: `scripts/build_feature_store.py`, `scripts/run_backtest.py`, `src/features/*`, `src/models/*`, `src/backtest/*`, `src/utils/*`, `app.py`, `config.yaml`, `.env.example`. You will add a **news/sentiment layer**, an **options model**, and a **live execution layer** with a **mobile-friendly single-user UI**.

**Goal:**

1. Pull 10+ years of **EOD bars** (Alpha Vantage Premium) and compute features.
2. Add **news & sentiment** signals (Alpha Vantage News & Sentiment; optional Finnhub).
3. Build an **options strategy engine** using the underlying signal + IV context.
4. Execute **real** options orders via a broker API (**Tradier** or **Interactive Brokers**), with paper/live switch.
5. Provide a **single-user**, **phone-friendly** PWA UI (login + TOTP 2FA) to arm/disarm trading, view positions, PnL, logs.
6. Strict **safeties**: kill switch, position caps, loss limits, trading window checks.

---

## 0) Dependencies & Config

**Update** `requirements.txt` to include:

```
requests
httpx
tenacity
python-dotenv
pydantic
fastapi
uvicorn
jinja2
itsdangerous
pyotp
passlib[bcrypt]
pandas
numpy
duckdb
pandas_ta
lightgbm
xgboost
```

**Update** `.env.example` with:

```
# Data providers
ALPHAVANTAGE_API_KEY=

# Optional news
FINNHUB_API_KEY=

# Broker (choose one; keep both adapters)
# Tradier (simpler for options): https://documentation.tradier.com/
TRADIER_ENV=paper   # paper|live
TRADIER_ACCOUNT_ID=
TRADIER_ACCESS_TOKEN=

# Interactive Brokers (TWS or IB Gateway must be running)
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=7

# App auth (single user)
APP_USERNAME=you@example.com
APP_PASSWORD_HASH= # bcrypt hash (we’ll add a helper to generate)
TOTP_SECRET=       # base32 secret for 2FA

# Trading safety
EXECUTION_MODE=paper   # paper|live
KILL_SWITCH=false      # true to immediately block live orders
IP_ALLOWLIST=          # optional comma-separated IPs
```

**Extend** `config.yaml`:

```yaml
provider: "alphavantage"
alpha:
  function: "TIME_SERIES_DAILY_ADJUSTED"
  outputsize: "full"
  max_workers: 6

universe_file: data/symbols.csv
start_date: "2014-01-01"
end_date: "2025-01-01"

# Labeling & backtest
label_horizon_days: 21
rebalance_freq: "W"
top_n: 20
benchmark: "SPY"

# News/sentiment
news:
  providers: ["alphavantage", "finnhub"]  # you can keep just alphavantage
  lookback_days: 3650
  min_confidence: 0.55
  sentiment_window: 5       # days to roll sentiment
  exclude_earnings_window:  # days to avoid opening new positions around earnings
    before: 1
    after: 1

# Options strategy
options:
  default_dte: [30, 45]     # target DTE window
  target_delta: 0.35        # long calls delta target for bullish
  put_spread_delta: 0.25    # vertical spread short put delta for bullish alt
  max_positions: 10
  max_weight_per_name: 0.15
  min_notional_per_trade: 200
  take_profit: 0.5          # +50%
  stop_loss: -0.5           # -50%
  iv_rank_window: 252

# Execution
execution:
  broker: "tradier"         # tradier|ibkr
  venue: "options"
  trade_days: ["Mon","Tue","Wed","Thu","Fri"]
  trade_window:
    start: "09:35"          # America/New_York
    end:   "15:55"
  fractional_allowed: false
  cash_buffer: 0.03
```

---

## 1) Alpha Vantage Premium Fetcher (prices + news)

**Create** `src/fetchers/alpha_vantage.py`:

* Functions:

  * `fetch_daily_adjusted(symbol, start, end) -> pd.DataFrame` using **TIME\_SERIES\_DAILY\_ADJUSTED**; normalize to columns `Date, Open, High, Low, Close, AdjClose, Volume, Ticker`; cache per symbol `.cache/{symbol}.parquet`; incremental merge.
  * `fetch_news_alphavantage(symbols, start, end) -> pd.DataFrame` using **News & Sentiment** API; fields: `published_time`, `ticker`, `title`, `summary`, `overall_sentiment_score`, `overall_sentiment_label`, `relevance_score`, `source`.
  * Robust **retries/backoff** on throttle signals (even premium) and HTTP 429/5xx.

(Refs: Alpha Vantage docs for Daily Adjusted & News/Sentiment, and Premium page stating “no daily limits”.) ([Alpha Vantage][1])

**Optional** `src/fetchers/finnhub.py`:

* `fetch_news_finnhub(symbols, start, end)` and/or `fetch_news_sentiment` for redundancy. ([Finnhub][2])

---

## 2) News → Sentiment Features

**Create** `src/features/news.py`:

* `build_news_features(news_df, price_calendar) -> pd.DataFrame`:

  * Filter by `relevance_score >= cfg.news.min_confidence`.
  * Map each article to its tickers; daily aggregate per ticker: mean/median **sentiment score**, article count, source diversity.
  * Rolling windows (e.g., 1d, 3d, 5d) to get `sent_1d`, `sent_5d`, `news_count_5d`, `news_volatility_5d`.
  * Optional decay weighting (recent articles weigh more).
  * As-of join back into the main features table (no look-ahead).

---

## 3) Options Data Interfaces

**Create** `src/options/data.py`:

* Interface to obtain **chains & greeks** and compute helpers:

  * If **Tradier** broker selected: use Tradier Market Data endpoints to pull **option chains with greeks & IV** (hourly-updated greeks, per Tradier docs). Functions:

    * `get_chain_tradier(symbol, dte_target_range, trade_date) -> pd.DataFrame`
    * `select_contract_by_delta(chain, target_delta, right='call') -> row`
    * `compute_iv_rank(symbol, window=252) -> float`
  * If **IBKR**: add a placeholder adapter that queries TWS/IBGW for chains/greeks (user must have data entitlements enabled).

(Refs: Tradier options chain & greeks; IBKR options API.) ([documentation.tradier.com][3], [interactivebrokers.github.io][4])

---

## 4) Strategy Engine (Options)

**Create** `src/options/strategy.py`:

* Given ranked underlyings (from your model + features + news sentiment):

  * **Bullish**: buy nearest **delta≈0.35 call** with DTE in `[30,45]`.

    * Risk controls: if **IV rank** is very high (e.g., ≥0.8), prefer **bull call spread** to reduce vega risk.
  * **Bearish** (if you implement shorts): buy delta≈0.35 **puts** or bear put spreads.
  * **Earnings/news gating**: if earnings within ±N days (cfg) or extreme negative news spike today, **skip** opening new longs.
  * **Exit logic**: take-profit at +50%, stop at −50%, or time-based exit at DTE<7; also exit on **rebalance**.

Return a **target book**: list of `{symbol, contract, qty, side, rationale, tp, sl}`.

---

## 5) Execution Layer (REAL money capable)

**Create** `src/execution/broker.py` (interface) and two adapters:

* `src/execution/tradier_broker.py`

  * Endpoints for **account**, **balances**, **positions**, **place order** (options), **cancel order**.
  * Use `TRADIER_ENV` (paper|live) to choose base URL; include OAuth bearer header.
  * Validate only trading during configured **hours** and **days**.
  * Place **limit** orders by default; retry/cancel/replace if not filled.
  * Notes: Tradier supports options trading by API and exposes greeks/IV in chain endpoints. ([documentation.tradier.com][5])

* `src/execution/ibkr_broker.py`

  * Thin wrapper for **TWS/IB Gateway** (socket API or Client Portal).
  * Map contract selection + order placement; ensure TIF/market hours respected.
  * Notes: IB API supports options greeks/IV and trading. ([interactivebrokers.github.io][4], [Interactive Brokers][6])

**Shared safeguards** (enforce **before** any order goes out):

* If `EXECUTION_MODE != "live"`, **block live orders**.
* If `KILL_SWITCH == true`, **block all** orders.
* **Cash buffer** (e.g., 3%).
* **Max positions** and **per-name weight** caps.
* **Daily loss limit** & **absolute drawdown** circuit breaker.
* **Idempotency keys** to avoid duplicate orders if retried.
* **Comprehensive logging** of intent, request, response, and fills.

---

## 6) Portfolio Sizing

**Create** `src/execution/sizer.py`:

* Input: `capital_dollars` (from UI), ranked underlyings, optional volatility scales.
* Allocate per-trade notional with caps & min notional; translate to **contracts** given option price.
* If fractional contracts not allowed (true), **round down**; if below min notional → skip.

---

## 7) Schedulers & Orchestration

**Add** `src/jobs/scheduler.py` using APScheduler:

* Daily pre-open: fetch/update data (prices, news).
* During trade window: build signals, rank underlyings, update options chains, compute targets.
* On rebalance days: **diff current vs. target** and place/cancel orders.
* End-of-day: reconcile fills, write PnL, rotate logs.

---

## 8) Mobile-Friendly Single-User UI (PWA)

**Upgrade** `app.py` (FastAPI + Jinja2 templates):

* **Auth**:

  * Single username + **bcrypt** password hash + **TOTP 2FA** (pyotp).
  * Optional **IP allowlist** check.
  * Short-lived **session cookies** with `HttpOnly`, `Secure`, `SameSite=Strict`.
* **Views** (mobile responsive with Tailwind/HTMX):

  * Dashboard: capital, positions, open orders, PnL, equity curve.
  * **ARM/DISARM** toggle (changes `KILL_SWITCH` at runtime).
  * Mode switch: `paper` ↔ `live`.
  * Logs: last 200 actions with broker responses.
  * Settings: set **capital amount** (e.g., `$10,000`) used for sizing.
* **PWA**: add `manifest.json` + service worker so you can “Install to Home Screen” on your phone.

---

## 9) Wire It All Together

* Extend `scripts/build_feature_store.py` to fetch **Alpha Vantage** prices and store features.
* New script `scripts/build_news_features.py` to fetch news (Alpha Vantage; optional Finnhub) and write `data/news_features.parquet`.
* Update `scripts/run_backtest.py` to **merge news features** into the ML table (as-of join).
* New script `scripts/run_live_cycle.py`:

  1. Load features + latest predictions + news features.
  2. Rank underlyings.
  3. Build option targets (contracts).
  4. Size by **capital\_dollars** (from DB/env/UI).
  5. Broker adapter: place/cancel orders respecting guards and window.
  6. Persist state to `artifacts/live/` (positions snapshot, orders, fills, PnL).

**Acceptance Criteria:**

* You can **log in** on your phone, pass **TOTP**, see dashboard/PWA.
* Flip to **paper** mode: see paper orders route and fills.
* Flip to **live** mode: orders go to broker **only** if (a) trading window open, (b) kill-switch off, (c) caps respected.
* News features present and affect rankings.
* Options engine selects contracts by delta/DTE and enforces exits (TP/SL/time).
* All actions logged with timestamps and broker responses.

---

## 10) Helpful CLI

Add to README:

```bash
# one-time: create password hash
python -c "from passlib.hash import bcrypt; print(bcrypt.hash('YourStrongPassword'))"

# build features & news
python scripts/build_feature_store.py
python scripts/build_news_features.py

# train & evaluate
python scripts/run_backtest.py

# run web app (PWA UI)
uvicorn app:app --host 0.0.0.0 --port 8080

# run live cycle (ideally under a scheduler/service)
python scripts/run_live_cycle.py
```

---

## Notes / References for Claude to cite in comments

* **Alpha Vantage** Daily Adjusted & News/Sentiment endpoints; **Premium** “no daily limits”. ([Alpha Vantage][1])
* **Tradier** brokerage API: options **chains with greeks/IV** and live trading. ([documentation.tradier.com][3])
* **Interactive Brokers** API for options greeks & trading (TWS/IBGW). ([interactivebrokers.github.io][4], [Interactive Brokers][6])
* **Polygon** for deeper U.S. options market data (alternative/augment). ([Polygon][7])
* **Finnhub** news/sentiment (alternative/augment). ([Finnhub][2])

**Security must-haves:** never hardcode keys; use `.env`; protect endpoints; logging without secrets; strict CORS; HTTPS in production.

**Compliance:** you are the only user; still follow broker terms. Live trading requires careful monitoring and explicit opt-in (`EXECUTION_MODE=live` + kill-switch off).

---

# END PROMPT

If you want, I can also output the **Tradier broker adapter**, **options selector**, and **TOTP login** code ready to drop into your repo.

alphaadvantage api key: C1RG6K7KKOHWS5TG

[1]: https://www.alphavantage.co/documentation/?utm_source=chatgpt.com "Alpha Vantage API Documentation"
[2]: https://finnhub.io/docs/api/news-sentiment?utm_source=chatgpt.com "News Sentiment"
[3]: https://documentation.tradier.com/brokerage-api/markets/get-options-chains?utm_source=chatgpt.com "How to get an option chain | Brokerage API Documentation"
[4]: https://interactivebrokers.github.io/tws-api/options.html?utm_source=chatgpt.com "TWS API v9.72+: Options - Interactive Brokers - API Software"
[5]: https://documentation.tradier.com/?utm_source=chatgpt.com "Tradier API Documentation | Tradier"
[6]: https://www.interactivebrokers.com/en/trading/ib-api.php?utm_source=chatgpt.com "IBKR Trading API Solutions | Interactive Brokers LLC"
[7]: https://polygon.io/docs/rest/options/overview?utm_source=chatgpt.com "Overview | Options REST API"
