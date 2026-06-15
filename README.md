# Prediction Market Arbitrage Scanner

Real-time scanner that monitors **Polymarket** and **Kalshi** for the same event
priced differently across platforms. It matches markets with a two-stage
similarity filter confirmed by a Random Forest classifier, surfaces arbitrage
opportunities, and can optionally place hedged trades — though execution is
**off by default** and dry-runs unless you explicitly arm it.

> ⚠️ **Financial risk.** Prediction-market trading can lose money. "Arbitrage"
> opportunities frequently disappear before both legs fill, markets may resolve
> differently than they appear to mirror, fees and slippage erode edge, and
> bugs can place real orders. Nothing here is financial advice. Run in
> `DRY_RUN` first, understand every line before arming live execution, and never
> deploy money you cannot afford to lose. You are solely responsible for any
> trades placed with this software.

## Architecture

```
        +--------------+              +--------------+
        |  Polymarket  |              |    Kalshi    |
        |  Gamma API   |              | trade-api v2 |
        +------+-------+              +------+-------+
               | fetch                       | fetch (RSA-PSS signed if key set)
               +-------------+---------------+
                             v
                      +-------------+
                      |    MATCH    |   Jaccard > 0.15  ->  TF-IDF cosine > 0.35
                      +------+------+
                             v
                      +-------------+
                      |  CLASSIFY   |   Random Forest (cold-start: accept gate)
                      +------+------+
                             v
                      +-------------+
                      |    ARB?     |   yes_leg + no_leg cost < 1 - threshold
                      +------+------+
                             v
              +--------------+--------------+
              v                             v
       +-------------+              +----------------+
       |    ALERT    |              |    EXECUTE     |  size: 2% / $500 cap
       |  stdout +   |              |  DRY_RUN +     |  gated: ENABLE_EXECUTION
       | opps.log    |              |  ENABLE_EXEC   |  --> pnl.log
       +------+------+              +----------------+
              v
       +-------------+
       |  DASHBOARD  |   Flask on :5001  (reads opportunities.log + pnl.log)
       +-------------+

  resolved markets --> auto-label --> labels.csv --> weekly Random Forest retrain
```

## File structure

```
arb/
├── scanner.py        # entrypoint + fetch + arb math + main loop
├── auth.py           # KalshiAuth: RSA-PSS request signing
├── matching.py       # two-stage Jaccard -> TF-IDF cosine matcher
├── classifier.py     # Random Forest match classifier
├── execution.py      # alerting, position sizing, P&L, gated order placement
├── labeling.py       # auto-labeling from resolution + weekly retrain
├── dashboard.py      # Flask dashboard server (:5001)
├── dashboard.html    # dashboard UI
├── requirements.txt  # dependencies
└── .env.example      # environment variable template
```

Runtime artifacts (git-ignored): `opportunities.log`, `pnl.log`,
`predictions.log`, `labels.csv`, `model.pkl`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in the values you need
```

### Kalshi API key (RSA-PSS)

Reads work without a key (public data). The trading API requires an RSA keypair:

1. Generate a keypair:
   ```bash
   openssl genrsa -out kalshi_private.pem 2048
   openssl rsa -in kalshi_private.pem -pubout -out kalshi_public.pem
   ```
2. Upload `kalshi_public.pem` in your Kalshi account settings and copy the
   **Key ID** it gives you.
3. In `.env`:
   ```
   KALSHI_KEY_ID=your-key-id-uuid
   KALSHI_PRIVATE_KEY_PATH=./kalshi_private.pem
   ```

`*.pem` is git-ignored — your private key never gets committed. Each request is
signed with `RSA-PSS` (SHA-256, MGF1, max salt length) over
`timestamp_ms + METHOD + path`.

## Running

```bash
python scanner.py            # continuous scan loop (10s interval)
python scanner.py --once     # single scan, then exit (handy for testing)

python dashboard.py          # http://127.0.0.1:5001
```

The scanner prints opportunities to stdout and appends them to
`opportunities.log`; the dashboard reads that plus `pnl.log`.

## DRY_RUN vs. live trading

Execution is protected by **two independent switches**, and the safe state is
the default:

| Switch              | Default | Live orders require |
| ------------------- | ------- | ------------------- |
| `DRY_RUN`           | `true`  | `false`             |
| `ENABLE_EXECUTION`  | unset   | `true`              |

- **Default (recommended):** `execute_arb` only *logs* the trades it would have
  placed and records them to `pnl.log` flagged `dry_run: true`. No order book is
  touched.
- **Live:** you must set **both** `ENABLE_EXECUTION=true` **and**
  `DRY_RUN=false`, *and* provide the relevant keys (`KALSHI_PRIVATE_KEY_PATH`,
  `POLYMARKET_PRIVATE_KEY`). Even then, a trade is only placed when the net edge
  after estimated fees exceeds `MIN_EXECUTION_EDGE`.

Position sizing caps every arb at `min(2% of BANKROLL, MAX_POSITION_SIZE)`
(default $500). With `BANKROLL` unset, sizing returns zero and nothing executes.

> The Polymarket live leg additionally needs the optional `py-clob-client`
> package (`pip install py-clob-client`); without it the scanner stays in
> dry-run for that leg.

**Live execution is experimental and NOT atomic.** The two legs are placed
sequentially with no fill confirmation or automatic unwind. If the first
(Polymarket) leg fills and the second (Kalshi) leg fails, you may be left
holding a one-sided, **unhedged** position — the scanner logs a loud
`[exec CRITICAL]` alert and flags the P&L row `unhedged: true`, but **you must
flatten it manually**. As an extra guard, live orders are refused unless the
Random Forest is trained and the match probability is `>= LIVE_MATCH_PROB`
(default 0.9), so weak cold-start matches can never trade live.

## Environment variables

| Variable                  | Default | Purpose                                          |
| ------------------------- | ------- | ------------------------------------------------ |
| `KALSHI_KEY_ID`           | —       | Kalshi API key id (UUID)                         |
| `KALSHI_PRIVATE_KEY_PATH` | —       | Path to the RSA private key PEM                   |
| `POLYMARKET_PRIVATE_KEY`  | —       | Wallet key for Polymarket CLOB orders            |
| `ENABLE_EXECUTION`        | unset   | Must be `true` to place real orders              |
| `DRY_RUN`                 | `true`  | Log instead of trading                           |
| `MIN_EXECUTION_EDGE`      | `5.0`   | Minimum net edge % (after fees) to execute       |
| `MAX_POSITION_SIZE`       | `500`   | Max dollars deployed per arb                     |
| `BANKROLL`                | `0`     | Total capital for position sizing                |
| `POLL_INTERVAL`           | `10`    | Seconds between scans                            |
| `JACCARD_THRESHOLD`       | `0.15`  | Stage-1 match threshold                          |
| `COSINE_THRESHOLD`        | `0.35`  | Stage-2 match threshold                          |
| `MATCH_PROB_THRESHOLD`    | `0.5`   | Min classifier probability to accept a match     |

## How matching works

1. **Jaccard** token overlap is a cheap pre-filter (`> 0.15`).
2. **TF-IDF cosine** similarity reranks the survivors (`> 0.35`).
3. The **Random Forest** classifier decides if the pair is truly the same event.
   Before any labels exist it cold-starts by trusting the two-stage gate; as
   markets resolve, `labeling.py` auto-writes labels to `labels.csv` and the
   model is retrained weekly.

Only clean two-sided binaries are considered. Multi-outcome / grouped markets
(Kalshi MVE combos, comma-separated "yes A, yes B, ..." titles) and degenerate
or stale quotes (per-venue yes+no below `MIN_BOOK_SUM`) are filtered out before
matching — otherwise they manufacture phantom arbitrage. Candidate pairs are
also rejected when the two questions disagree on direction polarity
(above/below, win/lose, ...) or cite disjoint numbers (different thresholds or
dates). The public Kalshi feed prices vary by what's currently liquid; an
authenticated key gives the most complete book.

Residual limitation: the lexical matcher + polarity/number guards cannot, on
their own, tell apart same-template different-entity markets (e.g. "Will Trump
win?" vs "Will Biden win?"). The Random Forest learns to reject these as
auto-labels accumulate; until then the `MATCH_PROB_THRESHOLD` gate and the
trained-model requirement for live orders keep such pairs out of execution.
