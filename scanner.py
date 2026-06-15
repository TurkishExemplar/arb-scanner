"""Real-time prediction market arbitrage scanner.

Pipeline: fetch -> match -> classify -> alert -> execute

Monitors Polymarket and Kalshi for the same event priced differently across
platforms. Matching is a two-stage Jaccard -> TF-IDF cosine filter, confirmed
by a Random Forest classifier. Execution is opt-in and defaults to a dry run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

from auth import KalshiAuth
from classifier import MatchClassifier
from execution import alert, execute_arb, execution_banner, record_prediction
from labeling import auto_label_resolved, maybe_retrain
from matching import MatchPair, match_markets

# =========================
# CONFIG (all via env, sensible defaults)
# =========================
POLY_URL = os.getenv("POLY_URL", "https://gamma-api.polymarket.com")
KALSHI_HOST = os.getenv("KALSHI_HOST", "https://api.elections.kalshi.com")
KALSHI_PREFIX = "/trade-api/v2"

THRESHOLD = float(os.getenv("ARB_THRESHOLD", "0.01"))  # min raw arb edge to surface
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))  # seconds; respects API limits
KALSHI_RATE_SLEEP = 0.05  # 20 reads/sec ceiling

MATCH_PROB_THRESHOLD = float(os.getenv("MATCH_PROB_THRESHOLD", "0.5"))
AUTO_LABEL_EVERY = int(os.getenv("AUTO_LABEL_EVERY", "30"))  # scans between label sweeps
MIN_BOOK_SUM = float(os.getenv("MIN_BOOK_SUM", "0.98"))  # reject stale/one-sided books


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class Market:
    question: str
    yes_price: float
    no_price: float
    source: str
    market_id: str = ""       # polymarket id / kalshi ticker
    token_id_yes: str = ""    # polymarket CLOB token id (YES outcome)
    token_id_no: str = ""     # polymarket CLOB token id (NO outcome)


@dataclass
class Opportunity:
    poly: Market
    kalshi: Market
    edge: float               # fractional, e.g. 0.06 == 6%
    direction: str
    buy_platform: str
    buy_price: float
    sell_platform: str
    sell_price: float
    match: MatchPair
    max_size: float = 0.0


def eprint(*args: object) -> None:
    """Errors go to stderr; opportunities go to stdout."""
    print(*args, file=sys.stderr)


def is_multivariate(title: str) -> bool:
    """True for grouped / multi-outcome markets that aren't clean binaries.

    Detects the concatenated outcome format (e.g. "yes A, yes B, yes Tie, no C")
    by counting comma-separated segments that *begin* with a yes/no outcome
    label. Counting raw "yes "/"no " substrings or commas (thousands separators,
    dates) over-filtered legitimate binaries, so we key off structure instead.
    """
    segments = [s.strip().lower() for s in title.split(",")]
    enumerated = sum(1 for s in segments if s.startswith("yes ") or s.startswith("no "))
    return enumerated >= 2


def is_tradeable_binary(yes_price: float, no_price: float) -> bool:
    """A genuine two-sided binary has real offers on both sides whose prices form
    a coherent book (each in (0,1) and summing to ~1, not a stale/one-sided quote)."""
    if not (0.0 < yes_price < 1.0 and 0.0 < no_price < 1.0):
        return False
    return yes_price + no_price >= MIN_BOOK_SUM


def _poly_yes_no(market: dict) -> Optional[tuple[float, float, int, int]]:
    """Map a Polymarket market to (yes_price, no_price, yes_idx, no_idx).

    Polymarket's outcome order is not guaranteed to be ["Yes","No"], so we read
    the `outcomes` labels and align prices/token-ids by label. Returns None for
    markets that aren't a clean Yes/No binary (e.g. ["Up","Down"]).
    """
    try:
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        prices = [float(p) for p in prices]
    except (ValueError, TypeError):
        return None

    outcomes_raw = market.get("outcomes", "[]")
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        labels = [str(o).strip().lower() for o in outcomes]
    except (ValueError, TypeError):
        labels = []

    if "yes" in labels and "no" in labels:
        yi, ni = labels.index("yes"), labels.index("no")
    elif len(prices) == 2 and not labels:
        yi, ni = 0, 1  # no labels available; fall back to canonical order
    else:
        return None  # labelled but not a Yes/No binary

    try:
        return prices[yi], prices[ni], yi, ni
    except IndexError:
        return None


# =========================
# FETCH POLYMARKET (public gamma API)
# =========================
def fetch_polymarket(limit: int = 200) -> list[Market]:
    out: list[Market] = []
    try:
        r = requests.get(
            f"{POLY_URL}/markets",
            params={
                "limit": limit,
                "active": "true",
                "closed": "false",
                "archived": "false",
                "order": "volume24hr",
                "ascending": "false",
            },
            timeout=12,
        )
        r.raise_for_status()
        raw = r.json()
        raw = raw if isinstance(raw, list) else raw.get("markets", [])

        for m in raw:
            question = m.get("question", "")
            if not question or m.get("closed") or m.get("archived"):
                continue
            if is_multivariate(question):
                continue
            parsed = _poly_yes_no(m)
            if parsed is None:
                continue  # not a clean Yes/No binary
            yes_price, no_price, yes_idx, no_idx = parsed

            if not is_tradeable_binary(yes_price, no_price):
                continue
            if float(m.get("volume24hr", 0) or 0) == 0:
                continue

            token_yes, token_no = "", ""
            try:
                tokens_raw = m.get("clobTokenIds", "[]")
                tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
                token_yes, token_no = str(tokens[yes_idx]), str(tokens[no_idx])
            except (ValueError, TypeError, IndexError):
                pass

            out.append(
                Market(
                    question=question,
                    yes_price=yes_price,
                    no_price=no_price,
                    source="Polymarket",
                    market_id=str(m.get("id", "")),
                    token_id_yes=token_yes,
                    token_id_no=token_no,
                )
            )
    except Exception as e:
        eprint(f"[Polymarket ERROR] {e}")
    return out


def _kalshi_price(market: dict, ask_key: str, dollars_key: str) -> float:
    """Kalshi returns asks either in cents (yes_ask) or dollars (yes_ask_dollars)."""
    if market.get(dollars_key) is not None:
        return float(market.get(dollars_key) or 0)
    return float(market.get(ask_key, 0) or 0) / 100.0


# =========================
# FETCH KALSHI (signed if a key is set, public fallback otherwise)
# =========================
def fetch_kalshi(auth: KalshiAuth, limit: int = 200) -> list[Market]:
    out: list[Market] = []
    path = f"{KALSHI_PREFIX}/markets"
    try:
        headers = {"User-Agent": "arb-scanner/1.0", "Accept": "application/json"}
        if auth.enabled:
            headers.update(auth.headers("GET", path))
        else:
            print("[Kalshi] No signing key set — using public market data.")

        r = requests.get(
            f"{KALSHI_HOST}{path}",
            params={"limit": limit, "status": "open"},
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        for m in r.json().get("markets", []):
            try:
                question = m.get("title", "")
                if not question or is_multivariate(question):
                    continue
                if m.get("mve_collection_ticker"):  # multi-event combo leg, not a binary
                    continue
                yes_price = _kalshi_price(m, "yes_ask", "yes_ask_dollars")
                no_price = _kalshi_price(m, "no_ask", "no_ask_dollars")
                if not is_tradeable_binary(yes_price, no_price):
                    continue
                out.append(
                    Market(
                        question=question,
                        yes_price=yes_price,
                        no_price=no_price,
                        source="Kalshi",
                        market_id=str(m.get("ticker", "")),
                    )
                )
            except (ValueError, TypeError):
                continue
        time.sleep(KALSHI_RATE_SLEEP)
    except Exception as e:
        eprint(f"[Kalshi ERROR] {e}")
    return out


# =========================
# ARBITRAGE CHECK
# =========================
def find_arbs(pairs: list[MatchPair]) -> list[Opportunity]:
    """At most one Opportunity per pair: a genuine cross-venue arb exists in only
    one direction, so we keep the better of the two and never double-fire."""
    arbs: list[Opportunity] = []
    for pair in pairs:
        pm, km = pair.poly, pair.kalshi
        try:
            total_yes = pm.yes_price + km.no_price  # YES on Poly + NO on Kalshi
            total_no = pm.no_price + km.yes_price    # NO on Poly + YES on Kalshi
            yes_arb = Opportunity(
                poly=pm, kalshi=km, edge=1 - total_yes,
                direction="YES on Poly / NO on Kalshi",
                buy_platform="Polymarket", buy_price=pm.yes_price,
                sell_platform="Kalshi", sell_price=km.no_price,
                match=pair,
            )
            no_arb = Opportunity(
                poly=pm, kalshi=km, edge=1 - total_no,
                direction="NO on Poly / YES on Kalshi",
                buy_platform="Polymarket", buy_price=pm.no_price,
                sell_platform="Kalshi", sell_price=km.yes_price,
                match=pair,
            )
            best = yes_arb if yes_arb.edge >= no_arb.edge else no_arb
            if best.edge > THRESHOLD:
                arbs.append(best)
        except (TypeError, ValueError):
            continue
    return arbs


# =========================
# PIPELINE
# =========================
def run_once(auth: KalshiAuth, clf: MatchClassifier, scan: int = 1) -> None:
    poly = fetch_polymarket()
    kalshi = fetch_kalshi(auth)

    pairs = match_markets(poly, kalshi)              # two-stage Jaccard -> cosine
    confirmed: list[MatchPair] = []
    for pair in pairs:                              # classify
        pair.match_prob = clf.predict(pair)
        pair.model_trained = clf.is_trained
        if pair.match_prob >= MATCH_PROB_THRESHOLD:
            confirmed.append(pair)

    arbs = find_arbs(confirmed)

    print("=" * 70)
    print(f"  POLY <-> KALSHI ARB SCANNER  |  {time.strftime('%H:%M:%S')}")
    print("=" * 70)
    print(
        f"  Polymarket: {len(poly)}  |  Kalshi: {len(kalshi)}  |  "
        f"Matched: {len(confirmed)}/{len(pairs)} (post-classify)\n"
    )

    if confirmed:
        print("  TOP MATCHES:")
        for pair in sorted(confirmed, key=lambda p: p.cosine, reverse=True)[:5]:
            print(
                f"  [j={pair.jaccard:.2f} c={pair.cosine:.2f} p={pair.match_prob:.2f}] "
                f"POLY:   {pair.poly.question[:60]}"
            )
            print(f"         KALSHI: {pair.kalshi.question[:60]}\n")

    if arbs:
        print("  ARB OPPORTUNITIES:")
        for opp in sorted(arbs, key=lambda o: o.edge, reverse=True):
            alert(opp)            # stdout + opportunities.log
            execute_arb(opp, auth)  # gated; dry-run by default
            record_prediction(opp)
    else:
        print(f"  No arb opportunities above {THRESHOLD * 100:.1f}% threshold.\n")

    print(f"  Scan #{scan} done.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction market arbitrage scanner")
    parser.add_argument("--once", action="store_true", help="run a single scan and exit")
    args = parser.parse_args()

    auth = KalshiAuth()
    clf = MatchClassifier()

    print(f"Arb Scanner | threshold={THRESHOLD * 100:.1f}% | poll={POLL_INTERVAL}s")
    print(f"Kalshi auth: {'SIGNED' if auth.enabled else 'PUBLIC (no key set)'}")
    print(f"Classifier:  {'trained model' if clf.is_trained else 'cold-start (threshold gate)'}")
    for line in execution_banner():
        print(line)
    print()

    if args.once:
        run_once(auth, clf, scan=1)
        return

    scan = 1
    while True:
        try:
            run_once(auth, clf, scan=scan)
            if scan % AUTO_LABEL_EVERY == 0:
                auto_label_resolved(auth, fetch_poly_market, fetch_kalshi_market)
                if maybe_retrain(clf):
                    print("[classifier] retrained on accumulated auto-labels")
            scan += 1
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            eprint(f"[ERROR] {e}")
            time.sleep(5)


# =========================
# RESOLUTION LOOKUPS (used by auto-labeling)
# =========================
def fetch_poly_market(market_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{POLY_URL}/markets/{market_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        eprint(f"[Polymarket resolve ERROR] {e}")
        return None


def fetch_kalshi_market(ticker: str, auth: KalshiAuth) -> Optional[dict]:
    path = f"{KALSHI_PREFIX}/markets/{ticker}"
    try:
        headers = {"User-Agent": "arb-scanner/1.0", "Accept": "application/json"}
        if auth.enabled:
            headers.update(auth.headers("GET", path))
        r = requests.get(f"{KALSHI_HOST}{path}", headers=headers, timeout=10)
        r.raise_for_status()
        time.sleep(KALSHI_RATE_SLEEP)
        return r.json().get("market")
    except Exception as e:
        eprint(f"[Kalshi resolve ERROR] {e}")
        return None


if __name__ == "__main__":
    main()
