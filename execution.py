"""Alerting, position sizing, P&L tracking, and (opt-in) trade execution.

Safety model — real orders require BOTH of these, and default to neither:
    * ENABLE_EXECUTION == "true"   (env var, explicit opt-in)
    * DRY_RUN == "false"           (env var; DRY_RUN defaults to True)

With the defaults, ``execute_arb`` only ever logs what it *would* do. Nothing
touches a live order book until the operator deliberately flips both switches.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import requests

if TYPE_CHECKING:
    from scanner import Opportunity

from classifier import featurize

# --- files ---
OPPORTUNITIES_LOG = os.getenv("OPPORTUNITIES_LOG", "opportunities.log")
PNL_LOG = os.getenv("PNL_LOG", "pnl.log")
PREDICTIONS_LOG = os.getenv("PREDICTIONS_LOG", "predictions.log")

# --- Kalshi order endpoint ---
KALSHI_HOST = os.getenv("KALSHI_HOST", "https://api.elections.kalshi.com")
KALSHI_PREFIX = "/trade-api/v2"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- execution config ---
DRY_RUN = _env_bool("DRY_RUN", True)                       # log instead of trading
ENABLE_EXECUTION = os.getenv("ENABLE_EXECUTION", "").strip().lower() == "true"
MIN_EXECUTION_EDGE = float(os.getenv("MIN_EXECUTION_EDGE", "5.0"))  # percent, after fees
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "500"))    # dollars per arb
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.02"))     # 2% of bankroll
BANKROLL = float(os.getenv("BANKROLL", "0"))
KALSHI_FEE_RATE = float(os.getenv("KALSHI_FEE_RATE", "0.07"))       # Kalshi fee constant
POLY_FEE_RATE = float(os.getenv("POLY_FEE_RATE", "0.0"))            # Polymarket fee (notional)
SLIPPAGE_BUFFER = float(os.getenv("SLIPPAGE_BUFFER", "0.01"))       # fraction of notional reserved
LIVE_MATCH_PROB = float(os.getenv("LIVE_MATCH_PROB", "0.9"))        # min match prob for LIVE orders


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


# =========================
# ALERT
# =========================
def alert(opp: "Opportunity") -> None:
    """Print the opportunity to stdout and append it to opportunities.log."""
    max_size = min(BANKROLL * MAX_POSITION_PCT, MAX_POSITION_SIZE) if BANKROLL > 0 else 0.0
    opp.max_size = max_size

    print(f"  [{opp.direction}]  EDGE: {opp.edge * 100:.2f}%")
    print(
        f"  POLY:   {opp.poly.question[:70]} "
        f"@ yes={opp.poly.yes_price:.3f} no={opp.poly.no_price:.3f}"
    )
    print(
        f"  KALSHI: {opp.kalshi.question[:70]} "
        f"@ yes={opp.kalshi.yes_price:.3f} no={opp.kalshi.no_price:.3f}\n"
    )

    record = {
        "ts": _now_iso(),
        "edge_pct": round(opp.edge * 100, 2),
        "buy": opp.buy_platform,
        "sell": opp.sell_platform,
        "q_buy": opp.poly.question,
        "buy_price": round(opp.buy_price, 4),
        "sell_price": round(opp.sell_price, 4),
        "max_size": round(max_size, 2),
    }
    _append_json(OPPORTUNITIES_LOG, record)


# =========================
# POSITION SIZING
# =========================
def size_position(opp: "Opportunity", bankroll: float) -> tuple[float, float]:
    """Return (poly_size, kalshi_size) in dollars for one arb.

    Caps total deployed at min(2% of bankroll, MAX_POSITION_SIZE) and splits it
    across the two legs in proportion to each leg's price so the contract counts
    match (a hedged pair).
    """
    if bankroll <= 0:
        return (0.0, 0.0)
    max_total = min(bankroll * MAX_POSITION_PCT, MAX_POSITION_SIZE)
    cost_per_pair = opp.buy_price + opp.sell_price
    if cost_per_pair <= 0:
        return (0.0, 0.0)
    contracts = math.floor(max_total / cost_per_pair)
    if contracts <= 0:
        return (0.0, 0.0)
    return (round(opp.buy_price * contracts, 2), round(opp.sell_price * contracts, 2))


def _estimate_fees(price: float, contracts: int) -> float:
    """Kalshi trading-fee estimate: ceil(rate * C * P * (1-P)) dollars."""
    if contracts <= 0:
        return 0.0
    return math.ceil(KALSHI_FEE_RATE * contracts * price * (1 - price) * 100) / 100


# =========================
# EXECUTION (gated)
# =========================
def execute_arb(opp: "Opportunity", auth: Optional[object] = None) -> dict:
    """Place a hedged YES/NO pair. Dry-run + disabled by default (see module docstring)."""
    poly_size, kalshi_size = size_position(opp, BANKROLL)
    if poly_size <= 0 and kalshi_size <= 0:
        print("  [exec] skipped — bankroll too small to size a position (set BANKROLL).")
        return {"status": "skipped", "reason": "insufficient_size"}

    cost_per_pair = opp.buy_price + opp.sell_price
    contracts = math.floor(min(BANKROLL * MAX_POSITION_PCT, MAX_POSITION_SIZE) / cost_per_pair)

    # Conservative cost model: fees on BOTH legs + a slippage reserve, so the
    # gate compares against worst-case realized cost rather than the quoted ask.
    notional = poly_size + kalshi_size
    fees = _estimate_fees(opp.sell_price, contracts) + POLY_FEE_RATE * poly_size
    slippage = SLIPPAGE_BUFFER * notional
    entry_cost = notional + fees + slippage
    payout = float(contracts)                          # hedged pair pays $1/contract
    net_edge_pct = (payout - entry_cost) / entry_cost * 100 if entry_cost > 0 else 0.0

    if net_edge_pct < MIN_EXECUTION_EDGE:
        print(
            f"  [exec] skipped — net edge {net_edge_pct:.2f}% "
            f"< MIN_EXECUTION_EDGE {MIN_EXECUTION_EDGE:.2f}% (after fees + slippage)."
        )
        return {"status": "skipped", "reason": "below_min_edge", "net_edge_pct": net_edge_pct}

    if not ENABLE_EXECUTION:
        print("  [exec] disabled — set ENABLE_EXECUTION=true to arm execution.")
        return {"status": "disabled"}

    profit = payout - entry_cost

    if DRY_RUN:
        print(
            f"  [exec] DRY RUN — would buy {contracts} pairs "
            f"(poly ${poly_size:.2f} / kalshi ${kalshi_size:.2f}, "
            f"net edge {net_edge_pct:.2f}%)."
        )
        _record_pnl(opp, contracts, entry_cost, payout, profit, dry_run=True)
        return {"status": "dry_run", "contracts": contracts, "net_edge_pct": net_edge_pct}

    # --- LIVE path: refuse to trade on an unconfirmed match ---
    if not getattr(opp.match, "model_trained", False) or opp.match.match_prob < LIVE_MATCH_PROB:
        print(
            "  [exec] refused — LIVE orders require a trained classifier and "
            f"match prob >= {LIVE_MATCH_PROB:.2f} (got {opp.match.match_prob:.2f}, "
            f"trained={getattr(opp.match, 'model_trained', False)}). Staying flat."
        )
        return {"status": "refused", "reason": "unconfirmed_match"}

    # Legs are NOT atomic: place Polymarket first; if Kalshi then fails, the Poly
    # leg may be live and UNHEDGED. We loudly flag it for manual intervention
    # rather than silently assuming a perfect hedge.
    try:
        _place_polymarket_order(opp, contracts)
    except Exception as e:
        eprint(f"  [exec ERROR] first leg (Polymarket) failed before any fill: {e}")
        return {"status": "error", "leg": "polymarket", "error": str(e)}

    try:
        _place_kalshi_order(opp, contracts, auth)
    except Exception as e:
        eprint(
            "  [exec CRITICAL] SECOND LEG (Kalshi) FAILED after the Polymarket leg "
            f"was placed — you may hold an UNHEDGED {contracts}-contract position. "
            f"Manual action required. error={e}"
        )
        _record_pnl(opp, contracts, entry_cost, payout, profit, dry_run=False, unhedged=True)
        return {"status": "error", "leg": "kalshi", "unhedged": True, "error": str(e)}

    _record_pnl(opp, contracts, entry_cost, payout, profit, dry_run=False)
    print(f"  [exec] LIVE — placed {contracts} pairs.")
    return {"status": "executed", "contracts": contracts}


def execution_banner() -> list[str]:
    """Startup status lines describing how execution is armed (printed by scanner)."""
    if not ENABLE_EXECUTION:
        return ["Execution:   DISABLED (dry-run only; set ENABLE_EXECUTION=true to arm)"]
    if DRY_RUN:
        return ["Execution:   ARMED but DRY_RUN=true (logging only, no real orders)"]
    lines = ["Execution:   *** LIVE — REAL ORDERS ENABLED ***"]
    if BANKROLL <= 0:
        lines.append("  WARNING: BANKROLL<=0 — sizing returns 0, nothing will actually trade.")
    return lines


def _place_kalshi_order(opp: "Opportunity", contracts: int, auth: Optional[object]) -> dict:
    """Signed POST to Kalshi's portfolio/orders endpoint."""
    if auth is None:
        from auth import KalshiAuth

        auth = KalshiAuth()
    if not getattr(auth, "enabled", False):
        raise RuntimeError("Kalshi execution requires KALSHI_KEY_ID + KALSHI_PRIVATE_KEY_PATH")

    side = "no" if "NO on Kalshi" in opp.direction else "yes"
    price_cents = int(round(opp.sell_price * 100))
    path = f"{KALSHI_PREFIX}/portfolio/orders"
    body = {
        "ticker": opp.kalshi.market_id,
        "action": "buy",
        "side": side,
        "count": contracts,
        "type": "limit",
        f"{side}_price": price_cents,
        "client_order_id": f"arb-{opp.kalshi.market_id}-{int(datetime.now(timezone.utc).timestamp())}",
    }
    headers = {"Content-Type": "application/json", **auth.headers("POST", path)}
    r = requests.post(f"{KALSHI_HOST}{path}", headers=headers, json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def _place_polymarket_order(opp: "Opportunity", contracts: int) -> dict:
    """Place the Polymarket leg via the CLOB API.

    Requires POLYMARKET_PRIVATE_KEY and the optional ``py-clob-client`` package
    (signing CLOB orders needs the wallet key). Kept import-guarded so the
    scanner runs without it; live Polymarket execution is opt-in.
    """
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not private_key:
        raise RuntimeError("Polymarket execution requires POLYMARKET_PRIVATE_KEY")
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY
    except ImportError as e:
        raise RuntimeError(
            "Polymarket live execution needs 'py-clob-client' (pip install py-clob-client)"
        ) from e

    token_id = opp.poly.token_id_yes if "YES on Poly" in opp.direction else opp.poly.token_id_no
    if not token_id:
        raise RuntimeError("missing Polymarket CLOB token id for this market")

    client = ClobClient(
        os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com"),
        key=private_key,
        chain_id=137,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    order = client.create_order(
        OrderArgs(token_id=token_id, price=opp.buy_price, size=float(contracts), side=BUY)
    )
    return client.post_order(order)


# =========================
# P&L
# =========================
def _record_pnl(
    opp: "Opportunity",
    contracts: int,
    entry_cost: float,
    payout: float,
    profit: float,
    dry_run: bool,
    unhedged: bool = False,
) -> None:
    record = {
        "timestamp": _now_iso(),
        "market_pair": f"{opp.poly.question[:60]} <-> {opp.kalshi.question[:60]}",
        "poly_side": "YES" if "YES on Poly" in opp.direction else "NO",
        "kalshi_side": "NO" if "NO on Kalshi" in opp.direction else "YES",
        "size": contracts,
        "entry_cost": round(entry_cost, 2),
        "payout": round(payout, 2),
        "profit": round(profit, 2),
        "dry_run": dry_run,
        "unhedged": unhedged,  # True == only one leg filled; hedge is broken
    }
    _append_json(PNL_LOG, record)


# =========================
# PREDICTION LOG (for auto-labeling)
# =========================
def record_prediction(opp: "Opportunity") -> None:
    """Persist the matched pair so it can be auto-labeled once both markets resolve."""
    feats = featurize(opp.match)
    record = {
        "ts": _now_iso(),
        "poly_id": opp.poly.market_id,
        "kalshi_ticker": opp.kalshi.market_id,
        "direction": opp.direction,
        "features": dict(zip(["jaccard", "cosine", "len_ratio", "num_overlap"], feats)),
    }
    _append_json(PREDICTIONS_LOG, record)


def _append_json(path: str, record: dict) -> None:
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        eprint(f"[log ERROR] could not write {path}: {e}")
