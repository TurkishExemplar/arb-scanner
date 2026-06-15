"""Automatic labeling and weekly retraining.

Replaces manual labeling. Every matched pair the scanner acts on is recorded to
``predictions.log``. Once *both* markets in a pair resolve, we derive a label
(do the two markets resolve consistently -> same event) and append it to
``labels.csv``. The Random Forest is retrained weekly from those auto-labels.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Callable, Optional

PREDICTIONS_LOG = os.getenv("PREDICTIONS_LOG", "predictions.log")
LABELS_CSV = os.getenv("LABELS_CSV", "labels.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
RETRAIN_INTERVAL = float(os.getenv("RETRAIN_INTERVAL_SEC", str(7 * 24 * 3600)))  # weekly
MAX_LOOKUPS_PER_SWEEP = int(os.getenv("MAX_LABEL_LOOKUPS", "25"))  # bound the network work
FEATURE_NAMES = ["jaccard", "cosine", "len_ratio", "num_overlap"]


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _poly_resolution(market: Optional[dict]) -> Optional[bool]:
    """Return True if YES won, False if NO won, None if unresolved.

    Maps the resolved price by the `outcomes` label (order isn't guaranteed) so a
    reversed ["No","Yes"] market doesn't invert the auto-label.
    """
    if not market or not market.get("closed"):
        return None
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
    yi = labels.index("yes") if "yes" in labels else 0

    try:
        yes_price = prices[yi]
    except IndexError:
        return None
    if yes_price in (0.0, 1.0):
        return yes_price == 1.0
    return None


def _kalshi_resolution(market: Optional[dict]) -> Optional[bool]:
    if not market:
        return None
    result = (market.get("result") or "").lower()
    if result == "yes":
        return True
    if result == "no":
        return False
    return None


def auto_label_resolved(
    auth: object,
    fetch_poly_market: Callable[[str], Optional[dict]],
    fetch_kalshi_market: Callable[[str, object], Optional[dict]],
) -> int:
    """Resolve pending predictions; write labels for any that have settled.

    Returns the number of newly written labels.
    """
    if not os.path.exists(PREDICTIONS_LOG):
        return 0

    pending: list[dict] = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pending.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Bound the network work per sweep so a backlog can't stall the scan loop:
    # only the first MAX_LOOKUPS_PER_SWEEP pending predictions are checked now;
    # the rest are carried over untouched to the next sweep.
    to_check = pending[:MAX_LOOKUPS_PER_SWEEP]
    carried = pending[MAX_LOOKUPS_PER_SWEEP:]

    still_pending: list[dict] = []
    new_labels = 0
    for pred in to_check:
        poly_won = _poly_resolution(fetch_poly_market(pred.get("poly_id", "")))
        kalshi_won = _kalshi_resolution(
            fetch_kalshi_market(pred.get("kalshi_ticker", ""), auth)
        )
        if poly_won is None or kalshi_won is None:
            still_pending.append(pred)  # not settled yet — keep waiting
            continue
        label = 1 if poly_won == kalshi_won else 0
        _append_label(pred.get("features", {}), label)
        new_labels += 1

    _rewrite(PREDICTIONS_LOG, still_pending + carried)
    if new_labels:
        print(f"[auto-label] wrote {new_labels} new label(s) to {LABELS_CSV}")
    return new_labels


def _append_label(features: dict, label: int) -> None:
    write_header = not os.path.exists(LABELS_CSV)
    try:
        with open(LABELS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(FEATURE_NAMES + ["label"])
            writer.writerow([features.get(name, 0.0) for name in FEATURE_NAMES] + [label])
    except OSError as e:
        _eprint(f"[auto-label ERROR] {e}")


def _rewrite(path: str, records: list[dict]) -> None:
    """Atomically replace `path` so an interrupt mid-write can't lose pending data."""
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        _eprint(f"[auto-label ERROR] {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def maybe_retrain(clf: object) -> bool:
    """Retrain the classifier if a week has passed since the last fit."""
    last_train = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0
    if time.time() - last_train < RETRAIN_INTERVAL:
        return False
    return bool(clf.train(LABELS_CSV))
