"""Arb scanner dashboard.

Run:  python dashboard.py
Open: http://127.0.0.1:5001

Reads opportunities.log and pnl.log (written by scanner.py in the same folder)
and serves them to dashboard.html. Auto-refreshes client-side every 30s.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from flask import Flask, jsonify

app = Flask(__name__)
OPPS_FILE = os.getenv("OPPORTUNITIES_LOG", "opportunities.log")
PNL_FILE = os.getenv("PNL_LOG", "pnl.log")
PORT = int(os.getenv("DASHBOARD_PORT", "5001"))
HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")  # localhost only by default


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


@app.route("/api/opportunities")
def get_opps():
    return jsonify(_load_jsonl(OPPS_FILE))


@app.route("/api/stats")
def get_stats():
    opps = _load_jsonl(OPPS_FILE)
    if not opps:
        return jsonify({"total": 0, "avg_edge": 0, "best_edge": 0,
                        "total_potential": 0, "by_platform": {}, "by_hour": {}})
    by_platform: dict[str, int] = defaultdict(int)
    by_hour: dict[str, int] = defaultdict(int)
    for o in opps:
        by_platform[f"{o.get('buy', '?')} -> {o.get('sell', '?')}"] += 1
        try:
            by_hour[o["ts"][:13]] += 1
        except (KeyError, TypeError):
            pass
    return jsonify({
        "total": len(opps),
        "avg_edge": round(sum(o.get("edge_pct", 0) for o in opps) / len(opps), 2),
        "best_edge": round(max(o.get("edge_pct", 0) for o in opps), 2),
        "total_potential": round(sum(o.get("max_size", 0) for o in opps), 2),
        "by_platform": dict(by_platform),
        "by_hour": dict(sorted(by_hour.items())),
    })


@app.route("/api/pnl")
def get_pnl():
    rows = _load_jsonl(PNL_FILE)
    if not rows:
        return jsonify({"trades": 0, "total_deployed": 0, "total_profit": 0,
                        "win_rate": 0, "dry_run": 0, "live": 0})
    wins = sum(1 for r in rows if r.get("profit", 0) > 0)
    return jsonify({
        "trades": len(rows),
        "total_deployed": round(sum(r.get("entry_cost", 0) for r in rows), 2),
        "total_profit": round(sum(r.get("profit", 0) for r in rows), 2),
        "win_rate": round(wins / len(rows) * 100, 1),
        "dry_run": sum(1 for r in rows if r.get("dry_run")),
        "live": sum(1 for r in rows if not r.get("dry_run")),
    })


@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "dashboard.html")) as f:
        return f.read()


if __name__ == "__main__":
    print("\n  Arb Dashboard starting...")
    print(f"  Open -> http://{HOST}:{PORT}")
    print("  Reads: opportunities.log + pnl.log (same folder)")
    print("  Auto-refreshes every 30s\n")
    app.run(host=HOST, port=PORT, debug=False)
