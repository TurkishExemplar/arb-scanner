"""Evaluate matching + the classifier on real, live market data.

Run:  python evaluate_matching.py

Broadly fetches both platforms (the live scanner only pulls a narrow top-volume
slice per cycle, which often doesn't overlap), runs the real match pipeline, and
prints what the classifier accepts vs. rejects — with a cold-start-vs-trained
comparison so you can see the trained model filtering phantom matches.

Nothing here trades; it's read-only.
"""

import requests

import auth
import classifier
import matching
import scanner

POLY_PAGES = 6
KALSHI_LIMIT = 980


def fetch_polymarket_broad() -> list:
    """Paginate Polymarket so topic coverage overlaps Kalshi's diverse feed."""
    out = []
    for page in range(POLY_PAGES):
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 500, "offset": page * 500, "active": "true",
                    "closed": "false", "archived": "false"},
            timeout=15,
        )
        raw = r.json()
        raw = raw if isinstance(raw, list) else raw.get("markets", [])
        if not raw:
            break
        for m in raw:
            q = m.get("question", "")
            if not q or m.get("closed") or m.get("archived") or scanner.is_multivariate(q):
                continue
            parsed = scanner._poly_yes_no(m)
            if parsed is None:
                continue
            yes_price, no_price, _, _ = parsed
            if not scanner.is_tradeable_binary(yes_price, no_price):
                continue
            out.append(scanner.Market(q, yes_price, no_price, "Polymarket", str(m.get("id", ""))))
    return out


def main() -> None:
    poly = fetch_polymarket_broad()
    kalshi = scanner.fetch_kalshi(auth.KalshiAuth(), limit=KALSHI_LIMIT)
    pairs = matching.match_markets(poly, kalshi)
    clf = classifier.MatchClassifier()

    print(f"Fetched: Polymarket={len(poly)}  Kalshi={len(kalshi)}")
    print(f"Classifier: {'TRAINED' if clf.is_trained else 'COLD-START (run seed_labels.py to train)'}")
    print(f"Candidate pairs (passed two-stage + guards): {len(pairs)}\n")

    accepted, rejected = [], []
    for p in pairs:
        p.match_prob = clf.predict(p)
        (accepted if p.match_prob >= scanner.MATCH_PROB_THRESHOLD else rejected).append(p)

    cold_arbs = len(scanner.find_arbs(pairs))
    live_arbs = len(scanner.find_arbs(accepted))
    print(f"Cold-start (accept all):  {len(pairs)} pairs -> {cold_arbs} arbs")
    print(f"Classifier-filtered:      {len(accepted)} pairs -> {live_arbs} arbs "
          f"({len(rejected)} rejected)\n")

    if rejected:
        print("REJECTED as look-alikes (lowest confidence first):")
        for p in sorted(rejected, key=lambda x: x.match_prob)[:8]:
            print(f"  p={p.match_prob:.2f} | {p.poly.question[:44]!r}  ~  {p.kalshi.question[:40]!r}")
    if accepted:
        print("\nKEPT as real matches (highest confidence first):")
        for p in sorted(accepted, key=lambda x: -x.match_prob)[:8]:
            print(f"  p={p.match_prob:.2f} | {p.poly.question[:44]!r}  ~  {p.kalshi.question[:40]!r}")


if __name__ == "__main__":
    main()
