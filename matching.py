"""Two-stage market matcher with polarity / identity guards.

Stage 1: Jaccard token overlap as a fast pre-filter (threshold 0.15).
Stage 2: TF-IDF cosine similarity on the survivors (threshold 0.35).

Lexical overlap alone pairs a market with its opposite ("...reach 100k" vs
"...fall below 100k") or a sibling ("...in March" vs "...in June"), which would
feed find_arbs a phantom arbitrage. So a candidate is rejected before scoring if
the two questions disagree on direction polarity or cite disjoint numbers.
Finally, matches are assigned one-to-one (each market used at most once).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:  # avoid a circular import at runtime
    from scanner import Market

JACCARD_THRESHOLD = float(os.getenv("JACCARD_THRESHOLD", "0.15"))
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.35"))

# Pairs whose two questions lean to opposite sides are not the same outcome.
DIRECTION_ANTONYMS = [
    ("above", "below"), ("over", "under"), ("above", "under"), ("over", "below"),
    ("higher", "lower"), ("more", "less"), ("up", "down"), ("rise", "fall"),
    ("rises", "falls"), ("increase", "decrease"), ("win", "lose"),
    ("wins", "loses"), ("gain", "lose"), ("positive", "negative"),
    ("reach", "fall"), ("exceed", "fall"), ("exceed", "drop"), ("rise", "drop"),
    ("beat", "miss"), ("beats", "misses"),
]


@dataclass
class MatchPair:
    poly: "Market"
    kalshi: "Market"
    jaccard: float
    cosine: float
    match_prob: float = 0.0
    model_trained: bool = False


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", normalize(text)))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _lean(tokens: set[str], a: str, b: str) -> int:
    ha, hb = a in tokens, b in tokens
    return 1 if (ha and not hb) else (-1 if (hb and not ha) else 0)


def polarity_conflict(t1: set[str], t2: set[str]) -> bool:
    """True if the two token sets lean to opposite sides of any antonym pair."""
    return any(_lean(t1, a, b) * _lean(t2, a, b) < 0 for a, b in DIRECTION_ANTONYMS)


def _thresholds(nums: set[str]) -> set[str]:
    """Drop year-like numbers so a shared deadline doesn't mask differing thresholds."""
    return {n for n in nums if not (len(n) == 4 and 1900 <= int(n) <= 2999)}


def numeric_conflict(n1: set[str], n2: set[str]) -> bool:
    """True if the two questions cite incompatible numbers.

    Two cases: (a) both cite numbers but share none at all, or (b) both cite a
    non-year threshold (price/count/etc.) and those thresholds are disjoint even
    if they share a year — e.g. "above 100000 in 2026" vs "above 110000 in 2026".
    """
    if n1 and n2 and n1.isdisjoint(n2):
        return True
    t1, t2 = _thresholds(n1), _thresholds(n2)
    return bool(t1) and bool(t2) and t1.isdisjoint(t2)


def match_markets(poly: list["Market"], kalshi: list["Market"]) -> list[MatchPair]:
    if not poly or not kalshi:
        return []

    # Normalize/tokenize each market once, then reuse everywhere below.
    p_norm = [normalize(m.question) for m in poly]
    k_norm = [normalize(m.question) for m in kalshi]
    p_tok = [set(s.split()) for s in p_norm]
    k_tok = [set(s.split()) for s in k_norm]
    p_num = [set(re.findall(r"\d+", s)) for s in p_norm]
    k_num = [set(re.findall(r"\d+", s)) for s in k_norm]

    # --- Stage 1: Jaccard pre-filter + polarity/numeric guards ---
    candidates: list[tuple[int, int, float]] = []
    for i, ti in enumerate(p_tok):
        if not ti:
            continue
        for j, tj in enumerate(k_tok):
            if not tj:
                continue
            jac = len(ti & tj) / len(ti | tj)
            if jac <= JACCARD_THRESHOLD:
                continue
            if polarity_conflict(ti, tj) or numeric_conflict(p_num[i], k_num[j]):
                continue
            candidates.append((i, j, jac))
    if not candidates:
        return []

    # --- Stage 2: TF-IDF cosine on the survivors ---
    try:
        tfidf = TfidfVectorizer().fit_transform(p_norm + k_norm)
    except ValueError:
        return []  # empty vocabulary
    offset = len(poly)

    scored: list[tuple[float, int, int, float]] = []
    for i, j, jac in candidates:
        cos = float(cosine_similarity(tfidf[i], tfidf[offset + j])[0][0])
        if cos > COSINE_THRESHOLD:
            scored.append((cos, i, j, jac))

    # --- One-to-one assignment: greedily take the best cosine, each side once ---
    scored.sort(key=lambda x: x[0], reverse=True)
    used_p: set[int] = set()
    used_k: set[int] = set()
    out: list[MatchPair] = []
    for cos, i, j, jac in scored:
        if i in used_p or j in used_k:
            continue
        used_p.add(i)
        used_k.add(j)
        out.append(MatchPair(poly=poly[i], kalshi=kalshi[j], jaccard=jac, cosine=cos))
    return out
