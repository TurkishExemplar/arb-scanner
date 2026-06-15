"""Bootstrap the match classifier with a starter labels.csv, then train it.

Run once from the project root:  python seed_labels.py

The cold-start classifier accepts every pair that clears the two-stage gate,
which is noisy on real data. This writes a small hand-labeled starter set —
TRUE = the same event phrased two ways, FALSE = look-alikes that slip past the
guards (same template, different subject/office/year) — and fits the Random
Forest so matching is filtered from the first run.

This is a *bootstrap*, not a validated dataset. Real labels accumulate
automatically as markets resolve (see labeling.py), and the model retrains
weekly from those; they should eventually supersede this seed.
"""

import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import classifier
import matching
import scanner

# --- TRUE: same event, two phrasings (label 1) ---
TRUE = [
    ("Will Bitcoin close above $100,000 by December 31, 2026?", "Will Bitcoin be above 100000 by end of 2026?"),
    ("Will Bitcoin close above 100000 by Dec 31 2026?", "Will BTC exceed $100,000 during 2026?"),
    ("Will Ethereum reach $5,000 in 2026?", "Will ETH be above 5000 dollars by end of 2026?"),
    ("Will the Fed cut interest rates in June 2026?", "Will the Federal Reserve lower rates at the June 2026 meeting?"),
    ("Will a Democrat win the 2028 presidential election?", "Will the Democratic party win the Presidency in 2028?"),
    ("Will Gavin Newsom win the 2028 Democratic nomination?", "Will Gavin Newsom be the 2028 Democratic nominee?"),
    ("Will the Lakers win the 2026 NBA Championship?", "Will the Los Angeles Lakers win the 2026 NBA title?"),
    ("Will SpaceX land humans on Mars before 2030?", "Will SpaceX put people on Mars before 2030?"),
    ("Will US CPI inflation be above 3% in 2026?", "Will US inflation exceed 3 percent during 2026?"),
    ("Will Trump win the 2024 presidential election?", "Will Donald Trump win the 2024 election?"),
    ("Will US unemployment rise above 5% in 2026?", "Will the US unemployment rate exceed 5% in 2026?"),
    ("Will Apple stock close above $300 in 2026?", "Will AAPL trade above 300 dollars by end of 2026?"),
    ("Will there be a US government shutdown in 2026?", "Will the US government shut down during 2026?"),
    ("Will OpenAI release GPT-6 in 2026?", "Will OpenAI launch GPT-6 during 2026?"),
    ("Will the S&P 500 close above 7000 in 2026?", "Will the S&P 500 index exceed 7000 by end of 2026?"),
    ("Will Kamala Harris win the 2028 Democratic nomination?", "Will Kamala Harris be the 2028 Democratic nominee?"),
    ("Will Manchester City win the 2026 Premier League?", "Will Man City win the 2026 EPL title?"),
    ("Will the US enter a recession in 2026?", "Will there be a US recession during 2026?"),
    ("Will Tesla deliver 2 million cars in 2026?", "Will Tesla sell 2000000 vehicles during 2026?"),
    ("Will Joe Biden run for president in 2028?", "Will Joe Biden be a 2028 presidential candidate?"),
    ("Will gold close above $3000 per ounce in 2026?", "Will gold exceed 3000 dollars an ounce by end of 2026?"),
    ("Will the Chiefs win the 2027 Super Bowl?", "Will the Kansas City Chiefs win Super Bowl 2027?"),
    ("Will NASA return astronauts to the Moon by 2027?", "Will NASA land humans on the Moon before 2027?"),
    ("Will the UK hold a general election in 2026?", "Will there be a UK general election during 2026?"),
    ("Will Nvidia stock close above $200 in 2026?", "Will NVDA be above 200 dollars by end of 2026?"),
    ("Will Eurozone inflation exceed 2% in 2026?", "Will inflation in the Eurozone be above 2 percent in 2026?"),
    ("Will Argentina win the 2026 World Cup?", "Will Argentina be 2026 FIFA World Cup champions?"),
    ("Will the Fed hold rates steady in March 2026?", "Will the Federal Reserve keep rates unchanged at the March 2026 meeting?"),
]

# --- FALSE: similar wording, DIFFERENT event — look-alikes that pass the guards (label 0) ---
FALSE = [
    ("Will LeBron James win the 2028 Democratic primary?", "Who will win the 2028 presidential election? LeBron James"),
    ("Will Newsom win the 2028 Democratic primary?", "Will Newsom win the 2028 presidential election?"),
    ("Will the Democrats win the Senate in 2026?", "Will the Democrats win the House in 2026?"),
    ("Will the Lakers win the 2026 NBA Championship?", "Will the Celtics win the 2026 NBA Championship?"),
    ("Will it rain in New York on July 4 2026?", "Will it rain in Los Angeles on July 4 2026?"),
    ("Will AOC win the 2028 Democratic primary?", "Will AOC win the 2028 Senate race?"),
    ("Will Mexico win the 2026 World Cup?", "Will Brazil win the 2026 World Cup?"),
    ("Will Bitcoin close above 100000 in 2026?", "Will Ethereum close above 100000 in 2026?"),
    ("Will Trump win the 2028 Republican primary?", "Will Trump win the 2028 presidential election?"),
    ("Will the Chiefs win the 2027 Super Bowl?", "Will the Eagles win the 2027 Super Bowl?"),
    ("Will Apple announce a new iPhone in 2026?", "Will Apple announce a new iPad in 2026?"),
    ("Will the Fed cut rates in June 2026?", "Will the Fed cut rates in September 2026?"),
    ("Will Newsom win the California governor race in 2026?", "Will Newsom win the 2028 presidential election?"),
    ("Will gold close above 3000 in 2026?", "Will silver close above 3000 in 2026?"),
    ("Will Manchester City win the 2026 Premier League?", "Will Manchester United win the 2026 Premier League?"),
    ("Will the US enter a recession in 2026?", "Will the US enter a recession in 2028?"),
    ("Will Harris win the 2028 Democratic primary?", "Who will win the 2028 presidential election? Kamala Harris"),
    ("Will Tesla deliver 2 million cars in 2026?", "Will Tesla open 2 million charging stations in 2026?"),
    ("Will SpaceX land on Mars before 2030?", "Will SpaceX land on the Moon before 2030?"),
    ("Will Nvidia close above 200 in 2026?", "Will AMD close above 200 in 2026?"),
    ("Will the Yankees win the 2026 World Series?", "Will the Dodgers win the 2026 World Series?"),
    ("Will Biden win the 2024 election?", "Will Trump win the 2024 election?"),
    ("Will CPI inflation exceed 3% in 2026?", "Will GDP growth exceed 3% in 2026?"),
    ("Will the UK hold a general election in 2026?", "Will France hold a general election in 2026?"),
    ("Will Pritzker win the 2028 Democratic primary?", "Will Pritzker win the Illinois governor race in 2026?"),
    ("Will it snow in Chicago in December 2026?", "Will it snow in Denver in December 2026?"),
    ("Will Argentina win the 2026 World Cup?", "Will Argentina win the 2026 Copa America?"),
    ("Will the S&P 500 close above 7000 in 2026?", "Will the Nasdaq close above 7000 in 2026?"),
]


def main() -> None:
    # One TF-IDF space over the whole seed corpus, mirroring how a scan scores cosine.
    corpus = [matching.normalize(s) for pair in TRUE + FALSE for s in pair]
    vec = TfidfVectorizer().fit(corpus)

    def features(a: str, b: str) -> list[float]:
        pair = matching.MatchPair(
            scanner.Market(a, 0.5, 0.5, "Polymarket"),
            scanner.Market(b, 0.5, 0.5, "Kalshi"),
            matching.jaccard(a, b),
            float(cosine_similarity(
                vec.transform([matching.normalize(a)]),
                vec.transform([matching.normalize(b)]),
            )[0][0]),
        )
        return classifier.featurize(pair)

    rows = [(features(a, b), 1) for a, b in TRUE] + [(features(a, b), 0) for a, b in FALSE]
    with open(classifier.LABELS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(classifier.FEATURE_NAMES + ["label"])
        for fv, y in rows:
            w.writerow(fv + [y])
    print(f"wrote {classifier.LABELS_CSV}: {len(TRUE)} true + {len(FALSE)} false = {len(rows)} rows")

    clf = classifier.MatchClassifier()
    if clf.train(classifier.LABELS_CSV):
        importances = dict(zip(classifier.FEATURE_NAMES, clf.model.feature_importances_.round(3)))
        print(f"trained {classifier.MODEL_PATH}; feature importances: {importances}")
    else:
        print("training skipped (need >= MIN_TRAIN_ROWS rows and both classes)")


if __name__ == "__main__":
    main()
