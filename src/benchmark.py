"""Compare simple sentiment methods on the labelled example messages.

Writes docs/results.json, which the web page reads.
Run from the repo root:  python src/benchmark.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["negative", "neutral", "positive"]

d = pd.read_csv(ROOT / "data" / "sample_tweets.csv")
y = d.sentiment

vader = SentimentIntensityAnalyzer()
d["vader_score"] = d.text.map(lambda t: vader.polarity_scores(t)["compound"])
d["vader"] = d.vader_score.map(lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral"))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
d["tfidf"] = cross_val_predict(tfidf, d.text, y, cv=cv)
d["majority"] = y.value_counts().idxmax()


def score(col):
    return {
        "accuracy": round(accuracy_score(y, d[col]), 3),
        "macro_f1": round(f1_score(y, d[col], average="macro", labels=LABELS, zero_division=0), 3),
    }


out = {
    "n": int(len(d)),
    "label_counts": {k: int(v) for k, v in y.value_counts().items()},
    "methods": {
        "majority": {"name": "Always guess the most common label", **score("majority")},
        "tfidf": {"name": "TF-IDF + logistic regression (5-fold cross-validation)", **score("tfidf")},
        "vader": {"name": "VADER lexicon (no training)", **score("vader")},
    },
    "confusion_vader": {a: {b: int(((y == a) & (d.vader == b)).sum()) for b in LABELS} for a in LABELS},
    "rows": [
        {"text": r.text, "label": r.sentiment, "vader": r.vader, "score": round(r.vader_score, 3),
         "tfidf": r.tfidf, "topic": r.route_mentioned}
        for r in d.itertuples()
    ],
}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out["methods"], indent=1))
