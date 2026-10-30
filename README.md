# Toronto Transit Sentiment

How well can simple tools read the tone of rider messages about the TTC?

**Live demo:** https://danieldemoz.github.io/toronto-transit-sentiment-nlp/
**Status:** working pipeline, tested on 40 hand-labelled example messages. A small prototype, not an analysis of real rider opinion.

Part of a Toronto transit series: [ttc-predict](https://github.com/DanielDemoz/ttc-predict) | [TTC-Line1-Optimization](https://github.com/DanielDemoz/TTC-Line1-Optimization) | [toronto-transit-sentiment-nlp](https://github.com/DanielDemoz/toronto-transit-sentiment-nlp) | [interactive-transit-dashboard](https://github.com/DanielDemoz/interactive-transit-dashboard)

## What it does

`src/sentiment_analyzer.py` labels a message as negative, neutral or positive using VADER, TF-IDF with logistic regression, or an off-the-shelf DistilBERT model. `src/topic_model.py` groups messages into topics with LDA or NMF. `src/benchmark.py` scores the methods against hand labels and writes the results the demo page shows.

## Results

40 example messages: 17 negative, 7 neutral, 16 positive.

| Method | Accuracy | Macro F1 |
|---|---|---|
| Always guess the most common label | 42.5% | 0.20 |
| TF-IDF + logistic regression (5-fold CV) | 45.0% | 0.33 |
| **VADER lexicon, no training** | **80.0%** | **0.78** |

**Honest finding:** with 40 messages, the trained model barely beats guessing. The lexicon does better with no training. VADER's errors are complaints without angry words ("Felt very unsafe on the late night bus") and complaints phrased as questions. The demo lists every error.

## Responsible AI, following the Reliable AI Adoption Framework

1. **Business problem:** help a transit team sort rider feedback by tone and topic, so complaints reach the right people sooner.
2. **Data readiness:** 40 short example messages written for testing and labelled by one person. Enough to test the code, not to describe real riders.
3. **Governance and risk:** real posts contain usernames and locations. Any real use needs consent, data minimization and a privacy review. Safety messages must always reach a person.
4. **Technology and tools:** the simpler lexicon is the right choice until there are many more labelled messages.
5. **People and adoption:** scores sort the queue. People read the messages and decide what to do.
6. **Outcomes and monitoring:** re-score against fresh human labels each quarter, and track errors on safety messages separately.

## How to run

```bash
pip install -r requirements.txt
python src/benchmark.py      # rebuilds docs/results.json
pytest -q
```

## Project history

An earlier README and dashboard listed accuracy figures (including 88.7% for a fine-tuned BERT model) and topic shares that the code and data in this repository do not produce. They were removed in October 2026 and replaced with results the benchmark script generates.

## Tech stack

Python, pandas, scikit-learn, vaderSentiment, NLTK, optional transformers, vanilla JavaScript, GitHub Pages.
