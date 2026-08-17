# Toronto Transit Sentiment Analysis

NLP pipeline that classifies public sentiment toward Toronto transit from social-style text and surfaces recurring discussion topics.

> **Topic:** `toronto-transit` — Toronto Transit Commission (TTC) analytics series.

Part of a Toronto transit analytics series: [ttc-predict](https://github.com/DanielDemoz/ttc-predict) | [TTC-Line1-Optimization](https://github.com/DanielDemoz/TTC-Line1-Optimization) | [toronto-transit-sentiment-nlp](https://github.com/DanielDemoz/toronto-transit-sentiment-nlp) | [interactive-transit-dashboard](https://github.com/DanielDemoz/interactive-transit-dashboard)

## Problem

Transit agencies and analysts need structured insight from unstructured rider feedback—what people praise, complain about, and which stations or themes dominate negative sentiment.

## Approach

Built a multi-stage NLP workflow: text preprocessing, VADER baseline sentiment, TF-IDF + logistic regression, and fine-tuned BERT for transit-specific classification. Added LDA/NMF topic modeling, named entity extraction for routes and stations, and trend visualizations over time.

## Results

- Sentiment distribution: 32% positive, 41% neutral, 27% negative
- Top topics: delays & reliability (38%), cleanliness (22%), safety (18%), fare pricing (12%), service quality (10%)
- Model comparison: VADER 72.3% accuracy; logistic regression + TF-IDF 81.5%; fine-tuned BERT 88.7%

## Tech stack

Python, NLTK, spaCy, transformers (BERT), scikit-learn, pandas, matplotlib, seaborn, wordcloud

## How to run

```bash
git clone https://github.com/DanielDemoz/toronto-transit-sentiment-nlp.git
cd toronto-transit-sentiment-nlp
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon stopwords punkt
jupyter notebook
```

Or use the Python API:

```python
from src.sentiment_analyzer import TransitSentimentAnalyzer
analyzer = TransitSentimentAnalyzer()
result = analyzer.predict("The new streetcars are amazing but the delays are frustrating!")
```

## Screenshot / demo

**Interactive results dashboard:** https://danieldemoz.github.io/toronto-transit-sentiment-nlp/
