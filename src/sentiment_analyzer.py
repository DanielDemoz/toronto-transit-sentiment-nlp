"""
Toronto Transit Sentiment Analyzer
Advanced NLP sentiment classification for transit-related text
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Optional: Transformer models
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TransitSentimentAnalyzer:
    """
    Comprehensive sentiment analyzer for Toronto transit feedback.
    Supports multiple models: VADER, TF-IDF + Logistic Regression, BERT
    """
    
    def __init__(self, model_type: str = 'tfidf'):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type: 'vader', 'tfidf', or 'bert'
        """
        self.model_type = model_type
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # TTC-specific keywords to preserve
        self.ttc_keywords = {
            'ttc', 'subway', 'streetcar', 'bus', 'delay', 'late', 'crowded',
            'clean', 'dirty', 'safe', 'unsafe', 'presto', 'fare', 'station',
            'line1', 'line2', 'line3', 'line4', 'yonge', 'bloor', 'spadina'
        }
        
        # Update stop words to keep TTC terms
        self.stop_words = self.stop_words - self.ttc_keywords
        
        # Initialize models
        self.vectorizer = None
        self.classifier = None
        self.bert_model = None
        
        if model_type == 'bert' and TRANSFORMERS_AVAILABLE:
            self.bert_model = pipeline("sentiment-analysis", 
                                       model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags (but keep the word)
        text = re.sub(r'[@#](\w+)', r'\1', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def extract_transit_features(self, text: str) -> Dict[str, any]:
        """
        Extract transit-specific features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        text_lower = text.lower()
        
        features = {
            'has_delay': any(word in text_lower for word in ['delay', 'late', 'slow', 'wait']),
            'has_cleanliness': any(word in text_lower for word in ['clean', 'dirty', 'smell', 'mess']),
            'has_safety': any(word in text_lower for word in ['safe', 'unsafe', 'security', 'danger']),
            'has_route': any(word in text_lower for word in ['line', 'route', 'subway', 'streetcar', 'bus']),
            'has_station': 'station' in text_lower,
            'has_price': any(word in text_lower for word in ['fare', 'price', 'cost', 'expensive', 'presto']),
            'exclamation_marks': text.count('!'),
            'question_marks': text.count('?'),
        }
        
        return features
    
    def train_tfidf_model(self, texts: List[str], labels: List[str]):
        """
        Train TF-IDF + Logistic Regression model.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (positive, neutral, negative)
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.8
        )
        
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Train classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        self.classifier.fit(X, labels)
        
        print("✓ TF-IDF model trained successfully!")
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict sentiment for given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment, confidence, and features
        """
        if self.model_type == 'vader':
            return self._predict_vader(text)
        elif self.model_type == 'tfidf':
            return self._predict_tfidf(text)
        elif self.model_type == 'bert':
            return self._predict_bert(text)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _predict_vader(self, text: str) -> Dict[str, any]:
        """VADER sentiment analysis"""
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        
        # Classify based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(scores['compound']),
            'scores': scores,
            'features': self.extract_transit_features(text)
        }
    
    def _predict_tfidf(self, text: str) -> Dict[str, any]:
        """TF-IDF + Logistic Regression prediction"""
        if self.classifier is None or self.vectorizer is None:
            raise ValueError("Model not trained! Call train_tfidf_model() first.")
        
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        sentiment = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': dict(zip(self.classifier.classes_, probabilities)),
            'features': self.extract_transit_features(text)
        }
    
    def _predict_bert(self, text: str) -> Dict[str, any]:
        """BERT-based prediction"""
        if not TRANSFORMERS_AVAILABLE or self.bert_model is None:
            raise ValueError("BERT model not available!")
        
        result = self.bert_model(text[:512])[0]  # BERT max length
        
        # Map BERT labels to our labels
        label_map = {'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral'}
        sentiment = label_map.get(result['label'], result['label'].lower())
        
        return {
            'sentiment': sentiment,
            'confidence': result['score'],
            'features': self.extract_transit_features(text)
        }
    
    def batch_predict(self, texts: List[str]) -> pd.DataFrame:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append({
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                **result.get('features', {})
            })
        
        return pd.DataFrame(results)
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'model_type': self.model_type,
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        
        print(f"✓ Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TransitSentimentAnalyzer(model_type='vader')
    
    # Test samples
    test_texts = [
        "The new streetcars are amazing! So comfortable and clean.",
        "Waited 30 minutes for a bus in freezing weather. This is unacceptable!",
        "Just another regular commute on the TTC.",
        "Presto card didn't work again. This system is so frustrating.",
        "Great service today, the driver was very helpful and friendly."
    ]
    
    print("=" * 60)
    print("TORONTO TRANSIT SENTIMENT ANALYSIS")
    print("=" * 60)
    
    for text in test_texts:
        result = analyzer.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Features: {result['features']}")
        print("-" * 60)

