"""
Topic Modeling for Toronto Transit Feedback
Identifies common themes and discussion topics using LDA and NMF
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re

# NLP Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class TransitTopicModeler:
    """
    Topic modeling for transit feedback using LDA and NMF.
    """
    
    def __init__(self, n_topics: int = 5, method: str = 'lda'):
        """
        Initialize topic modeler.
        
        Args:
            n_topics: Number of topics to extract
            method: 'lda' or 'nmf'
        """
        self.n_topics = n_topics
        self.method = method
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        
    def preprocess_for_topics(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts specifically for topic modeling.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of preprocessed texts
        """
        processed = []
        
        for text in texts:
            # Lowercase
            text = text.lower()
            
            # Remove URLs, mentions
            text = re.sub(r'http\S+|www.\S+|@\w+|#', '', text)
            
            # Keep only letters and spaces
            text = re.sub(r'[^a-z\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            processed.append(text)
        
        return processed
    
    def fit_lda(self, texts: List[str]) -> LatentDirichletAllocation:
        """
        Fit LDA topic model using sklearn.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Fitted LDA model
        """
        # Create document-term matrix
        self.vectorizer = CountVectorizer(
            max_df=0.8,
            min_df=5,
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit LDA
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch'
        )
        
        self.model.fit(doc_term_matrix)
        
        print(f"✓ LDA model fitted with {self.n_topics} topics")
        return self.model
    
    def fit_nmf(self, texts: List[str]) -> NMF:
        """
        Fit NMF topic model.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Fitted NMF model
        """
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_df=0.8,
            min_df=5,
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit NMF
        self.model = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=500
        )
        
        self.model.fit(tfidf_matrix)
        
        print(f"✓ NMF model fitted with {self.n_topics} topics")
        return self.model
    
    def get_topics(self, n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract top words for each topic.
        
        Args:
            n_words: Number of top words to return per topic
            
        Returns:
            Dictionary mapping topic number to list of (word, weight) tuples
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not fitted! Call fit_lda() or fit_nmf() first.")
        
        topics = {}
        
        for topic_idx, topic in enumerate(self.model.components_):
            # Get top word indices
            top_indices = topic.argsort()[-n_words:][::-1]
            
            # Get words and weights
            top_words = [
                (self.feature_names[i], topic[i])
                for i in top_indices
            ]
            
            topics[topic_idx] = top_words
        
        return topics
    
    def display_topics(self, n_words: int = 10):
        """
        Display topics in readable format.
        
        Args:
            n_words: Number of words to display per topic
        """
        topics = self.get_topics(n_words)
        
        # Topic labels based on common TTC themes
        topic_labels = {
            0: "Service Delays & Reliability",
            1: "Cleanliness & Maintenance",
            2: "Safety & Security",
            3: "Fare & Payment",
            4: "Customer Service",
            5: "Route & Schedule",
            6: "Accessibility",
            7: "Vehicle Quality"
        }
        
        print("=" * 70)
        print("DISCOVERED TOPICS IN TORONTO TRANSIT FEEDBACK")
        print("=" * 70)
        
        for topic_id, words in topics.items():
            label = topic_labels.get(topic_id, f"Topic {topic_id}")
            
            print(f"\n📌 {label} (Topic {topic_id})")
            print("-" * 70)
            
            word_str = ", ".join([f"{word} ({weight:.3f})" for word, weight in words])
            print(f"   {word_str}")
        
        print("=" * 70)
    
    def get_document_topics(self, texts: List[str]) -> np.ndarray:
        """
        Get topic distribution for documents.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Array of shape (n_docs, n_topics) with topic probabilities
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not fitted!")
        
        if self.method == 'lda':
            doc_term_matrix = self.vectorizer.transform(texts)
            return self.model.transform(doc_term_matrix)
        else:  # NMF
            tfidf_matrix = self.vectorizer.transform(texts)
            return self.model.transform(tfidf_matrix)
    
    def assign_dominant_topic(self, texts: List[str]) -> pd.DataFrame:
        """
        Assign dominant topic to each document.
        
        Args:
            texts: List of texts
            
        Returns:
            DataFrame with texts and their dominant topics
        """
        topic_distributions = self.get_document_topics(texts)
        dominant_topics = topic_distributions.argmax(axis=1)
        topic_confidences = topic_distributions.max(axis=1)
        
        results = pd.DataFrame({
            'text': texts,
            'dominant_topic': dominant_topics,
            'confidence': topic_confidences
        })
        
        return results
    
    def visualize_topics(self, save_path: str = None):
        """
        Create visualization of topic word clouds.
        
        Args:
            save_path: Optional path to save figure
        """
        from wordcloud import WordCloud
        
        topics = self.get_topics(n_words=50)
        
        n_cols = 2
        n_rows = (self.n_topics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if self.n_topics > 1 else [axes]
        
        for topic_id, words in topics.items():
            # Create word frequency dict
            word_freq = {word: weight for word, weight in words}
            
            # Generate word cloud
            wc = WordCloud(
                width=400,
                height=300,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)
            
            # Plot
            axes[topic_id].imshow(wc, interpolation='bilinear')
            axes[topic_id].set_title(f'Topic {topic_id}', fontsize=14, fontweight='bold')
            axes[topic_id].axis('off')
        
        # Hide extra subplots
        for i in range(self.n_topics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
    
    def plot_topic_distribution(self, texts: List[str], save_path: str = None):
        """
        Plot distribution of topics across documents.
        
        Args:
            texts: List of texts
            save_path: Optional path to save figure
        """
        topic_distributions = self.get_document_topics(texts)
        topic_counts = topic_distributions.argmax(axis=1)
        
        plt.figure(figsize=(10, 6))
        
        # Count topics
        unique, counts = np.unique(topic_counts, return_counts=True)
        
        # Create bar plot
        plt.bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Topic Number', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.title('Distribution of Dominant Topics Across Documents', fontsize=14, fontweight='bold')
        plt.xticks(unique)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Distribution plot saved to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Sample TTC feedback data
    sample_texts = [
        "The subway was delayed again this morning. Always late during rush hour!",
        "Streetcar was very clean today, impressed with the maintenance.",
        "Felt unsafe on the late night bus, need more security presence.",
        "Presto card system is confusing and often doesn't work properly.",
        "The driver was rude when I asked for directions. Poor customer service.",
        "Bus didn't show up according to schedule. Waited 40 minutes!",
        "New accessible ramps at the station are great for wheelchair users.",
        "The new trains are comfortable but still experience frequent delays.",
        "Station smells terrible and is very dirty. Needs cleaning.",
        "Fare prices keep increasing but service quality is decreasing.",
    ]
    
    # Initialize and fit model
    modeler = TransitTopicModeler(n_topics=5, method='lda')
    
    # Preprocess
    processed_texts = modeler.preprocess_for_topics(sample_texts)
    
    # Fit model
    modeler.fit_lda(processed_texts)
    
    # Display topics
    modeler.display_topics(n_words=8)
    
    # Assign dominant topics
    results = modeler.assign_dominant_topic(processed_texts)
    print("\n" + "=" * 70)
    print("DOCUMENT-TOPIC ASSIGNMENTS")
    print("=" * 70)
    print(results)

