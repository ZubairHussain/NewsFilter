import numpy as np
import pandas as pd
from typing import List, Tuple
from textblob import TextBlob
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize stopwords and lemmatizer
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

class NewsProcessor:
    def __init__(self):
        """
        Initializes the NewsProcessor class.
        This class is responsible for loading, preprocessing, filtering, analyzing sentiment, 
        and visualizing distribution of sentiment polarity scores.
        """
        pass

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load news articles from a JSON file.

        Args:
            file_path (str): Path to the JSONL file containing news articles.

        Returns:
            pd.DataFrame: DataFrame containing the loaded articles.
        """
        try:
            return pd.read_json(file_path, lines=True)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()

    def preprocess_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the content of news articles by converting text to lowercase and stripping spaces.

        Args:
            data (pd.DataFrame): DataFrame containing news articles.

        Returns:
            pd.DataFrame: Processed DataFrame with cleaned text.
        """
        try:
            #article = data["content"].apply(lambda x: x.lower().strip() if isinstance(x, str) else "")
            article = data["content"].lower()
            # Remove URLs
            article = re.sub(r"http\S+|www\S+|https\S+", "", article)
            # Remove special characters & numbers
            article = re.sub(r"[^a-z\s]", "", article)
            # Tokenize using NLTK
            article = word_tokenize(article)
            # Remove stopwords and apply lemmatization
            article = [LEMMATIZER.lemmatize(word) for word in article if word not in STOPWORDS]
            data["content"] = article
            return data
        except Exception as e:
            print(f"Error during text preprocessing: {e}")
            return data

    def filter_by_keyword(self, data: pd.DataFrame, target_keywords: List[str]) -> pd.DataFrame:
        """
        Filter articles that contain at least one of the target keywords.

        Args:
            data (pd.DataFrame): DataFrame containing news articles.
            target_keywords (List[str]): List of keywords to filter the articles.

        Returns:
            pd.DataFrame: Filtered DataFrame containing articles with relevant keywords.
        """
        try:
            articles = [row for _, row in data.iterrows() if any(keyword in row["content"] for keyword in target_keywords)]
            return pd.DataFrame(articles).reset_index(drop=True)
        except Exception as e:
            print(f"Error filtering articles by keywords: {e}")
            return data

    def generate_polarity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sentiment polarity scores for each news article using TextBlob.

        Args:
            data (pd.DataFrame): DataFrame containing news articles.

        Returns:
            pd.DataFrame: DataFrame with an additional 'polarity_score' column.
        """
        try:
            articles = []
            for _, row in data.iterrows():
                row["polarity_score"] = TextBlob(row["content"]).sentiment.polarity
                articles.append(row)
            return pd.DataFrame(articles)
        except Exception as e:
            print(f"Error computing polarity scores: {e}")
            return data

    def generate_labels(self, data: pd.DataFrame, polarity_thresholds: Tuple = (-0.2, 0.4)) -> pd.DataFrame:
        """
        Assign sentiment labels based on polarity scores.

        Args:
            data (pd.DataFrame): DataFrame containing polarity scores.
            polarity_thresholds (Tuple): Thresholds for classifying negative, neutral, and positive sentiments.

        Returns:
            pd.DataFrame: DataFrame with an additional 'label' column for sentiment classification.
        """
        try:
            if "polarity_score" not in data.columns:
                raise KeyError("The 'polarity_score' column is missing in the dataset.")
            
            data = data.reset_index(drop=True)
            data["sentiment"] = 0  # Default to neutral sentiment
            data.loc[data["polarity_score"] < polarity_thresholds[0], "sentiment"] = -1  # Negative sentiment
            data.loc[data["polarity_score"] > polarity_thresholds[1], "sentiment"] = 1  # Positive sentiment
            
            return data
        except Exception as e:
            print(f"Error generating labels: {e}")
            return data
    
    def visualize_polarity_distribution(self, data: pd.DataFrame, output_path: str = None):
        """
        Visualize the distribution of sentiment polarity in the dataset.
        
        If the output path exists, the plot is saved. Otherwise, it returns a Matplotlib figure 
        for inline visualization in Jupyter Notebook.

        Args:
            data (pd.DataFrame): Dataset containing articles with polarity scores.
            output_path (str): Path to save the polarity distribution plot.

        Returns:
            plt.Figure: Matplotlib figure if output_path doesn't exist, else saves the figure.
        """
        try:
            if "polarity_score" not in data.columns:
                raise KeyError("The 'polarity_score' column is missing in the dataset.")
            
            polarities = data["polarity_score"]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(polarities, bins=20, color="blue", alpha=0.7, edgecolor="black")
            ax.set_title("Sentiment Polarity Distribution")
            ax.set_xlabel("Polarity")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Check if the output path exists; if not, return the figure for inline display
            if output_path:
                plt.savefig(output_path)
                plt.close()
            else:
                return fig  # Return figure for inline display in Jupyter Notebook

        except Exception as e:
                print(f"Error visualizing polarity distribution: {e}")

    def cluster_articles(self, embeddings: np.ndarray, n_clusters=3) -> np.ndarray:
        """
        Cluster news articles using KMeans clustering algorithm.
        
        Args:
            embeddings (np.ndarray): Word embeddings of news articles.
            n_clusters (int): Number of clusters to form.
        
        Returns:
            np.ndarray: Cluster labels assigned to each news article.
        """
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            return cluster_labels
        except Exception as e:
            print(f"Error clustering articles: {e}")
            return np.array([])
    
    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray, output_path="clusters_visualization.png"):
        """
        Visualize clustered articles in a 2D space using PCA.
        
        Args:
            embeddings (np.ndarray): Word embeddings of news articles.
            labels (np.ndarray): Cluster labels.
            output_path (str): Path to save the visualization.
        """
        try:
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(8, 6))
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.title("News Articles Clustering")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.colorbar(label="Cluster Label")
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            print(f"Error visualizing clusters: {e}")

