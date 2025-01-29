import numpy as np
import pandas as pd
from typing import List, Tuple
from textblob import TextBlob
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


# BERT Dataset Class
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class NewsProcessor:
    def __init__(self):
        pass

    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_json(file_path, lines=True) 

    def preprocess_text(self, data: pd.DataFrame) -> pd.DataFrame:
        data["content"] = data["content"].apply(lambda x: x.lower().strip())
        return data

    def filter_by_keyword(self, data: pd.DataFrame, target_keywords: List[str]) -> pd.DataFrame:        
        articles = []
        for _, row in data.iterrows():
            content = row["content"]
            if any(keyword in content for keyword in target_keywords):
                articles.append(row)
        return pd.DataFrame(articles).reset_index(drop=True)

    def generate_polarity(self, data: pd.DataFrame) ->pd.DataFrame:
        articles = []
        for _, row in data.iterrows():
            content = row["content"]
            row["polarity_score"] = TextBlob(content).sentiment.polarity
            articles.append(row)
        return pd.DataFrame(articles)
    
    def generate_labels(self, data: pd.DataFrame, polarity_thresholds: Tuple = (-0.2, 0.4)):
        
        if "polarity_score" not in data.columns:
            raise KeyError("The 'polarity_score' column is missing in the dataset.")

        data = data.reset_index(drop=True)

        data["label"] = 0 #neutral sentiment
        data.loc[np.where(data["polarity_score"] < polarity_thresholds[0])[0], "label"] = 1 #negative sentiment
        data.loc[np.where(data["polarity_score"] > polarity_thresholds[1])[0], "label"] = 2 #positive sentiment

        return data        
    
    def visualize_polarity_distribution(self, data, output_path="polarity_distribution.png"):
        """
        Visualize the distribution of sentiment polarity in the dataset and save it as an image.

        Args:
            data (pd.DataFrame): Dataset containing articles.
            output_path (str): Path to save the polarity distribution plot.

        Returns:
            None: Saves a histogram of sentiment polarity.
        """
        polarities = data["polarity_score"]
        plt.figure(figsize=(8, 6))
        plt.hist(polarities, bins=20, color="blue", alpha=0.7, edgecolor="black")
        plt.title("Sentiment Polarity Distribution")
        plt.xlabel("Polarity")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(output_path)
        plt.close()