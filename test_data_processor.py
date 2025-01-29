import numpy as np
import pandas as pd
import pytest

from src.data_processor import NewsProcessor

def test_data_processor():
     
    data_path = "../data/raw/signal-1m-nasa.jsonl"

    data_processor = NewsProcessor()

    print(f"Loading data ...")
    news_df = data_processor.load_data(file_path=data_path)
    assert len(news_df) > 0, "Data loading failed."
    
    print(f"Preprocessing text ...")
    news_df = data_processor.preprocess_text(data=news_df)
    assert len(news_df) > 0, "Data preprocessing failed."

    target_keywords = ["nasa", "space", "rocket"]
    print(f"Filtering articles by keywords: {target_keywords}")
    news_df = data_processor.filter_by_keyword(news_df, target_keywords=target_keywords)
    print(f"No. of filtered articles : {len(news_df)}")
    assert len(news_df) > 0, "Article filter by keyword failed."
    news_df.to_csv("../data/processed/filtered_articles_by_keywords.csv")

    news_df = data_processor.generate_polarity(news_df)
    assert len(news_df) > 0, "Generate polarity function failed."
    news_df.to_csv("../data/processed/articles_with_sentiment_polarity.csv")

    news_df = data_processor.generate_labels(news_df)
    assert len(news_df) > 0, "Generate labels function failed."
    news_df.to_csv("../data/processed/articles_with_labels.csv")
