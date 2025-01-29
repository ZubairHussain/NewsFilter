import numpy as np
import pandas as pd
from src.data_utils import NewsProcessor
from src.embeddings import Word2VecEmbeddingGenerator


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # Initialize Data Processor
    file_path = "data/raw/signalmedia-1m_challenge_dataset/signal-1m-nasa.jsonl"
    data_processor = NewsProcessor()

    news_df = data_processor.load_data(file_path=file_path)
    news_df = data_processor.preprocess_text(data=news_df)
    
    news_df = data_processor.filter_by_keyword(news_df, target_keywords=["nasa", "space", "rocket"])
    print('no. of filtered articles : ', len(news_df))
    
    news_df = data_processor.generate_polarity(news_df)
    news_df = data_processor.generate_labels(news_df)

    print('unique labels : ', set(news_df.label))

    print('no of articles with positive sentiment : ', np.where(news_df.label == 1)[0].shape)
    print('no of articles with negative sentiment : ', np.where(news_df.label == -1)[0].shape)
    print('no of articles with neutral sentiment : ', np.where(news_df.label == 0)[0].shape)

    data_processor.visualize_polarity_distribution(news_df)


    # Generate Word2Vec embeddings using pretrained model
    #model_path = "/path/to/pretrained/word2vec.bin"  # Update with actual path
    #token = "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"
    #embedding_generator = Word2VecEmbeddingGenerator(download_url = "fse/word2vec-google-news-300", filename = "word2vec-google-news-300.model", token = token, model_path = "data/models/")
    embedding_generator = Word2VecEmbeddingGenerator(model_path = "data/models/models--fse--word2vec-google-news-300/snapshots/528f381952a0b7d777bb4a611c4a43f588d48994/word2vec-google-news-300.model")
    tokenized_texts = [content.split() for content in news_df["content"]]
    embeddings = embedding_generator.generate_embeddings(tokenized_texts)
    labels = news_df.get("label", [0] * len(news_df))

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    y_pred = model.predict(y_test)
    print("Random Forest Evaluation:")
    print(classification_report(y_test, y_pred))
