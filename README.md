# News Filtering and Sentiment Analysis System

##  Project Overview
This project builds an **automated news filtering and sentiment analysis system** that processes articles from multiple sources and determines whether they should be pushed to users. The system analyzes the sentiment of news articles and categorizes them based on their potential impact on organizations of interest.

## Approaches Implemented
The project provides multiple approaches for **sentiment classification and regression**, leveraging **traditional ML models, Word2Vec, and BERT embeddings**.

### **Approach 1: Sentiment Analysis Using TextBlob**
- Computes **polarity scores** using TextBlob.
- Assigns **sentiment labels (-1, 0, 1)** based on predefined thresholds.
- **Simple and effective**, but lacks contextual understanding.

### **Approach 2: KMeans Clustering for Unsupervised Sentiment Labeling**
- Generates **Word2Vec embeddings** for articles.
- Uses **KMeans clustering** to categorize similar articles.
- Groups articles based on **semantic similarity** instead of predefined labels.

### **Approach 3: Regression-Based Sentiment Score Prediction**
- Treats sentiment analysis as a **regression problem**.
- Predicts **continuous polarity scores (-1 to +1)** instead of discrete labels.
- Uses **Random Forest Regressor** with **Word2Vec embeddings**.

### **Approach 4: Sentiment Classification Using Word Embeddings**
- Frames the problem as a **classification task**.
- Uses **Random Forest Classifier** to predict sentiment labels.
- Works well for **structured classification**, but requires balanced training data.

### **Approach 5: BERT-Based Sentiment Analysis (LLM Approach)**
- Uses **Sentence-BERT (SBERT) embeddings** for better **contextual understanding**.
- **Trains a Random Forest Classifier** on BERT embeddings.
- **More accurate** than Word2Vec since it captures **word order & context**.

---

## Installation & Dependencies

### **1️. Setup conda environment**
```bash
conda create -n news_filter python=3.10

conda activate news_filter

pip install -r requirements.txt
```

## Project Structure
```
News_Filtering_System/
├── configs/                # Configuration files
├── data/                   
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for experimentation
│   ├── tutorial.ipynb      
├── scripts/                # Scripts for data processing and model training
│   ├── preprocess.py       # Prepares and cleans data for training
│   ├── bert_regressor.py   # Implements a BERT-based regression model
│   ├── rf_classifier.py    # Random Forest classifier for news categorization
│   ├── rf_regressor.py     # Random Forest regression model for prediction
├── src/                    # Source code for the project
│   ├── data
│   |   ├── dataset.py      # News Dataset for BERT      
│   ├── model
│   |   ├── bert.py         # BERT   
│   ├── utils
│   |   ├── data_processor.py #data utils
│   |   ├── embeddings.py   # Word2Vec embeddings       
├── tests/                  # Unit tests
│   ├── test_data_processor.py  # Tests for data preprocessing
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Usage

### Running the Tutorial

To get started, open the Jupyter notebook:
```bash
jupyter notebook notebooks/tutorial.ipynb
```
This notebook provides an overview of the project, data preprocessing steps, model training procedures and conclusion and observations for each approach.


## Running the Scripts

### 1. Preprocessing Data
```bash
python scripts/preprocess.py --input_file="data/raw/signalmedia-1m_challenge_dataset/signal-1m-nasa.jsonl" --output_dir="data/processed/" --polarity_thresholds="-0.1,0.2"
```
- `--input_file`: Path to the input json file.
- `--output_dir`: Directory to save the processed data.
- `--polarity_thresholds`: Sentiment polarity threshold for categorizing articles into sentiment labels.

### 2. Train and evaluate a Random Forest Classifer
```bash
python scripts/rf_classifier.py --input_file="data/processed/articles_with_labels.csv" --config_file="configs/general.yaml" --output_dir="outputs/" 
```
- `--input_file`: Path to the input csv file.
- `--output_dir`: Directory to save the trained model.
- `--config_file`: Path to the config file (for downloading the pretrained Word2Vec model.)

### 3. Train and evaluate a Random Forest Regressor
```bash
python scripts/rf_regressor.py --input_file="data/processed/articles_with_labels.csv" --config_file="configs/general.yaml" --output_dir="outputs/" 
```
- `--input_file`: Path to the input csv file.
- `--output_dir`: Directory to save the trained model.
- `--config_file`: Path to the config file (for downloading the pretrained Word2Vec model.)

### 4. Fine-tune and evaluate a BERT Regressor model
```bash
python scripts/bert_regressor.py --input_file="data/processed/articles_with_labels.csv" --output_dir="outputs/BERT/" --config_file="configs/general.yaml"
```
- `--input_file`: Path to the input csv file.
- `--output_dir`: Directory to save the trained model.
- `--config_file`: Path to the config file for training settings.

### Running Tests
To verify the implementation, run:
```sh
pytest tests/
```

