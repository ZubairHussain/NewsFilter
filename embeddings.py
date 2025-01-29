import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

class Word2VecEmbeddingGenerator:
    def __init__(self, download_url: str = None, filename: str = None, token: str = None, model_path: str = "data/models/"):
        """
        Initialize the embedding generator using the Hugging Face `word2vec-google-news-300` model.
        """
        if not Path(model_path).exists():
            if download_url and filename and token:
                print(f"\n\n Downloading the pre-trained model {filename} from Hugging face ...")
                model_path = hf_hub_download(repo_id=download_url, filename=filename, use_auth_token = token, cache_dir=model_path)
            
        print(f"Loading pre-trained Word2Vec model : {model_path}")
        self.model = KeyedVectors.load(model_path) 
        print(f"Pre-trained Word2Vec model loaded successfully!")

    def generate_embeddings(self, tokenized_texts):
        """
        Generate embeddings for tokenized texts using the pre-trained Word2Vec model.

        Args:
            tokenized_texts (list of list of str): Tokenized texts.

        Returns:
            np.ndarray: Generated embeddings.
        """
        embeddings = [
            np.mean(
                [self.model[word] for word in text if word in self.model] or [np.zeros(self.model.vector_size)],
                axis=0,
            )
            for text in tokenized_texts
        ]
        return np.array(embeddings)
