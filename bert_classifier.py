from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
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

# BERT Classifier Class
class BERTClassifier:
    def __init__(self, model_name="bert-base-uncased", num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def train(self, texts, labels, epochs=3, batch_size=4):
        """
        Train the BERT model.

        Args:
            texts (list): List of texts.
            labels (list): Corresponding labels.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size.

        Returns:
            Trainer: Trained BERT model.
        """
        dataset = BERTDataset(texts, labels, self.tokenizer)
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir="./logs",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        trainer.train()
        return trainer