from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class RandomForestClassifierModel:
    def __init__(self, random_state=42):
        """
        Initialize the Random Forest Classifier Model.

        Args:
            random_state (int): Random state for reproducibility.
        """
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state)

    def train(self, X, y):
        """
        Train the Random Forest model.

        Args:
            X (np.ndarray): Feature matrix (Word2Vec embeddings).
            y (np.ndarray): Labels.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: X_test, y_test, y_pred for evaluation.
        """
        self.model.fit(X, y)
        
    def predict(self, X):
        """
        Predict using the trained Random Forest model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model's performance.

        Args:
            y_true (list): Ground truth labels.
            y_pred (list): Predicted labels.

        Returns:
            dict: Classification report.
        """
        return classification_report(y_true, y_pred, output_dict=True)
