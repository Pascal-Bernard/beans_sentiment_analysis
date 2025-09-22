import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np


# A custom PyTorch Dataset class to handle the text data
class TextDataset(Dataset):
    def __init__(self, X, y):
        # The data is already vectorized, so we can convert it to tensors
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# A simple neural network model for text classification
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        # Define a simple feed-forward neural network
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        # Pass the input through the layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_and_evaluate_model(data_path):
    """
    Loads data, trains a PyTorch neural network, and evaluates its performance.
    Input args:
        data_path (str): The path to the training CSV file.
    """
    try:
        # Load the training dataset. We'll assume the provided dataset is saved as this filename.
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        print("Please ensure the training dataset is in the same directory and named correctly.")
        return

    # Drop any rows where the 'text' or 'true_sentiment' columns are missing.
    df.dropna(subset=['text', 'true_sentiment'], inplace=True)
    
    # Map sentiment labels to numerical values for PyTorch
    label_map = {label: i for i, label in enumerate(df['true_sentiment'].unique())}
    df['true_sentiment'] = df['true_sentiment'].map(label_map)
    
    # Define features (X) and labels (y)
    X = df['text']
    y = df['true_sentiment']

    # Split data into training and testing sets.
    # We'll use 80% of the data for training and 20% for testing.
    # The 'stratify' parameter ensures that the proportion of each sentiment class is
    # the same in both the training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # Feature Engineering: Convert text into numerical features using TF-IDF.
    # mnote : TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    output_dim = len(label_map)
    
    # Create PyTorch Datasets and DataLoaders
    train_dataset = TextDataset(X_train_vec, y_train.values)
    test_dataset = TextDataset(X_test_vec, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model Training: Train the PyTorch classifier.
    print("Training the PyTorch neural network...")
    model = SentimentClassifier(input_dim=X_train_vec.shape[1], output_dim=output_dim)
    
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train() # Set model to training mode
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Model Evaluation: Make Predictions > evaluate the model performance.
    print("Evaluating the model...")
    model.eval() # Set model to evaluation mode
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    # Get original label from the label map for the clasisfication report
    inv_label_map = {i: label for label, i in label_map.items()}
    class_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

    
    # Print the classification report
    print("\n----------------- Custom PyTorch Model Classification Report: ----------------")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Print the confusion matrix
    print("\n----------------- Custom PyTorch Model Confusion Matrix: ----------------")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


if __name__ == "__main__":
    # Path to the training dataset
    training_data_file = 'data/synthetic_crypto_sentiment_training_1.5k.csv'
    train_and_evaluate_model(training_data_file)
