import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def train_and_evaluate_model(data_path):
    """
    Loads data, trains a Logistic Regression model, and evaluates its performance.

    Args:
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
    
    # Define features (X) and labels (y)
    X = df['text']
    y = df['true_sentiment']

    # Split data into training and testing sets.
    # We'll be using 80 % of the data for training and 20% for testing
    # The 'stratify' parameter ensures that the proportion of each sentiment class is
    # the same in both the training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # Feature Engineering: Convert text into numerical features using TF-IDF.
    # TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features.
    # It accounts for word frequency while giving less weight to common words.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model Training > to train a Logistic Regression classifier.
    # Logistic Regression is a simple yet powerful and interpretable algorithm for classification.
    print("Training the Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # 3. Model Evaluation: > make predictions and evaluate the model's performance.
    print("Evaluating the model...")
    y_pred = model.predict(X_test_vec)
    
    # Print the classification report
    print("\n----------------- Custom Model Classification Report: ----------------")
    print(classification_report(y_test, y_pred))
    
    # print the confusion matrix
    print("\n----------------- Custom Model Confusion Matrix: ----------------")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(cm)


if __name__ == "__main__":
    
    # Path to the training dataset
    training_data_file = 'data/synthetic_crypto_sentiment_training_1.5k.csv'
    train_and_evaluate_model(training_data_file)
