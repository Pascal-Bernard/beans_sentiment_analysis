import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline


def get_vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return "bullish"
    elif compound_score <= -0.05:
        return "bearish"
    else:
        return "neutral"


def get_bertweet_sentiment(text):
    classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    result = classifier(text)[0]
    
    label = result['label']
    
    if label == "POS":
        return "bullish"
    elif label == "NEG":
        return "bearish"
    else: # This covers the "NEU" label
        return "neutral"


# Main Evaluation Logic
def evaluate_models(filename='data/synthetic_crypto_sentiment_training_1.5k.csv'):
    """
    Loads a labeled dataset and evaluates the performance of two sentiment models.
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: Dataset file '{filename}' not found.")
        return

    # FIX: Use the correct column name 'true_sentiment' from the CSV file
    ground_truth_labels = df['true_sentiment'].str.lower()
    
    # Map 'positive' and 'negative' labels to 'bullish' and 'bearish'
    ground_truth_labels = ground_truth_labels.replace('positive', 'bullish')
    ground_truth_labels = ground_truth_labels.replace('negative', 'bearish')

    print("--- Running Evaluation ---")
    print(f"Evaluating models on {len(df)} posts.\n")

    # Evaluate VADER
    print("Evaluating VADER...")
    vader_predictions = df['text'].apply(get_vader_sentiment)
    
    print("\nVADER Classification Report:")
    print(classification_report(ground_truth_labels, vader_predictions))
    print("\nVADER Confusion Matrix:")
    print(confusion_matrix(ground_truth_labels, vader_predictions))
    
    print("\n" + "="*50 + "\n")

    # Evaluate BERTweet
    print("Evaluating BERTweet...")
    bertweet_predictions = df['text'].apply(get_bertweet_sentiment)
    
    print("\nBERTweet Classification Report:")
    print(classification_report(ground_truth_labels, bertweet_predictions))
    print("\nBERTweet Confusion Matrix:")
    print(confusion_matrix(ground_truth_labels, bertweet_predictions))


if __name__ == "__main__":
    
    evaluate_models()