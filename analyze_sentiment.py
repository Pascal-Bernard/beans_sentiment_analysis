import argparse
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone
import json
import re


# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    """
    Analyzes the sentiment of a given text using VADER.
    Returns a dictionary with sentiment label and confidence score.
    """
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        label = "bullish"
    elif compound_score <= -0.05:
        label = "bearish"
    else:
        label = "neutral"
        
    confidence = abs(compound_score)
    
    return {"label": label, "confidence": confidence}


def analyze_sentiment(token, window, min_confidence, filename='data/synthetic_crypto_sentiment_1.4k.csv'):
    """
    Analyzes sentiment for a given crypto token within a specified time window.
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        return {"error": f"Dataset file '{filename}' not found."}

    # FIX: Use a more robust regex to find the token
    # This regex looks for EITHER a dollar sign OR a word boundary,
    # followed by the token, followed by a word boundary.
    pattern = f'(?:\\$|\\b){re.escape(token)}\\b'
    df_filtered = df[df['text'].str.contains(pattern, case=False, na=False)].copy()
    
    if df_filtered.empty:
        return {"token": token, "bullish": 0, "neutral": 0, "bearish": 0, "bullish_ratio": 0.0, "top_posts": []}
    
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], utc=True)
    
    end_time = df_filtered['timestamp'].max()
    
    time_delta = pd.to_timedelta(window)
    start_time = end_time - time_delta
    
    df_filtered = df_filtered[(df_filtered['timestamp'] >= start_time) & (df_filtered['timestamp'] <= end_time)].copy()

    if df_filtered.empty:
        return {"token": token, "bullish": 0, "neutral": 0, "bearish": 0, "bullish_ratio": 0.0, "top_posts": []}

    df_filtered['sentiment'] = df_filtered['text'].apply(get_sentiment)

    df_filtered['label'] = df_filtered['sentiment'].apply(lambda x: x['label'])
    df_filtered['confidence'] = df_filtered['sentiment'].apply(lambda x: x['confidence'])
    
    df_filtered = df_filtered[df_filtered['confidence'] >= min_confidence]
    
    total_posts = len(df_filtered)
    if total_posts == 0:
        return {"token": token, "bullish": 0, "neutral": 0, "bearish": 0, "bullish_ratio": 0.0, "top_posts": []}
        
    bullish_count = len(df_filtered[df_filtered['label'] == 'bullish'])
    neutral_count = len(df_filtered[df_filtered['label'] == 'neutral'])
    bearish_count = len(df_filtered[df_filtered['label'] == 'bearish'])
    bullish_ratio = round(bullish_count / total_posts, 2) if total_posts > 0 else 0.0

    top_posts = df_filtered.sort_values(by='confidence', ascending=False).head(10).to_dict('records')
    
    formatted_top_posts = [
        {"text": post['text'], "label": post['label'], "confidence": round(post['confidence'], 2)}
        for post in top_posts
    ]
    
    result = {
        "token": token,
        "bullish": bullish_count,
        "neutral": neutral_count,
        "bearish": bearish_count,
        "bullish_ratio": bullish_ratio,
        "top_posts": formatted_top_posts
    }
    
    return result


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Analyze social media sentiment for crypto tokens.")
    parser.add_argument("--token", required=True, help="The crypto token to analyze (e.g., SOL, DOGE, PEPE).")
    parser.add_argument("--window", required=True, help="The time window for analysis (e.g., 1h, 24h, 7d).")
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Minimum confidence threshold for a post to be included (0.0 to 1.0).")
    parser.add_argument("--dataset", default="data/synthetic_crypto_sentiment_1.4k.csv", help="The path to the dataset CSV file.")
    
    args = parser.parse_args()
    
    analysis_result = analyze_sentiment(args.token.upper(), args.window, args.min_confidence, args.dataset)
    print(json.dumps(analysis_result, indent=2))