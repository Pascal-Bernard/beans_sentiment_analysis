import argparse
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw
import json
import re

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# ----------------------------- NOTE ----------------------------------------- #

# The credentials are now hardcoded directly into the script.
# This is for demonstration purposes and is not recommended for production.
# I have shared some temprary working credential in the email that you can use for testing
client_id=""
client_secret=""
user_agent=""

# ---------------------------------------------------------------------------- #

# Create the Reddit instance by passing the credentials directly
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

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

def analyze_sentiment(reddit_instance, token, subreddit_name, limit, min_confidence):
    """
    Analyzes sentiment for a given crypto token from a live Reddit subreddit.
    
    Args:
        reddit_instance (praw.Reddit): The authenticated Reddit instance.
        token (str): The crypto token to analyze (e.g., SOL, BTC).
        subreddit_name (str): The name of the subreddit to search (e.g., 'cryptocurrency').
        limit (int): The number of posts to fetch from the subreddit.
        min_confidence (float): Minimum confidence threshold for a post to be included.
    """
    try:
        subreddit = reddit_instance.subreddit(subreddit_name)
    except Exception as e:
        return {"error": f"Could not connect to subreddit: {e}"}

    all_posts = []
    pattern = re.compile(r'\b' + re.escape(token) + r'\b', re.IGNORECASE)

    print(f"Fetching posts from r/{subreddit_name}...")
    for submission in subreddit.hot(limit=limit):
        if pattern.search(submission.title) or (submission.selftext and pattern.search(submission.selftext)):
            all_posts.append(submission.title + " " + submission.selftext)
        
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if pattern.search(comment.body):
                all_posts.append(comment.body)

    if not all_posts:
        return {"token": token, "bullish": 0, "neutral": 0, "bearish": 0, "bullish_ratio": 0.0, "top_posts": []}

    df = pd.DataFrame(all_posts, columns=['text'])
    
    df['sentiment'] = df['text'].apply(get_sentiment)
    df['label'] = df['sentiment'].apply(lambda x: x['label'])
    df['confidence'] = df['sentiment'].apply(lambda x: x['confidence'])
    
    df_filtered = df[df['confidence'] >= min_confidence].copy()

    total_posts = len(df_filtered)
    if total_posts == 0:
        return {"token": token, "bullish": 0, "neutral": 0, "bearish": 0, "bullish_ratio": 0.0, "top_posts": []}
        
    bullish_count = len(df_filtered[df_filtered['label'] == 'bullish'])
    neutral_count = len(df_filtered[df_filtered['label'] == 'neutral'])
    bearish_count = len(df_filtered[df_filtered['label'] == 'bearish'])
    bullish_ratio = round(bullish_count / total_posts, 2) if total_posts > 0 else 0.0

    top_posts = df_filtered.sort_values(by='confidence', ascending=False).head(10).to_dict('records')
    
    formatted_top_posts = [
        {"text": str(post['text'])[:120]+"...", "label": post['label'], "confidence": round(post['confidence'], 2)}
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
    parser.add_argument("--subreddit", default="CryptoCurrency", help="The subreddit to analyze (e.g., CryptoCurrency, dogecoin).")
    parser.add_argument("--limit", type=int, default=100, help="The number of posts to fetch from the subreddit.")
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Minimum confidence threshold for a post to be included (0.0 to 1.0).")
    
    args = parser.parse_args()
    
    # Analyze the sentiment with the hardcoded credentials
    analysis_result = analyze_sentiment(reddit, args.token.upper(), args.subreddit, args.limit, args.min_confidence)
    
    # Print the results as JSON
    print(json.dumps(analysis_result, indent=2))
