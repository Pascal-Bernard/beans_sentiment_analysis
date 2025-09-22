Crypto Sentiment Analysis Tool
------------------------------

### Project Objective

This project aims to create a command-line tool in **Python** to analyze sentiment in crypto-related social media posts. The tool leverages Python's extensive ecosystem for natural language processing (NLP) and data analysis to provide actionable insights into market sentiment.

### AI Usage Disclosure

I have only utilized Anthropic Claude.ai for the following :
- Wording completion, grammar, and spelling corrections.
- Code completion
The entire process of brainstorming, reasoning, machine learning approach, and option selection was prototyped and developed without any use of AI tools.

### Datasets

Two primary datasets are used in this project:

*   **synthetic\_crypto\_sentiment\_1.4k.csv**: This serves as the main dataset for real-world sentiment analysis.
    
*   **synthetic\_crypto\_sentiment\_training\_1.5k.csv**: A labeled dataset used for evaluating and comparing the performance of different sentiment models.
    

### Methodology

The tool's core functionality is a classifier that categorizes text into three sentiment classes: **bullish**, **bearish**, or **neutral**.

The primary strategy involves an initial assessment of pre-trained models. This approach is favored because these models are often effective at understanding social media slang and crypto-specific jargon, and the limited size of the training dataset (1.5k examples) poses a significant risk of overfitting if a custom model were to be trained from scratch.

To establish a performance baseline, the following three models will be evaluated and compared:

*   **VADER** (Valence Aware Dictionary and sEntiment Reasoner)
    
*   **DistilBERT**
    
*   **BERTweet**
    

The evaluation results will guide the decision on whether to develop a custom model for improved accuracy.

<br>

Model overview and why we shortlisted them
------------------------------------------

### VADER (Valence Aware Dictionary and and sEntiment Reasoner)

VADER is a rule-based model specifically tuned for social media sentiment. It's highly relevant because it doesn't require training data and is effective at understanding common internet slang and emojis, making it a quick and lightweight baseline for initial evaluation.

### DistilBERT

DistilBERT is a smaller, faster version of BERT, making it efficient while retaining strong performance. Its relevance lies in its ability to capture complex language context and nuances, offering a powerful, generalized comparison against VADER's rule-based approach.

### BERTweet

BERTweet is a variant of the BERT model trained exclusively on a massive dataset of tweets. This specialization makes it highly relevant for analyzing crypto social media posts, as its training data closely matches the language, style, and context of your project's target text.


<br>

I'll be using the provided Python script as a **base template** for testing all three sentiment models: VADER, DistilBERT, and BERTweet. This template is a flexible command-line tool for crypto sentiment analysis.

### Template Structure

The template is organized into key sections for clarity and modularity.

#### Imports and Initialization

The script begins by importing all necessary libraries and initializing the sentiment model. For example, the pipeline for a Hugging Face model is created once at the start to ensure efficiency.

```python
import argparse
import pandas as pd
...
from transformers import pipeline

# Initialize the sentiment analysis pipeline using the BERTweet model.
classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
```

The `get_sentiment()` Function


This is the core function for sentiment analysis. It takes a text string and returns the sentiment label and a confidence score. This is the only section that will change when we switch between models.

```python
def get_sentiment(text):
    """
    Analyzes the sentiment of a given text
    """
    result = classifier(text)[0]
    
    label = result['label']
    confidence = result['score']
    
    ... # Logic to map labels and return sentiment
    
    return {"label": label, "confidence": confidence}
```

#### The `analyze_sentiment()` function

This main analysis engine handles all data processing. It reads the dataset, filters posts by a specific token and time window, applies the get_sentiment() function to each post, and calculates the final sentiment counts.

```python
def analyze_sentiment(
    token, window, min_confidence, 
    filename='data/synthetic_crypto_sentiment_1.4k.csv'
    ):
    """
    Analyzes sentiment for a given crypto token
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        return {"error": f"Dataset file '{filename}' not found."}

    ... # Data filtering logic
    
    df_filtered['sentiment'] = df_filtered['text'].apply(get_sentiment)
    
    ... # Sentiment counting and result formatting
```


#### Main Block

The final section sets up the command-line interface, allowing you to run the script with different parameters directly from the terminal.

```python
if __name__ == "__main__":

    parser = argparse.ArgumentParser(...)
    parser.add_argument("--token", "--window", required=True, ...)
    
    args = parser.parse_args()
    
    analysis_result = analyze_sentiment(args.token.upper(), ...)
    print(json.dumps(analysis_result, indent=2))
```


How to run the script
---------------------

This project is a command-line tool, and getting it set up is a straightforward process.


### 1. Download the Repository
First, you'll need a copy of the project files on your local machine. You can do this in two ways:


Download as a ZIP: On the main GitHub page for the repository, click the green Code button and select Download ZIP. This will download all the files in a single compressed folder. Unzip the folder to a location of your choice.


Clone the Repository (Recommended): If you have Git installed, you can clone the repository directly from your terminal. This is the preferred method as it makes it easy to pull future updates.

`
git clone [https://github.com/Pascal-Bernard/beans_sentiment_analysis.git](https://github.com/Pascal-Bernard/beans_sentiment_analysis.git)
`


### 2. Install Dependencies
The script requires a few Python libraries to run, which are listed in the requirements.txt file. You can install all of them at once using pip:

`
pip install -r requirements.txt
`

### 3. Run the Analysis

Once the dependencies are installed, you can run the sentiment analysis script from your terminal.
Navigate to the project directory in your terminal:

`cd beans_sentiment_analysis`

Then, execute the script by specifying the crypto token you want to analyze. The script defaults to VADER, but you can choose between other models using the --model flag.

Example: Analyzing Bitcoin (BTC) using the VADER model using SOL (Solana) and a time windows of 240 hours :

`python analyze_sentiment.py --token SOL --window 2400h --min-confidence 0.50`

NOTE : 
- The default value of 0.75 can be too high for some models to provide an output. That's why I suggest to set at 0.50
- Extend the window value to : 2400h in order to cover the maximum of data from the sample


You should get this expected output : 

```bash
(base) pgb@pgbs-MacBook-Air beans_sentiment_analysis % python analyze_sentiment.py --token SOL --window 2400h --min-confidence 0.50
{
  "token": "SOL",
  "bullish": 1,
  "neutral": 0,
  "bearish": 1,
  "bullish_ratio": 0.5,
  "top_posts": [
    {
      "text": "$DOGE to the moon \ud83d\ude80 but $SOL is dumping hard.",
      "label": "bearish",
      "confidence": 0.55
    },
    {
      "text": "$SOL looks stable, nothing crazy.",
      "label": "bullish",
      "confidence": 0.5
    }
  ]
}
```
<br>

Example: Analyzing using the BERTweet model using SOLANA and a time windows of 240 hours:

`python analyze_sentiment_v2.py --token SOL --window 2400h --min-confidence 0.50`

<br>

Example: Analyzing using the BERTweet model using SOLANA and a time windows of 240 hours:

`python analyze_sentiment_v3.py --token SOL --window 2400h --min-confidence 0.50`

<br>

How to evaluate the Classifier Performance ?
--------------------------------------------

It's key to evaluate the type of classifier. in order to find out which model is "the best," we need to evaluate them on a labeled dataset and compute standard classification metrics. Here’s a plan to do just that using the optional training dataset.

Since the `synthetic_crypto_sentiment_training_1.5k.csv` file contains pre-labeled data, we can treat it as a test set to measure how well each model's predictions align with the provided ground truth. We'll run both models on this dataset and compare their outputs to the label column in the csv.

We'll use standard machine learning metrics to assess performance:

<b>Accuracy</b>: The percentage of posts the model classified correctly. While easy to understand, it can be misleading on imbalanced datasets.

<b>Precision</b>: Of all the posts the model labeled as "bullish," what percentage were actually "bullish"?

<b>Recall</b>: Of all the posts that were actually "bullish," what percentage did the model correctly identify?

<b>F1-Score</b>: The harmonic mean of precision and recall. It's a great single metric for a model's overall performance.

<b>Confusion Matrix</b>: A table that shows the number of correct and incorrect predictions for each sentiment category.

<br>

The compete evaluator code named `evaluate_models.py` can be found <a href="https://github.com/Pascal-Bernard/beans_sentiment_analysis/blob/main/evaluate_models.py">here</a>.

NOTe : We dont recommand to run again because of the API limitation that will make it crash or slow it down endlessly. I ran it after many splits in the process.


```python
RESULTS :

----------------- VADER Classification Report: ----------------

              precision    recall  f1-score   support

     bearish       0.65      0.40      0.50       495
     bullish       0.49      0.39      0.44       489
     neutral       0.38      0.59      0.47       516

    accuracy                           0.47      1500
   macro avg       0.51      0.46      0.47      1500
weighted avg       0.51      0.47      0.47      1500

VADER Confusion Matrix:
[[193 296   0]
 [ 99 307 110]
 [ 99 196 200]]

-------------- DistilBERT Classification Report: --------------

              precision    recall  f1-score   support

     bearish       0.35      0.82      0.50       495
     bullish       0.62      0.39      0.48       489
     neutral       0.21      0.02      0.03       516

    accuracy                           0.41      1500
   macro avg       0.40      0.41      0.34      1500
weighted avg       0.39      0.41      0.33      1500

DistilBERT Confusion Matrix:
[[191   0 298]
 [ 62   9 445]
 [ 54  33 408]]

--------------- BERTweet Classification Report: ---------------

BERTweet Classification Report:
              precision    recall  f1-score   support

     bearish       0.73      0.61      0.67       495
     bullish       0.73      0.44      0.55       489
     neutral       0.41      0.64      0.50       516

    accuracy                           0.56      1500
   macro avg       0.63      0.56      0.57      1500
weighted avg       0.62      0.56      0.57      1500

BERTweet Confusion Matrix:
[[213 276   0]
 [ 77 329 110]
 [  0 193 302]]
```

<br>

BREAKDOWN OF THE RESULTS & ANALYSIS
-----------------------------------

Based on the data returned by the evaluator we can conclude that :

The <b>BERTweet</b> model is the clear winner with an accuracy of 56%. VADER comes in second at 47%, and the DistilBERT model performed the worst with an accuracy of 41%.

<b>BERTweet:</b> This model shows the most balanced and effective performance. It has the highest precision and recall for both 'bearish' and 'bullish' sentiments, meaning it's the most reliable at correctly identifying these categories. The confusion matrix shows that while it does confuse some 'bearish' and 'bullish' posts, it handles the 'neutral' category much better than the other models.

<b>VADER:</b> VADER's performance is mediocre. Its overall accuracy and F1-score are significantly lower than BERTweet's. The confusion matrix reveals a major weakness: it struggles to distinguish between 'bearish' and 'bullish' sentiments, leading to a high number of false positives in both categories. It also misclassifies a lot of bullish and bearish posts as neutral.

<b>DistilBERT:</b> This model performed the poorest. Its extremely low recall (0.02) and F1-score (0.03) for the 'neutral' category are a significant red flag. The confusion matrix shows that the model is classifying a large number of 'bullish' and 'bearish' posts as 'neutral' (445 and 298, respectively), but it's not correctly identifying posts that are actually neutral. This skewed performance makes it an unreliable choice for this specific task.


Next step ?
-----------

Given the previous results that not bad - but not amazing either - its worth trying to build our own ML classifier.

1. Scikit learn and `TfidfVectorizer`

Using TfidfVectorizer from Scikit-learn is quite relevqnt because it allows us to create a model specifically tailored to the language in our dataset. Unlike pre-trained models that have a fixed vocabulary and may misinterpret crypto jargon, TF-IDF calculates the importance of each word based on its frequency within our specific collection of text. This process turns the raw text into a set of numerical features that highlight the most relevant terms, enabling a custom model to learn the unique patterns and nuances of the crypto community's language directly from our data.

You can find the code here : <a href="https://github.com/Pascal-Bernard/beans_sentiment_analysis/blob/main/custom_model_using_sklearn.py">here</a> and here is the result :

Training on 1200 samples, testing on 300 samples.
Training the Logistic Regression model...
Evaluating the model...

```python
----------------- Custom Model Classification Report: ----------------
              precision    recall  f1-score   support

     bearish       1.00      1.00      1.00        99
     bullish       1.00      1.00      1.00        98
     neutral       1.00      1.00      1.00       103

    accuracy                           1.00       300
   macro avg       1.00      1.00      1.00       300
weighted avg       1.00      1.00      1.00       300


----------------- Custom Model Confusion Matrix: ----------------
[[ 99   0   0]
 [  0  98   0]
 [  0   0 103]]
 ```

These results are a classic ML case of "good news, bad news". The good news is that the 100% accuracy and perfect confusion matrix show the model successfully learned the patterns in your synthetic data. However, this is also the bad news, as it is a strong sign of overfitting. The model has likely memorized the specific examples rather than learning how to generalize, meaning it would perform poorly on more varied, real-world text... 

Given the perfect - but therefore overfitted - the results from our first run, it's would relevant to try building another to confirm the overfittig. If by any luck we get a slightly different output it would provide valuable insight into the model's stability and its ability to generalize, or at least how badly it's memorizing the data. This will help us confirm if the overfitting is a one-time fluke or a consistent behavior of the model on this specific dataset.

NOTE : This second test is just for the sake of the test, because it is already quite clear why we have this overfitting..

2. PyTorch

PyTorch is an excellent choice for our 2nd model because it's a powerful deep learning framework that provides flexibility to build a custom neural net from the ground up. It allows us to have complete control over the model's architecture and training process, etc.. which is key for tackling the potential nuances of our data,  and moving beyond simple pattern memorization..


You can find the code here : <a href="https://github.com/Pascal-Bernard/beans_sentiment_analysis/blob/main/custom_model_using_torch.py">here</a> and here is the result :

```python
----------------- Custom PyTorch Model Classification Report: ----------------
              precision    recall  f1-score   support

     bullish       1.00      1.00      1.00        98
     neutral       1.00      1.00      1.00       103
     bearish       1.00      1.00      1.00        99

    accuracy                           1.00       300
   macro avg       1.00      1.00      1.00       300
weighted avg       1.00      1.00      1.00       300


----------------- Custom PyTorch Model Confusion Matrix: ----------------
[[ 98   0   0]
 [  0 103   0]
 [  0   0  99]]
 ```

CONCLUSION
----------

The flawless performance of our custom models, from both Scikit-learn and PyTorch, is a definitive indication of extreme overfitting. This outcome is not a sign of a perfect model, but rather a direct result of two obvious key factors:  
- First, the synthetic nature of the data means it lacks the complexity and unpredictability of real-world text, allowing the models to simply memorize the simple patterns rather than learning to generalize. 
- Second, the total number of data points is far too small to provide enough variation for a model to learn from, compounding the memorization problem. With these two combined limitations, it's impossible to build a performimg and generalizable model, and our perfect scores are the proof.




<br><br>


BONUS
-----

<b>Question : Once you have a working implementation, consider expanding it to work with real-time data using any available API for Reddit or X.</b>

Using the template we preivously built and make it work with real-time data is quite straight.

You can see the complete working code using Reddit API real-time data : <a href="https://github.com/Pascal-Bernard/beans_sentiment_analysis/blob/main/analyze_sentiment_reddit.py">here</a>

To run it, you can use this command : 

`python analyze_sentiment_reddit.py --token SOL --subreddit solana --limit 50`

NOTE: 
- Important to add the '--subreddit' for Reddit !
- The system can precisaly replicated with any other social API (X, telegram, news feed, etc..)


And the output should be as follows :


```python
(base) pgb@pgbs-MacBook-Air beans_sentiment_analysis % python analyze_sentiment_reddit.py --token SOL --subreddit solana --limit 50
Fetching posts from r/solana...
{
  "token": "SOL",
  "bullish": 20,
  "neutral": 0,
  "bearish": 1,
  "bullish_ratio": 0.95,
  "top_posts": [
    {
      "text": "It's not a draw. Use the Solana ecosystem while staked. New projects (protocols) then use things like activity and amoun...",
      "label": "bullish",
      "confidence": 1.0
    },
    {
      "text": "Depends on if the project dropping wants to include pSOL holders. Up to them. Generally you'll want to stick with the mo...",
      "label": "bullish",
      "confidence": 0.99
    },
    {
      "text": "No, that's phantom's LST and phantom charges way too much in fees. I would stick with native staking on good Jito based ...",
      "label": "bullish",
      "confidence": 0.99
    },
    {
      "text": "Solana Just Landed 4 Major Wins Today \u2014 Are We Watching It Turn Into a Global Crypto Backbone? Here\u2019s a quick breakdown ...",
      "label": "bullish",
      "confidence": 0.98
    },
    {
      "text": "**The Institutional Shift is Real \u2014 Here\u2019s Why This Momentum Could Actually Stick**\n\nGreat breakdown on these developmen...",
      "label": "bullish",
      "confidence": 0.98
    },
    {
      "text": "I simply look at the market and different financial indicators. We know the relevance of trad-fi price action and if we ...",
      "label": "bullish",
      "confidence": 0.97
    },
    {
      "text": "A paper wallet is really just a way of storing your private key offline. It doesn\u2019t have built-in functionality like a s...",
      "label": "bullish",
      "confidence": 0.96
    },
    ...
  ]
}
```

