Crypto Sentiment Analysis Tool
------------------------------

### Project Objective

This project aims to create a command-line tool in **Python** to analyze sentiment in crypto-related social media posts. The tool leverages Python's extensive ecosystem for natural language processing (NLP) and data analysis to provide actionable insights into market sentiment.

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


#### How to Run the Script
--------------------------

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

Example: Analyzing Bitcoin (BTC) using the VADER model using SOL (Solane and a toie windows of 240 hours)

`python analyze_sentiment.py --token SOL --window 2400h --min-confidence 0.50`

NOTE : the default value of 0.75 can be too high for some models to provide an output. That's why I suggest to set at 0.50