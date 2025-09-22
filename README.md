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