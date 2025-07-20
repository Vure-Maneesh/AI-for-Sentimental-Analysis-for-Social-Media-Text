import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import numpy as np
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import plot
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Offline mode for Plotly and Cufflinks
cf.go_offline()

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Preprocess review text
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub("[^a-zA-Z ]", '', text)
        text = text.lower().strip()
        return text
    else:
        return ''

# Sentiment analysis using VADER
def vader_sentiment_analysis(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.7:
        return 'Very Positive'
    elif compound >= 0.3:
        return 'Positive'
    elif compound > -0.3:
        return 'Neutral'
    elif compound > -0.7:
        return 'Negative'
    else:
        return 'Very Negative'

# Aspect-based sentiment analysis
def aspect_sentiment_analysis(text):
    aspects = ['quality', 'price', 'delivery', 'customer service']
    aspect_sentiments = {aspect: [] for aspect in aspects}
    
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    
    for aspect in aspects:
        if aspect in tokens:
            aspect_indices = [i for i, token in enumerate(tokens) if token == aspect]
            for idx in aspect_indices:
                if idx > 0 and idx < len(tokens) - 1:
                    left_word = tokens[idx - 1]
                    right_word = tokens[idx + 1]
                    aspect_sentiment = None
                    if left_word in ['good', 'excellent']:
                        aspect_sentiment = 'Positive'
                    elif left_word in ['poor', 'bad']:
                        aspect_sentiment = 'Negative'
                    elif right_word in ['good', 'excellent']:
                        aspect_sentiment = 'Positive'
                    elif right_word in ['poor', 'bad']:
                        aspect_sentiment = 'Negative'
                    if aspect_sentiment:
                        aspect_sentiments[aspect].append(aspect_sentiment)
    
    return aspect_sentiments

# Fake review detection for a single review
def detect_fake_review_single(review_text):
    review_length = len(review_text.split())
    positive_words = ['excellent', 'awesome', 'fantastic', 'amazing', 'great', 'love']
    overly_positive = any(word in review_text for word in positive_words)
    short_review = review_length < 5
    highly_generic = len(set(review_text.split())) < 3
    return overly_positive or short_review or highly_generic

# Apply fake review detection for dataset
def detect_fake_reviews_enhanced(df):
    df['review_length'] = df['reviewText'].apply(lambda x: len(x.split()))
    positive_words = ['excellent', 'awesome', 'fantastic', 'amazing', 'great', 'love']
    df['overly_positive'] = df['reviewText'].apply(lambda x: any(word in x for word in positive_words))
    df['short_review'] = df['review_length'] < 5
    df['highly_generic'] = df['reviewText'].apply(lambda x: len(set(x.split())) < 3)
    df['potential_fake'] = np.where(
        df['overly_positive'] | df['short_review'] | df['highly_generic'],
        True,
        False
    )
    return df

# Plot aspect sentiments
def plot_aspect_sentiments(df):
    aspects = ['quality', 'price', 'delivery', 'customer service']
    aspect_counts = {aspect: {'Positive': 0, 'Negative': 0} for aspect in aspects}
    
    for aspect_sentiments in df['aspect_sentiments']:
        for aspect, sentiments in aspect_sentiments.items():
            for sentiment in sentiments:
                aspect_counts[aspect][sentiment] += 1
    
    aspect_df = pd.DataFrame(aspect_counts)
    aspect_df.plot(kind='bar', stacked=True)
    plt.title('Aspect-based Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot potential fake reviews
def plot_potential_fake_reviews(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='potential_fake', data=df)
    plt.title('Potential Fake Reviews')
    plt.xlabel('Is Potential Fake')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Plot VADER sentiment distribution using Plotly
def categorical_variable_summary(df, column_name):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Countplot', 'Percentage'),
                        specs=[[{"type": "xy"}, {'type': 'domain'}]])
    
    fig.add_trace(go.Bar(
        y=df[column_name].value_counts().values.tolist(),
        x=[str(i) for i in df[column_name].value_counts().index],
        text=df[column_name].value_counts().values.tolist(),
        textfont=dict(size=14),
        name=column_name,
        textposition='auto',
        showlegend=False,
        marker=dict(color=['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9'],
                    line=dict(color='#DBE6EC', width=1))),
        row=1, col=1)

    fig.add_trace(go.Pie(
        labels=df[column_name].value_counts().keys(),
        values=df[column_name].value_counts().values,
        textfont=dict(size=18),
        textposition='auto',
        showlegend=False,
        name=column_name,
        marker=dict(colors=['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9'])),
        row=1, col=2)

    fig.update_layout(title={'text': column_name,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')
    
    plot(fig, filename=f'{column_name}_summary.html', auto_open=True)

# Process dataset and visualize
def analyze_dataset(df):
    df['reviewText'] = df['reviewText'].map(preprocess_text)
    df['vader_sentiment'] = df['reviewText'].apply(vader_sentiment_analysis)
    df['aspect_sentiments'] = df['reviewText'].apply(aspect_sentiment_analysis)
    df = detect_fake_reviews_enhanced(df)
    categorical_variable_summary(df, 'vader_sentiment')
    plot_aspect_sentiments(df)
    plot_potential_fake_reviews(df)

# Analyze a single review
def analyze_single_review(review_text):
    review_text_processed = preprocess_text(review_text)
    vader_sentiment = vader_sentiment_analysis(review_text_processed)
    aspect_sentiments = aspect_sentiment_analysis(review_text_processed)
    potential_fake = detect_fake_review_single(review_text_processed)
    
    print("\n*** Text Analysis Results ***")
    print(f"Text: {review_text}")
    print(f"VADER Sentiment: {vader_sentiment}")
    print(f"Potential Review: {'Yes' if potential_fake else 'No'}")

# Input loop for live review analysis
def input_review_data():
    while True:
        review_text = input("\nEnter a text review (or type 'exit' to quit): ")
        if review_text.lower() == 'exit':
            break
        analyze_single_review(review_text)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Load your dataset here
    df = pd.read_csv("C:/Users/Abhi/text.csv")

    # Run dataset analysis
    analyze_dataset(df)

    # Start real-time review analysis
    input_review_data()
