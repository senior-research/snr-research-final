
import re
from collections import defaultdict
from textblob import TextBlob
import nltk

def analyze_sentiment(article_text):
    blob = TextBlob(article_text)

    return blob.sentiment.polarity


def sentiment_given_tickers(article_text, tickers):
    sentiments = {}
    blob = TextBlob(article_text)
    for ticker in tickers:
        sentences = [sentence for sentence in blob.sentences if ticker in sentence.string]
        if sentences:
            sentiments[ticker] = sum(sentence.sentiment.polarity for sentence in sentences) / len(sentences)
        else:
            sentiments[ticker] = None

    return sentiments
