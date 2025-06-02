from newsScrape.Aggregator import Aggregator
import time
import datetime
import os
from colorama import init, Fore
from newsAnalyze.findNames import extract_tickers
from newsAnalyze.sentiment import sentiment_given_tickers, analyze_sentiment
from newsScrape.TestWebsite import run_tests

init(autoreset=True)

x = Aggregator()

x.createThreads()

def pretty_print(tickers: list, url: str, ticker_sentiment: dict, whole_sentiment: float):
    while len(tickers) != 3:
        tickers.append('   ')
    
    primaryTickerSentiment = ticker_sentiment[tickers[0]]

    if primaryTickerSentiment >= 0:
        primaryTickerSentiment = f"{primaryTickerSentiment:.5f}"
    else:
        primaryTickerSentiment = f"{primaryTickerSentiment:.4f}"

    wholeSentiment = whole_sentiment
    if wholeSentiment >= 0:
        wholeSentiment = f"{wholeSentiment:.5f}"
    else:
        wholeSentiment = f"{wholeSentiment:.5f}"

    sentiment_value = float(primaryTickerSentiment)
    color = Fore.LIGHTCYAN_EX if abs(0 - sentiment_value) < 0.1 else (Fore.LIGHTRED_EX if sentiment_value <= 0 else Fore.LIGHTGREEN_EX)

    print(f"{datetime.datetime.now()} {color}{tickers[0]}{Fore.RESET}"
          #f"\t - {Fore.LIGHTMAGENTA_EX}{sentiment_value:.5f}{Fore.RESET}"
          f"- {Fore.LIGHTYELLOW_EX}{tickers}{Fore.RESET} - {url}")

while True:
    for _ in range(len(x.articles)):
        article = x.popArticle()
        
        if not article.content or len(article.content) < 100:
            print(f"{datetime.datetime.now()} - {article.url}")
            continue
                
        tickers = extract_tickers(article.content)
        
        if len(tickers) == 0:
            continue
        
        ticker_sentiment = sentiment_given_tickers(article.content, list(tickers.keys()))
        whole_sentiment = analyze_sentiment(article.content)

        pretty_print(list(tickers.keys())[:3], article.url, ticker_sentiment, whole_sentiment)
    
    time.sleep(1)
