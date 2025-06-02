
from colorama import Fore, Back, Style
from newsAnalyze.findNames import extract_company_names, extract_tickers
from newsAnalyze.sentiment import sentiment_given_tickers, analyze_sentiment
from newsScrape.Article import Article
from newsScrape.websites.MotleyFool import MotleyFool
from newsScrape.websites.YahooNews import YahooNews


def run_tests():
    content_test()

def content_test():
    website = YahooNews()

    x = website.getRecentArticles()

    print(x)
    
def sentiment_test():
    website = YahooNews()

    x: list[Article] = website.getRecentArticles()

    for y in x:
        if len(y.content) < 100: continue

        z = extract_tickers(y.content)

        if len(list(z.keys())) == 0:
            continue
    
        pretty_print(list(z.keys())[:3], y.url, sentiment_given_tickers(y.content, list(z.keys())), analyze_sentiment(y.content))

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
        wholeSentiment = f"{wholeSentiment:10f}"
    else:
        wholeSentiment = f"{wholeSentiment:9f}"

    sentiment_value = float(primaryTickerSentiment)
    color = Fore.LIGHTCYAN_EX if abs(0 - sentiment_value) < 0.1 else (Fore.LIGHTRED_EX if sentiment_value <= 0 else Fore.LIGHTGREEN_EX)

    print(f"{color}{tickers[0]}{Fore.RESET}"
      f"\t - {Fore.LIGHTMAGENTA_EX}{sentiment_value:.5f}{Fore.RESET} - {Fore.LIGHTYELLOW_EX}{wholeSentiment}{Fore.RESET} - {url}")