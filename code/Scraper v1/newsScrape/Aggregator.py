import threading
import time

from .Article import Article
from .AbstractWebsite import Website
from .websites.Bloomberg import Bloomberg
from .websites.ForbesMoney import Forbes
from .websites.MotleyFool import MotleyFool
from .websites.YahooNews import YahooNews

class Aggregator:
    def __init__(self):
        self.websites: list[Website] = [
            #Bloomberg(),
            #Forbes(),
            MotleyFool(),
            #YahooNews(),
        ]

        self.articles: list[Article] = []
        self.maxThreads = 5
        self.sleepTime = 60

        self.debug = False

        self.article_lock = threading.Lock()
    
    def popArticle(self):
        if len(self.articles) > 0:
            return self.articles.pop(0)
        return None
    
    def newArticles(self) -> bool:
        return len(self.articles) > 0
    
    def fetchArticlesFromWebsite(self, website: Website):
        while True:
            if self.debug:
                print(f"Fetching articles from {website.websiteName}")
            a = time.perf_counter()

            for article in website.getRecentArticles():
                with self.article_lock:
                    self.articles.append(article)
                    if self.debug:
                        print(f"New article added: {article.title}")
            
            if self.debug:
                print(f"{website.websiteName} took {time.perf_counter() - a} seconds")
                print(f"{website.websiteName}: Sleeping for {self.sleepTime} seconds...")
            time.sleep(self.sleepTime)

    def createThreads(self):
        for website in self.websites:
            thread = threading.Thread(target=self.fetchArticlesFromWebsite, args=(website,))
            thread.daemon = True 
            thread.start()

        if self.debug:
            print(f"Started {len(self.websites)} threads for fetching articles.")
