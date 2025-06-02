from abc import ABC, abstractmethod
import tls_client
from bs4 import BeautifulSoup
from .Article import Article

class Website(ABC):
    def __init__(self):
        self.session = tls_client.Session(
            client_identifier="chrome112",
            random_tls_extension_order=True,
        )
        self.baseURL = ""
        self.websiteName = ""
        self.seen_articles = set()

    @abstractmethod
    def getRecentArticles(self):
        pass

    def fetch_page(self, url):
        res = self.session.get(url)
        #with open('test.html', 'w') as f:
        #    f.write(res.text)
        return BeautifulSoup(res.content, 'html.parser')
