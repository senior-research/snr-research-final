import json
from ..AbstractWebsite import Website
from ..Article import Article
import re

compiled_re = re.compile("^morpheusGridBody.*")

json_data = {}
params = {}

class YahooNews(Website):
    def __init__(self):
        super().__init__()
        self.baseURL = "https://finance.yahoo.com"
        self.websiteName = "YahooNews"

    def getRecentArticles(self):
        response = self.session.post(
            "https://finance.yahoo.com/_finance_doubledown/api/resource",
            params=params,
            json=json_data,
        )

        resp = json.loads(response.text)

        return resp

    def parse_articles(self, articles):
        new_articles = []

        for article in articles:
            article_url = article["link"]
            article_author = article["publisher"]
            article_title = article["title"]
            isPremium = bool(article["finance"]["premiumFinance"]["isPremiumNews"])

            article_content = self.fetch_article_content(article_url)

            if article_url not in self.seen_articles:
                new_articles.append(
                    Article(
                        article_url,
                        article_title,
                        self.websiteName,
                        article_author,
                        article_content,
                        isPremium,
                    )
                )
                self.seen_articles.add(article_url)

        return new_articles

    def fetch_article_content(self, article_url):
        article_soup = self.fetch_page(article_url)

        main_content = article_soup.find("div", {"class": compiled_re})
        if not main_content:
            return "Content not found"

        unwanted_sections = main_content.find_all(
            ["footer", "aside", "div"], class_=["disclaimer", "footer", "promo", "ad"]
        )
        for section in unwanted_sections:
            section.decompose()

        paragraphs = main_content.find_all("p")

        content = "\n".join([p.get_text(strip=True) for p in paragraphs])

        return content
