
from ..AbstractWebsite import Website
from ..Article import Article


class Forbes(Website):
    def __init__(self):
        super().__init__()
        self.baseURL = "https://www.forbes.com"
        self.websiteName = "Forbes (Money)"

    def getRecentArticles(self):
        soup = self.fetch_page("https://www.forbes.com/money/")
        return self.parse_articles(soup)

    def parse_articles(self, soup):
        divs = soup.find_all('div', {"data-test-e2e": "card stream"})
        new_articles = []

        for div in divs:
            article_url = div.a['href']
            x = div.findAll('a', {"data-ga-track": True})
            article_title = x[0].text
            article_author = x[1].text

            if article_url not in self.seen_articles:
                new_articles.append(Article(
                    article_url,
                    article_title,
                    self.websiteName,
                    article_author
                ))
                self.seen_articles.add(article_url)
            
        return new_articles