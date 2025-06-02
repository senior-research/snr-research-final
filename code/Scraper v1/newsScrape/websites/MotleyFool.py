import tls_client
from bs4 import BeautifulSoup
from ..AbstractWebsite import Website
from ..Article import Article

class MotleyFool(Website):
    def __init__(self):
        super().__init__()
        self.baseURL = "https://www.fool.com"
        self.websiteName = "The Motley Fool"

    def getRecentArticles(self):
        soup = self.fetch_page("https://www.fool.com/investing-news/")
        yield from self.parse_articles(soup)

    def parse_articles(self, soup):
        divs = soup.find_all('div', {"class": "flex py-12px text-gray-1100"})
        
        for div in divs:
            try:
                article_url = div.a['href']
                if not article_url.startswith("http"):
                    article_url = self.baseURL + article_url
                
                if 'the-ascent' in article_url: 
                    continue 

                article_title = div.find('h5').get_text(strip=True) if div.find('h5') else "?"
                article_author = div.div.a.get_text(strip=True) if div.div.a else "?"

                article_content = self.fetch_article_content(article_url)

                if article_url not in self.seen_articles:
                    yield Article(
                        article_url,
                        article_title,
                        self.websiteName,
                        article_author,
                        content=article_content,
                    )
                    self.seen_articles.add(article_url)

            except AttributeError:
                continue

    def fetch_article_content(self, article_url):
        article_soup = self.fetch_page(article_url)

        main_content = article_soup.find('div', {'class': 'article-body'})
        if not main_content:
            return "Content not found"

        unwanted_sections = main_content.find_all(['footer', 'aside', 'div'], 
                                                  class_=['disclaimer', 'footer', 'promo', 'ad'])
        for section in unwanted_sections:
            section.decompose()

        paragraphs = main_content.find_all('p')
        content = "\n".join([p.get_text(strip=True) for p in paragraphs])

        return content
