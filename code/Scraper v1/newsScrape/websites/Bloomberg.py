
import json
from ..AbstractWebsite import Website
from ..Article import Article


class Bloomberg(Website):
    def __init__(self):
        super().__init__()
        self.baseURL = "https://www.bloomberg.com"
        self.websiteName = "Bloomberg"

    def getRecentArticles(self):
        soup = self.fetch_page("https://www.bloomberg.com/economics")
        return self.parse_articles(soup)

    def parse_articles(self, soup):
        scriptData = soup.find('script', {"id": "__NEXT_DATA__"})
        new_articles = []
 
        scriptData = json.loads(scriptData.text)

        stories = scriptData['props']['pageProps']['initialState']['modulesById']['top_stories']['items']
        
        for story in stories:
            
            article_url = self.baseURL + story['url']
            article_title = story['headline']
            article_author = story['byline']
                
            if article_url not in self.seen_articles:
                new_articles.append(Article(
                    article_url,
                    article_title,
                    self.websiteName,
                    article_author
                ))
                self.seen_articles.add(article_url)

        return new_articles
