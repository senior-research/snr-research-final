class Article:
    def __init__(self, url, title, website, author, content=None, premium=False):
        self.url = url
        self.title = title
        self.website = website
        self.author = author
        self.content = content
        self.premium = premium

    def __str__(self):
        return self.url + ' - ' + self.title + ' - ' + self.website + ' - ' + self.author