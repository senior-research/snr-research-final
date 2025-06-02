import random
import tls_client
from bs4 import *
from lxml import etree
import re

overallDivRegex = re.compile("sitemapindex*.")
dayDivRegex     = re.compile("sitemapcontent*.")

class SiteMap():
    def __init__(self, debug=False):
        self.baseUrl = "https://finance.yahoo.com/sitemap/"
        self.session = tls_client.Session(
            random.choice([
                #'okhttp4_android_13',
                #'firefox_120',
                'safari_16_0',
                #'safari_ios_16_0',
            ])
        )

        self.debug = debug
        
        self.years_and_months = self.get_months_and_years()

    def get_months_and_years(self):
        resp = self.session.get(self.baseUrl)
        
        if self.debug:
            with open("type.html", "w") as f:
                f.write(resp.text)
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        yearListDiv = soup.find('div', attrs={'class': overallDivRegex})
        yearsList = yearListDiv.ul
        
        years = {}
        
        for year in yearsList:           
            months = {}
            
            for month in year.ul:
                months[month.a.text] = month.a.attrs.get('href')
            
            years[year.h4.text] = months

        return years
    
    def get_month_days(self, month_url):
        resp = self.session.get(month_url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        days = {}
        
        dayEnum = soup.find('div', class_='D(ib)')
        for day in (dayEnum.children):
            days[day.text] = day.attrs.get('href')
            
        return days

    def get_links_from_day(self, day_url):
        links = []
        
        for _ in range(500): # should never get this high 
            linkList = None
            
            try:
                for __ in range(3):
                    linkList, nextUrl = (self.get_links_from_day_state(day_url))
                    if linkList:
                        break
            except:
                continue
            
            links.extend(linkList)
            
            if nextUrl is None or len(linkList) < 50:
                break
            
            day_url = nextUrl
        
        return links
            
    def get_links_from_day_state(self, day_url):
        resp = self.session.get(day_url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        stories = []

        if "No Results found during this period" in resp.text:
            return [], None
        
        div = soup.find('div', attrs={'class': dayDivRegex})
        for story in div.ul:
            stories.append(story.a.attrs.get('href'))
        
        nextDiv = soup.find('div', class_='Py(20px) Fz(15px)')
        if "Next" in nextDiv.text:
            nextButton = nextDiv.find('a', string='Next')
            return stories, nextButton.attrs.get('href')
        
        return stories, None
    
