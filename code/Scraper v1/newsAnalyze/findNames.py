from collections import defaultdict
import spacy
import re

nlp = spacy.load("en_core_web_sm")

exclusions = {
    'IPO', 'ETF', 'AI', 'S&P', 'GPU', 'COVID', 'EPS', 'ET', 'CEO', 'USA', 'LLC', 'PLC', 'SPAC', 'NASA', 'US',
    'AARP', 'AP', 'UK', 'EV', 'FDA', 'MAX', 'NYC', 'GEICO'
}

def is_valid_company(company):
    return not bool(re.search(r"\b(Inc\.|Corp\.|Ltd\.|ETF|IPO|AI)\b", company)) and company not in exclusions

def clean_company_name(name):
    cleaned_name = re.sub(r'[^a-zA-Z\s]', '', name)
    cleaned_name = cleaned_name.strip()
    return cleaned_name

def extract_company_names(article_text):
    doc = nlp(article_text)
    companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    filtered_companies = [clean_company_name(company) for company in companies if is_valid_company(company)]

    company_sorted = defaultdict(int)
    for company in filtered_companies:
        if company:
            company_sorted[company] += 1
            
    return company_sorted

def extract_tickers(article_text):
    ticker_pattern = re.compile(r'\b[A-Z]{2,5}(?:-[A-Z])?(?=\d|\b)')

    tickers = re.findall(ticker_pattern, article_text)

    tickers = [x for x in list(tickers) if x not in exclusions]

    ticker_sorted = defaultdict(int)
    for ticker in tickers:
        ticker_sorted[ticker] += 1

    return ticker_sorted


