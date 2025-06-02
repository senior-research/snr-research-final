from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

sitemap_engine = create_engine('sqlite:///sentiment.db')
Base.metadata.create_all(sitemap_engine)
Session = sessionmaker(bind=sitemap_engine)
sitemap_session = Session()
