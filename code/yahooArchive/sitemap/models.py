from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Float, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class YearsTable(Base):
    __tablename__ = 'years'
    id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    is_processed = Column(Boolean, default=False)
    processed_percent = Column(Float, default=0.0)

    months = relationship('MonthsTable', back_populates='year', cascade='all, delete-orphan')

class MonthsTable(Base):
    __tablename__ = 'months'
    id = Column(Integer, primary_key=True)
    month = Column(Integer, nullable=False)
    year_id = Column(Integer, ForeignKey('years.id'), nullable=False)
    url = Column(String, nullable=False)
    processed_percent = Column(Float, default=0.0)
    added_days = Column(Boolean, default=False)

    year = relationship('YearsTable', back_populates='months')
    days = relationship('DaysTable', back_populates='month', cascade='all, delete-orphan')

class DaysTable(Base):
    __tablename__ = 'days'
    id = Column(Integer, primary_key=True)
    day = Column(Integer, nullable=False)
    month_id = Column(Integer, ForeignKey('months.id'), nullable=False)
    url = Column(String, nullable=False)
    processed_percent = Column(Float, default=0.0)
    added_urls = Column(Boolean, default=False)
    
    month = relationship('MonthsTable', back_populates='days')
    urls = relationship('UrlsTable', back_populates='day', cascade='all, delete-orphan')

class UrlsTable(Base):
    __tablename__ = 'urls'
    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False)
    day_id = Column(Integer, ForeignKey('days.id'), nullable=False)
    is_processed = Column(Boolean, default=False)
    
    day = relationship('DaysTable', back_populates='urls')
