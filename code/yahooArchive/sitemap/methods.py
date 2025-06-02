from .models import *

def add_year(session, year_value):
    existing_year = session.query(YearsTable).filter_by(year=year_value).first()
    if existing_year:
        return existing_year
    year = YearsTable(year=year_value, is_processed=False, processed_percent=0.0)
    session.add(year)
    session.commit()
    return year

def add_month_with_url(session, year_id, month_value, month_url):
    existing_month = session.query(MonthsTable).filter_by(year_id=year_id, month=month_value).first()
    if existing_month:
        return existing_month
    year = session.query(YearsTable).filter_by(id=year_id).first()
    if not year:
        raise ValueError(f"Year with id {year_id} does not exist.")
    month = MonthsTable(month=month_value, year_id=year_id, url=month_url)
    session.add(month)
    session.commit()
    return month

def add_day_with_url(session, month_id, day_value, day_url):
    existing_day = session.query(DaysTable).filter_by(month_id=month_id, day=day_value).first()
    if existing_day:
        if existing_day.url != day_url:
            existing_day.url = day_url
            session.commit()
        return existing_day
    month = session.query(MonthsTable).filter_by(id=month_id).first()
    if not month:
        raise ValueError(f"Month with id {month_id} does not exist.")
    day = DaysTable(day=day_value, month_id=month_id, url=day_url)
    session.add(day)
    session.commit()
    return day

def set_urls_added_for_day(session, day_id):
    day = session.query(DaysTable).filter_by(id=day_id).first()
    if not day:
        raise ValueError(f"Day with id {day_id} does not exist.")
    day.added_urls = True
    session.commit()
    return day

def set_days_added(session, month_id):
    month = session.query(MonthsTable).filter_by(id=month_id).first()
    if not month:
        raise ValueError(f"Month with id {month_id} does not exist.")
    month.added_days = True
    session.commit()
    return month

def get_month_id_from_url(session, month_url):
    month = session.query(MonthsTable).filter_by(url=month_url).first()
    if not month:
        raise ValueError(f"Month with url {month_url} does not exist.")
    return month.id

def get_month_from_id(session, month_id):
    month = session.query(MonthsTable).filter_by(id=month_id).first()
    if not month:
        raise ValueError(f"Month with id {month_id} does not exist.")
    return month

def get_days_without_urls(session):
    return session.query(DaysTable).filter_by(added_urls=False).all()

def add_day_with_url_collection(session, month_id, day_value, day_url, url_collection):
    existing_day = session.query(DaysTable).filter_by(month_id=month_id, day=day_value).first()
    if existing_day:
        if existing_day.url != day_url:
            existing_day.url = day_url
            session.commit()
        day = existing_day
    else:
        month = session.query(MonthsTable).filter_by(id=month_id).first()
        if not month:
            raise ValueError(f"Month with id {month_id} does not exist.")
        day = DaysTable(day=day_value, month_id=month_id, url=day_url)
        session.add(day)
        session.flush()
    for url_value in url_collection:
        if not url_value:
            continue
        url_entry = UrlsTable(url=url_value, day_id=day.id)
        session.add(url_entry)
    session.commit()
    return day

def get_months_without_days(session):
    return session.query(MonthsTable).filter_by(added_days=False).all()

def delete_all_urls(session):
    session.query(UrlsTable).delete()
    session.commit()

def reset_added_urls_for_all_days(session):
    session.query(DaysTable).update({DaysTable.added_urls: False})
    session.commit()

