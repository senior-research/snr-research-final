import time
from sitemap.methods import *
from sitemap.conn import sitemap_session
from sitemap.api import SiteMap
import datetime

def add_years_and_months_to_database(session, sitemap_api: SiteMap):
    years = sitemap_api.get_months_and_years()

    for year in years:
        db_year = add_year(
            session, 
            year, 
        )
        
        for month in years[year]:
            add_month_with_url(
                session,
                db_year.id,
                month, 
                years[year][month],
            )


sleep_time = 10

def add_days_to_database(session, month_url, site_map: SiteMap):
    try:
        days = site_map.get_month_days(month_url)
        month = get_month_from_id(session, get_month_id_from_url(session, month_url))

        print(month.year_id, month.id)

        for day in days:
            print(day, end=" ")

            add_day_with_url(
                session,
                month.id,
                day,
                days[day],
            )

        else:
            set_days_added(session, month.id)

        print()
    except:
        
        print(f"Error adding days to database, sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        return False

def add_urls_to_day(session, day, site_map: SiteMap):
    starting_time = time.perf_counter()

    try:
        links = site_map.get_links_from_day(day.url)
    
        add_day_with_url_collection(
            session,
            day.month_id,
            day.day,
            day.url,
            links
        )
    except Exception as e:
        print(f"Error: {e}")
        return


    print(f"Processed {day.url} with {len(links)} links | Time: {time.perf_counter() - starting_time:.2f}s")

    set_urls_added_for_day(session, day.id)

months_to_int = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def populate_sitemap_db():
    sitemap = SiteMap()

    add_years_and_months_to_database(sitemap_session, sitemap)

    while len(months := get_months_without_days(sitemap_session)) > 0:
        for q_month in months:
            add_days_to_database(sitemap_session, q_month.url, sitemap)
    
    while len(unprocessed_days := get_days_without_urls(sitemap_session)) > 0:
        with open("unprocessed_days.txt", "w") as f:
            for day in unprocessed_days:
                f.write(day.url + "\n")

        current = datetime.datetime.date(datetime.datetime.now())


        for day in unprocessed_days:
            if (current.year < day.month.year.year or 
                (current.year == day.month.year.year and current.month < months_to_int.get(day.month.month)) or 
                (current.year == day.month.year.year and current.month == months_to_int[day.month.month] and current.day < day.day)):
                print(f"Skipping {day.url} as it is in the future...")
    
            add_urls_to_day(sitemap_session, day, sitemap)
    
    print("Sitemap database is fully populated...")