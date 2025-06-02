from sitemap.tasks import populate_sitemap_db
from sitemap.cleanup_duplicates import cleanup_duplicates
from sitemap.methods import delete_all_urls, reset_added_urls_for_all_days
import logging
from sitemap.conn import sitemap_session

logging.basicConfig(level=logging.ERROR)

