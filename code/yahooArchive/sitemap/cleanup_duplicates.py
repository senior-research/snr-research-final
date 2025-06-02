from sitemap.conn import sitemap_session
from sitemap.models import DaysTable
from sqlalchemy import func

CHUNK_SIZE = 100 

def cleanup_duplicates(session):
    subquery = (
        session.query(
            DaysTable.month_id,
            DaysTable.day,
            func.min(DaysTable.id).label('min_id')
        )
        .group_by(DaysTable.month_id, DaysTable.day)
        .subquery()
    )

    while True:
        duplicates = (
            session.query(DaysTable.id)
            .join(subquery, (DaysTable.month_id == subquery.c.month_id) & (DaysTable.day == subquery.c.day))
            .filter(DaysTable.id != subquery.c.min_id)
            .limit(CHUNK_SIZE)
            .all()
        )

        if not duplicates:
            break

        duplicate_ids = [duplicate.id for duplicate in duplicates]
        session.query(DaysTable).filter(DaysTable.id.in_(duplicate_ids)).delete(synchronize_session=False)
        session.commit()
        print(f"Processed {len(duplicates)} duplicates.")

if __name__ == "__main__":
    cleanup_duplicates(sitemap_session)
    print("Duplicate days cleaned up.")
