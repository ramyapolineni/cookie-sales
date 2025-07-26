from sqlalchemy import create_engine
import pandas as pd
import os

def upload_to_render_db(df, table_name="final_cookie_sales_all_years"):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not set in environment variables")

    engine = create_engine(db_url)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",  # or "append" if you're inserting year-by-year
        index=False
    )
    print(f"✅ Uploaded to table `{table_name}` in Render DB")
