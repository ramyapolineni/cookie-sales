import pandas as pd
import os
from sqlalchemy import create_engine
from gspread import service_account
import gspread_dataframe as gd

MAX_SHARE_COOKIES = 5

def fetch_and_upload_cookie_mapping():
    # Load service account credentials
    gc = service_account(filename="service_account.json")

    # Open spreadsheet
    print("🔍 Opening 'cookie_mapping' Google Sheet...")
    spreadsheet = gc.open("cookie_mapping")

    # Read sheets
    print("📥 Reading tabs from Google Sheet...")
    transitions_ws = spreadsheet.worksheet("cookie_transitions")
    active_ws = spreadsheet.worksheet("active_cookies")

    transitions_df = gd.get_as_dataframe(transitions_ws).dropna(how='all')
    active_df = gd.get_as_dataframe(active_ws).dropna(how='all')

    # Clean headers
    transitions_df.columns = [col.strip() for col in transitions_df.columns]
    active_df.columns = [col.strip() for col in active_df.columns]

    # === Transform transitions_df ===
    def explode_share_columns(row):
        takes_from = str(row.get("Takes Share From (Other Cookies)", "")).split(",")
        share_pct = str(row.get("Share % Taken", "")).replace("%", "").split(",")

        takes_from = [t.strip() for t in takes_from if t.strip()]
        share_pct = [p.strip() for p in share_pct if p.strip()]

        for i in range(MAX_SHARE_COOKIES):
            row[f"ShareFrom_{i+1}"] = takes_from[i] if i < len(takes_from) else None
            row[f"SharePct_{i+1}"] = float(share_pct[i]) if i < len(share_pct) else None

        return row

    transitions_df = transitions_df.apply(explode_share_columns, axis=1)
    transitions_df.drop(columns=["Takes Share From (Other Cookies)", "Share % Taken"], inplace=True)

    # Upload to DB
    db_url = os.getenv("RENDER_DATABASE_URL")
    if not db_url:
        raise ValueError("RENDER_DATABASE_URL not set")

    engine = create_engine(db_url)
    transitions_df.to_sql("cookie_transitions", con=engine, if_exists="replace", index=False)
    active_df.to_sql("active_cookies", con=engine, if_exists="replace", index=False)

    print("✅ Uploaded to Render DB: cookie_transitions and active_cookies")

if __name__ == "__main__":
    fetch_and_upload_cookie_mapping()
