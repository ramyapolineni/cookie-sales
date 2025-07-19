import pandas as pd
import os
from sqlalchemy import create_engine
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

MAX_SHARE_COOKIES = 5  # max number of share-from cookies to support

def fetch_and_upload_cookie_mapping():
    # Authenticate using service account
    gauth = GoogleAuth()
    gauth.settings['client_config_backend'] = 'service'
    gauth.settings['service_config'] = {
        "client_json_file_path": "service_account.json",
        "client_user_email": "cookie-drive-bot@cookie-sales-463514.iam.gserviceaccount.com",
        "client_id": "",
        "client_secret": ""
    }
    gauth.ServiceAuth()
    drive = GoogleDrive(gauth)

    # Google Drive folder and file
    FOLDER_ID = "1e_3Y1LEuvqCicUG64G4QOkX1yHMEOvem"
    TARGET_FILENAME = "cookie_mapping"

    # Step 1: Find the file
    print("🔍 Searching for cookie_mapping Google Sheet...")
    file_list = drive.ListFile({
        'q': f"'{FOLDER_ID}' in parents and trashed=false and mimeType='application/vnd.google-apps.spreadsheet'"
    }).GetList()

    file_id = None
    for file in file_list:
        if file['title'].lower() == TARGET_FILENAME.lower():
            file_id = file['id']
            print(f"✅ Found file: {file['title']}")
            break

    if not file_id:
        raise FileNotFoundError("❌ cookie_mapping file not found in Google Drive folder")

    # Step 2: Download as Excel
    print("⬇️ Downloading as Excel...")
    sheet_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
    excel_path = "data/cookie_mapping.xlsx"
    os.makedirs("data", exist_ok=True)
    response_code = os.system(f"curl -L '{sheet_url}' --output {excel_path}")
    if response_code != 0:
        raise Exception("Failed to download the Excel file. Check permissions or URL.")

    # Step 3: Read sheets
    print("📥 Reading sheets...")
    transitions_df = pd.read_excel(excel_path, sheet_name="cookie_transitions", engine="openpyxl")
    active_df = pd.read_excel(excel_path, sheet_name="active_cookies", engine="openpyxl")

    # === Clean column names ===
    transitions_df.columns = [col.strip().replace('\n', ' ') for col in transitions_df.columns]
    active_df.columns = [col.strip().replace('\n', ' ') for col in active_df.columns]

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

    # Step 4: Upload to DB
    db_url = os.getenv("RENDER_DATABASE_URL")
    if not db_url:
        raise ValueError("RENDER_DATABASE_URL not set in environment variables")

    engine = create_engine(db_url)
    transitions_df.to_sql("cookie_transitions", con=engine, if_exists="replace", index=False)
    active_df.to_sql("active_cookies", con=engine, if_exists="replace", index=False)

    print("✅ cookie_transitions and active_cookies uploaded to Render DB")

if __name__ == "__main__":
    fetch_and_upload_cookie_mapping()
