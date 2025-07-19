import pandas as pd
import os
from sqlalchemy import create_engine
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

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

    # Step 2: Export and read both sheets
    sheet_url = f"https://docs.google.com/spreadsheets/d/1MurIHbFjJQqrlI0DCigLfyGXLXPlvhobEicmiLq61fs/export?format=xlsx"
    excel_path = "data/cookie_mapping.xlsx"
    os.makedirs("data", exist_ok=True)
    os.system(f"curl -L '{sheet_url}' --output {excel_path}")

    # Step 3: Read sheets
    print("📥 Reading sheets...")
    transitions_df = pd.read_excel(excel_path, sheet_name="cookie_transitions", engine="openpyxl")
    active_df = pd.read_excel(excel_path, sheet_name="active_cookies", engine="openpyxl")


    # Optional: Clean column names
    transitions_df.columns = [col.strip().replace('\n', ' ') for col in transitions_df.columns]
    active_df.columns = [col.strip().replace('\n', ' ') for col in active_df.columns]

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
