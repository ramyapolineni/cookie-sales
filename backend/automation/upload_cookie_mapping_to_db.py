import pandas as pd
import os
import requests
from sqlalchemy import create_engine
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def fetch_and_upload_cookie_mapping():
    # Authenticate with service account
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

    # Google Drive Folder + target file name
    FOLDER_ID = "1e_3Y1LEuvqCicUG64G4QOkX1yHMEOvem"
    TARGET_FILENAME = "cookie_mapping"

    print("🔍 Searching for cookie_mapping Google Sheet...")
    file_list = drive.ListFile({
        'q': f"'{FOLDER_ID}' in parents and trashed=false and mimeType='application/vnd.google-apps.spreadsheet'"
    }).GetList()

    file_id = None
    for file in file_list:
        if file['title'].strip().lower() == TARGET_FILENAME.lower():
            file_id = file['id']
            print(f"✅ Found file: {file['title']}")
            break

    if not file_id:
        raise FileNotFoundError("❌ cookie_mapping file not found in Google Drive folder")

    # Step 2: Export file using Google Sheets export endpoint
    export_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
    excel_path = "data/cookie_mapping.xlsx"
    os.makedirs("data", exist_ok=True)

    print("⬇️ Downloading as Excel...")
    response = requests.get(export_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download Excel export. Status: {response.status_code}")
    with open(excel_path, "wb") as f:
        f.write(response.content)

    # Step 3: Read sheets
    print("📥 Reading sheets...")
    transitions_df = pd.read_excel(excel_path, sheet_name="cookie_transitions", engine="openpyxl")
    active_df = pd.read_excel(excel_path, sheet_name="active_cookies", engine="openpyxl")

    # Clean column names
    transitions_df.columns = [col.strip().replace('\n', ' ') for col in transitions_df.columns]
    active_df.columns = [col.strip().replace('\n', ' ') for col in active_df.columns]

    # Step 4: Upload to database
    db_url = os.getenv("RENDER_DATABASE_URL")
    if not db_url:
        raise ValueError("RENDER_DATABASE_URL not set in environment variables")

    engine = create_engine(db_url)
    transitions_df.to_sql("cookie_transitions", con=engine, if_exists="replace", index=False)
    active_df.to_sql("active_cookies", con=engine, if_exists="replace", index=False)

    print("✅ cookie_transitions and active_cookies uploaded to Render DB")

if __name__ == "__main__":
    fetch_and_upload_cookie_mapping()
