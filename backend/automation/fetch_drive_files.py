from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

def fetch_drive_files_from_google():
    # Use service account credentials (path set by env variable in Render)
    gauth = GoogleAuth()
    gauth.settings['client_config_backend'] = 'service'
    gauth.settings['service_config'] = {
        "client_json_file_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "client_user_email": "cookie-drive-bot@cookie-sales-463514.iam.gserviceaccount.com",
        "client_id": "",
        "client_secret": ""
    }
    gauth.ServiceAuth()

    # Initialize Google Drive
    drive = GoogleDrive(gauth)

    # Google Drive Folder IDs
    RAW_PARENT_FOLDER_ID = "1W9y9J8AfRYyhDwgRFYWVm4Gkp4jBQldz"        # Raw files folder
    HISTORICAL_FOLDER_ID = "1prylqI0RIr97bBGu--w9KQ1PefbTj7Xc"         # Final historical data folder

    os.makedirs("data", exist_ok=True)

    # === STEP 1: Find latest year folder ===
    folder_list = drive.ListFile({
        'q': f"'{RAW_PARENT_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    year_folders = {}
    for folder in folder_list:
        if folder['title'].isdigit():
            year_folders[int(folder['title'])] = folder['id']

    latest_year = max(year_folders.keys())
    latest_folder_id = year_folders[latest_year]
    print(f"📁 Latest year folder found: {latest_year}")

    # === STEP 2: Download files from latest year ===
    file_list = drive.ListFile({
        'q': f"'{latest_folder_id}' in parents and trashed=false"
    }).GetList()

    for file in file_list:
        local_path = os.path.join("data", file["title"])
        if not os.path.exists(local_path):
            print(f"⬇️  Downloading: {file['title']}")
            file.GetContentFile(local_path)
        else:
            print(f"📁 Already exists: {file['title']} (skipping)")

    print(f"✅ All files for {latest_year} downloaded to /data/")

    # === STEP 3: Download FinalSales2020to2024.csv if missing ===
    historical_filename = "FinalSales2020to2024.csv"
    historical_local_path = os.path.join("data", historical_filename)

    if os.path.exists(historical_local_path):
        print(f"📁 {historical_filename} already exists locally. Skipping download.")
    else:
        print(f"🔍 Looking for {historical_filename} in historical folder...")

        hist_file_list = drive.ListFile({
            'q': f"'{HISTORICAL_FOLDER_ID}' in parents and trashed=false"
        }).GetList()

        found = False
        for file in hist_file_list:
            if file["title"] == historical_filename:
                print(f"⬇️ Downloading historical file: {historical_filename}")
                file.GetContentFile(historical_local_path)
                print(f"✅ Saved: {historical_local_path}")
                found = True
                break

        if not found:
            print(f"⚠️ {historical_filename} not found in GS_Cookie_Sales_Final folder.")

    print("🚀 Fetching files from Google Drive completed successfully!\n")