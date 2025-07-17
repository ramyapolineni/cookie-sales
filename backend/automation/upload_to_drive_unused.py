import os
import shutil
import zipfile
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# === Authenticate using service account ===
gauth = GoogleAuth()
gauth.settings['client_config_backend'] = 'service'
gauth.settings['service_config'] = {
    'client_json_file_path': 'service_account.json',
    'client_user_email': 'cookie-drive-bot@cookie-sales-463514.iam.gserviceaccount.com',
    'client_id': '',
    'client_secret': ''
}
gauth.ServiceAuth()
drive = GoogleDrive(gauth)

# === Define paths and names ===
upload_folder_id = '1oYZceOPsUXtVff7nm6Iev2VOuUyvxQvD'  # 🔁 Replace this
base_file = 'data/FinalCookieSales_all_years.csv'

# Create versioned filename
today_str = datetime.today().strftime('%Y-%m-%d')
versioned_name = f'FinalCookieSales_all_years_{today_str}.csv'
versioned_path = f'data/{versioned_name}'
zip_path = f'{versioned_path}.zip'

# === Copy base file to versioned file ===
if os.path.exists(base_file):
    shutil.copy(base_file, versioned_path)
    print(f"📝 Created versioned copy: {versioned_path}")

    # === Zip the versioned file ===
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(versioned_path, arcname=versioned_name)
    print(f"📦 Compressed version: {zip_path}")

    # === Upload the .zip file to Google Drive ===
    gfile = drive.CreateFile({'title': os.path.basename(zip_path), 'parents': [{'id': upload_folder_id}]})
    gfile.SetContentFile(zip_path)
    gfile.Upload()
    print(f'✅ Uploaded {os.path.basename(zip_path)} to Google Drive folder {upload_folder_id}')
else:
    print(f'⚠️ Output file not found: {base_file}')
