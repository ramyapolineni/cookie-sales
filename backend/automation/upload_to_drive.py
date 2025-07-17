import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

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

upload_folder_id = '1oYZceOPsUXtVff7nm6Iev2VOuUyvxQvD'  # 🔁 Replace this
output_file = 'data/FinalCookieSales_all_years.csv'

if os.path.exists(output_file):
    file_name = os.path.basename(output_file)
    gfile = drive.CreateFile({'title': file_name, 'parents': [{'id': upload_folder_id}]})
    gfile.SetContentFile(output_file)
    gfile.Upload()
    print(f'✅ Uploaded {file_name} to Google Drive folder {upload_folder_id}')
else:
    print(f'⚠️ Output file not found: {output_file}')
