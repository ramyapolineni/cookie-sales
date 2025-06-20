import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# === CONFIG ===
SPREADSHEET_ID = "1MurIHbFjJQqrlI0DCigLfyGXLXPlvhobEicmiLq61fs"  # From the Google Sheet URL
SERVICE_ACCOUNT_FILE = "cookie-drive-bot.json"

# === AUTHENTICATION ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
gc = gspread.authorize(credentials)

# === CONNECT TO SPREADSHEET ===
sheet = gc.open_by_key(SPREADSHEET_ID)

# === READ cookie_mapping tab ===
mapping_ws = sheet.worksheet("cookie_mapping")
mapping_df = pd.DataFrame(mapping_ws.get_all_records())

# === READ active_cookies tab ===
active_ws = sheet.worksheet("active_cookies")
active_df = pd.DataFrame(active_ws.get_all_records())

# === OPTIONAL: Print preview ===
print("✅ Cookie Mapping (Preview):")
print(mapping_df.head())

print("\n✅ Active Cookies (Preview):")
print(active_df.head())


# === FUNCTION: Apply Cookie Mapping (Flexible) ===
def apply_cookie_mapping(df, mapping_df, year):
    for _, row in mapping_df.iterrows():
        if row.get('start_year', year) != year:
            continue

        old_cookie = row['old_cookie']
        original = df[df['cookie_type'] == old_cookie]

        if original.empty:
            continue

        used_multi_mapping = False

        for i in range(1, 4):  # supports up to 3 splits
            new_cookie_col = f'new_cookie_{i}'
            percent_col = f'percent_{i}'

            if pd.notna(row.get(new_cookie_col)) and not pd.isna(row.get(percent_col)):
                new_cookie = row[new_cookie_col]
                percent = row[percent_col]

                if percent > 0:
                    split = original.copy()
                    split['cookie_type'] = new_cookie
                    split['number_cases_sold'] *= percent / 100
                    df = pd.concat([df, split], ignore_index=True)
                    print(f"🔁 {old_cookie} → {new_cookie} ({percent}%)")
                    used_multi_mapping = True

        if not used_multi_mapping and pd.notna(row.get('new_cookie')):
            # Use legacy mapping
            percent = row.get('transfer_percent', 100)
            transferred = original.copy()
            transferred['cookie_type'] = row['new_cookie']
            transferred['number_cases_sold'] *= percent / 100
            df = pd.concat([df, transferred], ignore_index=True)
            print(f"🔁 {old_cookie} → {row['new_cookie']} ({percent}%) [Legacy format]")

        # Remove original
        df = df[df['cookie_type'] != old_cookie]

    return df
