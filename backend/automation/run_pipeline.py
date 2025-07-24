import os
import pandas as pd
import re
from glob import glob
from sqlalchemy import create_engine

from fetch_drive_files import fetch_drive_files_from_google
from transform_to_final_table import (
    load_and_clean_sales,
    load_and_clean_participation,
    merge_with_participation,
    save_final
)
from upload_to_render_db import upload_to_render_db  # <-- new import

# === CONFIG: DB CONNECTION ===
def get_db_engine():
    db_url = os.getenv("RENDER_DATABASE_URL")  # Ensure this is set in your environment
    if not db_url:
        raise ValueError("Missing RENDER_DATABASE_URL in environment variables.")
    return create_engine(db_url)

# === STEP 0: Fetch active cookie name mappings ===
def get_standard_cookie_names():
    engine = get_db_engine()
    query = "SELECT DISTINCT standard_name FROM active_cookies"
    df = pd.read_sql(query, engine)

    def normalize(name):
        return re.sub(r'\s+', ' ', name).strip().lower()

    name_map = {normalize(row['standard_name']): row['standard_name'] for _, row in df.iterrows()}
    return name_map

# === Helper: Clean and apply standard names ===
def standardize_cookie_names_from_map(df, name_column, name_map):
    def clean_and_map(name):
        if pd.isna(name):
            return name
        cleaned = re.sub(r'\s+', ' ', name).strip().lower()
        return name_map.get(cleaned, name.strip().title())  # fallback to Title case

    df[name_column] = df[name_column].apply(clean_and_map)

    # Log unmapped cookie names
    unmapped = df[~df[name_column].str.lower().isin(name_map.keys())][name_column].unique()
    if len(unmapped):
        print("⚠️ Unmapped cookie names found:", unmapped)

    return df

# === STEP 1: Get list of years to process ===
def get_unprocessed_years():
    part_files = glob("data/Participation_*.xlsx")
    sales_files = glob("data/TroopSales_*.xlsx")

    part_years = {int(re.search(r"(\d{4})", f).group(1)) for f in part_files}
    sales_years = {int(re.search(r"(\d{4})", f).group(1)) for f in sales_files}
    available_years = sorted(part_years & sales_years)

    # Only consider 2025 and beyond
    target_years = [y for y in available_years if y >= 2025]

    # Skip already processed years
    processed_files = glob("data/FinalCookieSales_*.csv")
    processed_years = {
        int(re.search(r"(\d{4})", f).group(1))
        for f in processed_files
        if "all_years" not in f
    }

    return [y for y in target_years if y not in processed_years]

# === STEP 2: Merge all years into final output ===
def combine_all_years():
    base_path = "data/FinalSales2020to2024.csv"
    new_files = glob("data/FinalCookieSales_*.csv")
    new_years_dfs = []

    for f in new_files:
        year_match = re.search(r"(\d{4})", f)
        if year_match and int(year_match.group(1)) >= 2025:
            new_years_dfs.append(pd.read_csv(f))

    if not new_years_dfs:
        print("ℹ️ No new year files found to merge with historical data.")
        return None

    all_new = pd.concat(new_years_dfs, ignore_index=True)

    if os.path.exists(base_path):
        base_df = pd.read_csv(base_path)
        combined = pd.concat([base_df, all_new], ignore_index=True)
    else:
        print("⚠️ Historical base file not found — using new years only.")
        combined = all_new

    output_path = "data/FinalCookieSales_all_years.csv"
    combined.to_csv(output_path, index=False)
    print(f"✅ Combined all-year file saved to: {output_path}")
    return combined

# === MAIN RUNNER ===
if __name__ == "__main__":
    print("🚀 Starting cookie sales pipeline...\n")

    # Step 1: Pull raw files from Google Drive
    fetch_drive_files_from_google()

    # Step 2: Identify unprocessed years
    new_years = get_unprocessed_years()

    if not new_years:
        print("✅ All available data is already processed.\n")
    else:
        # Load cookie name mapping from DB
        try:
            cookie_map = get_standard_cookie_names()
            print("🔍 Loaded standard cookie name mappings from database.")
        except Exception as e:
            print(f"❌ Failed to load cookie name map: {e}")
            cookie_map = {}

        for year in new_years:
            print(f"\n📦 Processing year {year}...")

            part_path = f"data/Participation_{year}.xlsx"
            sales_path = f"data/TroopSales_{year}.xlsx"

            try:
                sales_df = load_and_clean_sales(sales_path, year)
                part_df = load_and_clean_participation(part_path)
                merged_df = merge_with_participation(sales_df, part_df)

                # Apply standardization
                if 'Cookie' in merged_df.columns and cookie_map:
                    merged_df = standardize_cookie_names_from_map(merged_df, "Cookie", cookie_map)

                save_final(merged_df, year)
            except Exception as e:
                print(f"⚠️ Failed to process {year}: {e}")

    # Step 3: Merge with historical (2020–2024)
    combined_df = combine_all_years()

    # Step 4: Upload to Render PostgreSQL
    if combined_df is not None:
        try:
            upload_to_render_db(combined_df)
        except Exception as e:
            print(f"❌ Upload to database failed: {e}")
    else:
        print("⚠️ Skipping DB upload: No new data to combine.")

    print("\n🎉 Pipeline complete!")
