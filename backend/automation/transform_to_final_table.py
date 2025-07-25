import pandas as pd
import os
import re
from glob import glob
from sqlalchemy import create_engine


# === LOAD + CLEAN PARTICIPATION FILE ===
def load_and_clean_participation(file_path):
    part_df = pd.read_excel(file_path, sheet_name='eBudde Report', skiprows=4)
    part_df.columns = [col.strip().replace('\n', ' ') if isinstance(col, str) else col for col in part_df.columns]

    part_df = part_df.rename(columns={
        'SU Name': 'SU_Name',
        'SU #': 'SU_Num',
        'Troop': 'troop_id',
        '# Girls Sellg': 'number_of_girls'
    })

    part_df = part_df[['SU_Name', 'SU_Num', 'troop_id', 'number_of_girls']].copy()
    # Remove alphabetical characters from troop_id
    part_df['troop_id'] = part_df['troop_id'].astype(str).str.replace(r'[A-Za-z]', '', regex=True).str.strip()
    # Drop rows where troop_id is empty after cleaning
    part_df = part_df[part_df['troop_id'] != '']
    return part_df


# === LOAD + CLEAN TROOP SALES FILE ===
# === COOKIE TYPE NORMALIZATION ===
def get_canonical_cookie_lookup():
    db_url = os.getenv("RENDER_DATABASE_URL")
    if not db_url:
        raise ValueError("RENDER_DATABASE_URL not set in environment variables")
    engine = create_engine(db_url)
    active_cookies_df = pd.read_sql_table("active_cookies", con=engine)
    canonical_names = active_cookies_df.iloc[:, 0].dropna().unique()  # assumes first column is the name
    lookup = {}
    print("\n[DEBUG] Canonical names from DB:")
    for name in canonical_names:
        cleaned = re.sub(r'\s+|[-\'"`]', '', str(name).lower())
        lookup[cleaned] = name
        print(f"  Canonical: {repr(name)} | Cleaned: {repr(cleaned)}")
    return lookup

def normalize_cookie_type_dynamic(name, canonical_lookup):
    if pd.isnull(name):
        return name
    cleaned = re.sub(r'\s+|[-\'"`]', '', str(name).lower())
    result = canonical_lookup.get(cleaned, name)
    print(f"[DEBUG] Raw: {repr(name)} | Cleaned: {repr(cleaned)} | Normalized: {repr(result)}")
    return result

def repair_split_cookie_columns(df, canonical_names):
    # Remove all whitespace and dashes for matching
    canonical_cleaned = {re.sub(r'\s+|[-\'"`]', '', name.lower()): name for name in canonical_names}
    new_columns = []
    skip_next = False
    columns = list(df.columns)
    i = 0
    while i < len(columns):
        if skip_next:
            skip_next = False
            i += 1
            continue
        col = columns[i]
        # Try to combine with next column if not a known id col
        if i + 1 < len(columns):
            combined = str(col) + str(columns[i+1])
            cleaned = re.sub(r'\s+|[-\'"`]', '', combined.lower())
            if cleaned in canonical_cleaned:
                new_columns.append(canonical_cleaned[cleaned])
                skip_next = True
            else:
                new_columns.append(col)
        else:
            new_columns.append(col)
        i += 1
    df.columns = new_columns
    return df

def load_and_clean_sales(file_path, year):
    sales_df = pd.read_excel(file_path, skiprows=4)
    # Dynamically fetch canonical names for header repair
    canonical_lookup = get_canonical_cookie_lookup()
    canonical_names = list(canonical_lookup.values())
    sales_df = repair_split_cookie_columns(sales_df, canonical_names)
    sales_df.columns = [col.strip().replace('\n', ' ') if isinstance(col, str) else col for col in sales_df.columns]

    sales_df = sales_df.rename(columns={
        'Service Unit Name': 'SU_Name',
        'Service Unit Number': 'SU_Num',
        'Troop': 'troop_id'
    })

    id_cols = ['troop_id', 'SU_Name', 'SU_Num']
    cookie_cols = [col for col in sales_df.columns if col not in id_cols and "Total" not in col]

    melted = sales_df.melt(
        id_vars=id_cols,
        value_vars=cookie_cols,
        var_name='cookie_type',
        value_name='number_of_cookies_sold'
    )

    # Remove alphabetical characters from troop_id
    melted['troop_id'] = melted['troop_id'].astype(str).str.replace(r'[A-Za-z]', '', regex=True).str.strip()
    # Drop rows where troop_id is empty after cleaning
    melted = melted[melted['troop_id'] != '']

    print("\n[DEBUG] Sample normalization results:")
    sample = melted['cookie_type'].dropna().unique()[:10]
    for s in sample:
        normalize_cookie_type_dynamic(s, canonical_lookup)
    
    print("\n[DEBUG] Row-by-row cookie_type normalization (first 20 rows):")
    for idx, row in melted.head(20).iterrows():
        raw = row['cookie_type']
        cleaned = re.sub(r'\s+|[-\'"`]', '', str(raw).lower())
        normalized = canonical_lookup.get(cleaned, raw)
        print(f"  Row {idx}: Raw: {repr(raw)} | Cleaned: {repr(cleaned)} | Normalized: {repr(normalized)}")

    melted['cookie_type'] = melted['cookie_type'].apply(lambda x: normalize_cookie_type_dynamic(x, canonical_lookup))

    melted['number_cases_sold'] = melted['number_of_cookies_sold'] / 12
    melted.drop(columns=['number_of_cookies_sold'], inplace=True)
    melted['date'] = year

    return melted


# === MERGE SALES + PARTICIPATION FILES ===
def merge_with_participation(melted_df, part_df):
    # Drop SU_Name and SU_Num from sales file before merge to avoid duplication
    melted_df = melted_df.drop(columns=['SU_Name', 'SU_Num'], errors='ignore')

    # Merge on troop_id only; keep SU_Name and SU_Num from participation file
    merged = pd.merge(melted_df, part_df, how='left', on='troop_id')

    return merged

# === SAVE FINAL FILE ===
def save_final(df, year):
    df['period'] = year - 2019  # 2020 = period 1

    final_df = df.rename(columns={
        'troop_id': 'troop_id',
        'date': 'date'
    })

    os.makedirs("data", exist_ok=True)
    final_file = f"data/FinalCookieSales_{year}.csv"
    final_df.to_csv(final_file, index=False)
    print(f"✅ Final data for {year} saved to {final_file}")
    return final_df


# === MAIN SCRIPT TO PROCESS UNPROCESSED YEARS ONLY ===
if __name__ == "__main__":
    part_files = glob("data/Participation_*.xlsx")
    sales_files = glob("data/TroopSales_*.xlsx")

    part_years = {int(re.search(r"(\d{4})", f).group(1)) for f in part_files}
    sales_years = {int(re.search(r"(\d{4})", f).group(1)) for f in sales_files}
    all_years = sorted(part_years & sales_years)
    target_years = [y for y in all_years if y >= 2025]

    # Skip already processed years
    processed_files = glob("data/FinalCookieSales_*.csv")
    processed_years = {
        int(re.search(r"(\d{4})", f).group(1))
        for f in processed_files
        if "2025plus" not in f and "all_years" not in f
    }
    unprocessed_years = [y for y in target_years if y not in processed_years]

    if not unprocessed_years:
        print("ℹ️ No new year data to process.")
    else:
        for year in unprocessed_years:
            print(f"\n📦 Processing year {year}...")

            part_path = f"data/Participation_{year}.xlsx"
            sales_path = f"data/TroopSales_{year}.xlsx"

            try:
                sales_df = load_and_clean_sales(sales_path, year)
                part_df = load_and_clean_participation(part_path)
                merged_df = merge_with_participation(sales_df, part_df)
                save_final(merged_df, year)

            except Exception as e:
                print(f"⚠️ Skipped year {year} due to error: {e}")
