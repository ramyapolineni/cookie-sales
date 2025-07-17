import pandas as pd
import os
import re
from glob import glob


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
    return part_df


# === LOAD + CLEAN TROOP SALES FILE ===
def load_and_clean_sales(file_path, year):
    sales_df = pd.read_excel(file_path, skiprows=4)
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

    melted['number_cases_sold'] = melted['number_of_cookies_sold'] / 12
    melted.drop(columns=['number_of_cookies_sold'], inplace=True)
    melted['date'] = year

    return melted


# === MERGE SALES + PARTICIPATION FILES ===
def merge_with_participation(melted_df, part_df):
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
                final_df = apply_cookie_mapping(merged_df, mapping_df, year)
                save_final(final_df, year)

            except Exception as e:
                print(f"⚠️ Skipped year {year} due to error: {e}")
