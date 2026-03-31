"""
Download infant mortality (under-1) from World Bank WDI.
Indicator: SP.DYN.IMRT.IN (Mortality rate, infant, per 1,000 live births)
Saves in same format as child_mortality_u5.csv.
"""

import subprocess
import json
import pandas as pd
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

print("Downloading infant mortality data from World Bank...")
all_records = []
page = 1
while True:
    url = (f"https://api.worldbank.org/v2/country/all/indicator/SP.DYN.IMRT.IN"
           f"?date=1960:2022&per_page=1000&format=json&page={page}")
    result = subprocess.run(["curl", "-s", url], capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)

    meta = data[0]
    records = data[1] if len(data) > 1 else []
    if not records:
        break

    for r in records:
        all_records.append({
            "country": r["country"]["value"],
            "year": r["date"],
            "value": r["value"],
        })

    print(f"  Page {page}/{meta['pages']}: {len(records)} records")
    if page >= meta["pages"]:
        break
    page += 1

df = pd.DataFrame(all_records)
print(f"Downloaded {len(df)} records total")

# Pivot to wide format matching existing CSVs
pivot = df.pivot_table(index="country", columns="year", values="value")
pivot = pivot.sort_index(axis=1)
pivot.index.name = "Country"

# Lowercase country names to match existing data
pivot.index = pivot.index.str.lower()

out_path = os.path.join(DATA_DIR, "infant_mortality_u1.csv")
pivot.to_csv(out_path)
print(f"Saved to {out_path}")
print(f"Countries: {len(pivot)}, Years: {pivot.columns.min()}-{pivot.columns.max()}")
