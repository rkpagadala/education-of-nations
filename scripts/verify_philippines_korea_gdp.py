"""
Verify Philippines vs Korea GDP per capita comparison (1960).
Numbers cited in Section 9 of the paper.

Source: World Bank WDI, constant 2017 USD (NY.GDP.PCAP.KD)
"""

import pandas as pd

df = pd.read_csv("data/gdppercapita_us_inflation_adjusted.csv")
df["Country"] = df["Country"].str.lower()

EXPECTED = {
    "philippines": 1124,
    "korea, rep.": 1038,
    "thailand": 592,
    "indonesia": 598,
    "india": 313,
    "china": 241,
}

print("GDP per capita in 1960 (constant 2017 USD)")
print("=" * 50)

all_ok = True
for country, expected in EXPECTED.items():
    row = df[df["Country"] == country]
    if row.empty:
        row = df[df["Country"].str.contains(country, na=False)]
    actual = round(row.iloc[0]["1960"])
    match = "OK" if actual == expected else "MISMATCH"
    if actual != expected:
        all_ok = False
    print(f"  {country:20s}: ${actual:,}  (paper: ${expected:,})  {match}")

print()
if all_ok:
    print("All numbers verified.")
else:
    print("WARNING: mismatches found.")
