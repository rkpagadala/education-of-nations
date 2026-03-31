"""
Test whether China's LE trajectory shows a discontinuity at 1980
(when the barefoot doctor system was dismantled) relative to
countries at comparable education levels.

Method: event-study design.
For each education level China passed through, find all countries
that also passed through that level (at any calendar year), then
compare LE trajectories anchored to the year each country reached
that education level. A 20-year window around the anchor.
"""

import pandas as pd
import numpy as np

# ── Load data ──
edu = pd.read_csv("wcde/data/processed/lower_sec_both.csv")
le  = pd.read_csv("wcde/data/processed/e0.csv")

edu["country"] = edu["country"].str.lower()
le["country"]  = le["country"].str.lower()
edu = edu.set_index("country")
le  = le.set_index("country")

# Drop aggregates and regions
aggregates = ["africa", "asia", "europe", "latin america and the caribbean",
              "northern america", "oceania", "world", "less developed regions",
              "more developed regions", "least developed countries",
              "caribbean", "eastern asia", "south america", "southern africa",
              "western asia", "polynesia", "micronesia", "melanesia",
              "central america", "south-eastern asia", "southern asia",
              "central asia", "eastern europe", "northern europe",
              "southern europe", "western europe", "eastern africa",
              "middle africa", "northern africa", "western africa"]
edu = edu.drop([a for a in aggregates if a in edu.index], errors="ignore")
le  = le.drop([a for a in aggregates if a in le.index], errors="ignore")

# Common countries only
common = edu.index.intersection(le.index)
edu = edu.loc[common]
le  = le.loc[common]

years = [str(y) for y in range(1950, 2020, 5)]
edu_num = edu[years].astype(float)
le_num  = le[years].astype(float)


def find_crossing_year(row, threshold):
    """Find the first year a country crosses a threshold (linear interpolation)."""
    for i in range(len(years) - 1):
        v1, v2 = row[years[i]], row[years[i+1]]
        if pd.isna(v1) or pd.isna(v2):
            continue
        if v1 < threshold <= v2:
            # Linear interpolation
            frac = (threshold - v1) / (v2 - v1)
            return int(years[i]) + frac * 5
    return None


def le_at_year(country, yr):
    """Interpolate LE at a fractional year."""
    for i in range(len(years) - 1):
        y1, y2 = int(years[i]), int(years[i+1])
        if y1 <= yr < y2:
            frac = (yr - y1) / 5
            v1, v2 = le_num.loc[country, years[i]], le_num.loc[country, years[i+1]]
            if pd.isna(v1) or pd.isna(v2):
                return None
            return v1 + frac * (v2 - v1)
    return None


# ── Key education thresholds China crossed ──
# China 1965: 31.5%, 1975: 47.1%, 1980: 62.1%
# Test at three levels that bracket the 1980 dismantlement

thresholds = [30, 45, 60, 75]

for threshold in thresholds:
    print(f"\n{'=' * 90}")
    print(f"EDUCATION THRESHOLD: {threshold}% lower secondary completion")
    print(f"{'=' * 90}")

    # Find when each country crossed this threshold
    crossings = {}
    for country in edu_num.index:
        yr = find_crossing_year(edu_num.loc[country], threshold)
        if yr is not None and 1950 <= yr <= 2010:
            crossings[country] = yr

    if "china" not in crossings:
        print(f"  China did not cross {threshold}% in the data window.")
        continue

    china_cross = crossings["china"]
    print(f"\n  China crossed {threshold}% at year ≈ {china_cross:.1f}")
    print(f"\n  LE trajectory relative to crossing year (T=0):")

    offsets = [-10, -5, 0, 5, 10, 15, 20]

    # Collect data
    data = {}
    for country, cross_yr in crossings.items():
        row = {}
        for offset in offsets:
            target_yr = cross_yr + offset
            if 1950 <= target_yr <= 2015:
                val = le_at_year(country, target_yr)
                if val is not None:
                    row[offset] = val
        if len(row) >= 4:  # need at least 4 data points
            data[country] = row

    if "china" not in data:
        print("  China: insufficient LE data around crossing.")
        continue

    # Print header
    hdr = f"  {'Country':<35} {'Cross yr':>9}"
    for o in offsets:
        hdr += f"  {'T'+str(o) if o >=0 else 'T'+str(o):>7}"
    print(hdr)
    print("  " + "-" * (35 + 9 + len(offsets) * 9))

    # Print China first
    c = "china"
    line = f"  {'** CHINA **':<35} {crossings[c]:>9.1f}"
    for o in offsets:
        if o in data[c]:
            line += f"  {data[c][o]:>7.1f}"
        else:
            line += f"  {'—':>7}"
    print(line)

    # Print peers (sorted by crossing year)
    peers = {k: v for k, v in data.items() if k != "china"}
    for country in sorted(peers, key=lambda x: crossings[x]):
        line = f"  {country[:35]:<35} {crossings[country]:>9.1f}"
        for o in offsets:
            if o in peers[country]:
                line += f"  {peers[country][o]:>7.1f}"
            else:
                line += f"  {'—':>7}"
        print(line)

    # ── Compute gains at each offset relative to T=0 ──
    print(f"\n  LE GAINS from T=0 (year of crossing {threshold}%):")
    gain_offsets = [5, 10, 15, 20]
    hdr2 = f"  {'':35} {'':>9}"
    for o in gain_offsets:
        hdr2 += f"  {'ΔLE@T+'+str(o):>9}"
    print(hdr2)
    print("  " + "-" * (35 + 9 + len(gain_offsets) * 11))

    all_gains = {o: [] for o in gain_offsets}

    for country in sorted(data.keys(), key=lambda x: x != "china"):
        if 0 not in data[country]:
            continue
        le_at_0 = data[country][0]
        line = f"  {'** CHINA **' if country == 'china' else country[:35]:<35} {'':>9}"
        for o in gain_offsets:
            if o in data[country]:
                gain = data[country][o] - le_at_0
                line += f"  {gain:>+9.1f}"
                if country != "china":
                    all_gains[o].append(gain)
            else:
                line += f"  {'—':>9}"
        print(line)

    # Summary
    print(f"\n  {'PEER MEAN':<35} {'':>9}", end="")
    china_gains = {}
    for o in gain_offsets:
        if o in data.get("china", {}) and 0 in data["china"]:
            china_gains[o] = data["china"][o] - data["china"][0]
        if all_gains[o]:
            print(f"  {np.mean(all_gains[o]):>+9.1f}", end="")
        else:
            print(f"  {'—':>9}", end="")
    print()

    print(f"  {'PEER MEDIAN':<35} {'':>9}", end="")
    for o in gain_offsets:
        if all_gains[o]:
            print(f"  {np.median(all_gains[o]):>+9.1f}", end="")
        else:
            print(f"  {'—':>9}", end="")
    print()

    print(f"  {'CHINA':<35} {'':>9}", end="")
    for o in gain_offsets:
        if o in china_gains:
            print(f"  {china_gains[o]:>+9.1f}", end="")
        else:
            print(f"  {'—':>9}", end="")
    print()

    print(f"  {'CHINA - PEER MEAN':<35} {'':>9}", end="")
    for o in gain_offsets:
        if o in china_gains and all_gains[o]:
            diff = china_gains[o] - np.mean(all_gains[o])
            print(f"  {diff:>+9.1f}", end="")
        else:
            print(f"  {'—':>9}", end="")
    print()

    print(f"\n  n peers = {len([k for k in data if k != 'china'])}")


# ── CRITICAL TEST: Does China slow MORE than peers after crossing 45%? ──
print(f"\n{'=' * 90}")
print("CRITICAL TEST: LE deceleration after crossing 45%")
print("(China crossed 45% ≈ 1974; barefoot doctors dismantled 1980)")
print("Compare LE gain T+0→T+5 vs T+5→T+10 vs T+10→T+15")
print(f"{'=' * 90}")

threshold = 45
crossings = {}
for country in edu_num.index:
    yr = find_crossing_year(edu_num.loc[country], threshold)
    if yr is not None and 1950 <= yr <= 2005:
        crossings[country] = yr

data = {}
for country, cross_yr in crossings.items():
    row = {}
    for offset in [0, 5, 10, 15, 20]:
        target_yr = cross_yr + offset
        if 1950 <= target_yr <= 2015:
            val = le_at_year(country, target_yr)
            if val is not None:
                row[offset] = val
    if all(o in row for o in [0, 5, 10, 15]):
        data[country] = row

print(f"\n{'Country':<35} {'Cross':>6} {'ΔLE 0→5':>9} {'ΔLE 5→10':>10} {'ΔLE 10→15':>11} {'Decel':>8}")
print("-" * 80)

decels = []
for country in sorted(data.keys(), key=lambda x: x != "china"):
    g1 = data[country][5] - data[country][0]
    g2 = data[country][10] - data[country][5]
    g3 = data[country][15] - data[country][10]
    # Deceleration = ratio of second-period gain to first-period gain
    decel = g2 / g1 if g1 > 0.1 else float('nan')
    label = "** CHINA **" if country == "china" else country[:35]
    print(f"{label:<35} {crossings[country]:>6.0f} {g1:>+9.1f} {g2:>+10.1f} {g3:>+11.1f} {decel:>8.2f}")
    if country != "china" and not np.isnan(decel):
        decels.append(decel)

china_g1 = data["china"][5] - data["china"][0]
china_g2 = data["china"][10] - data["china"][5]
china_decel = china_g2 / china_g1 if china_g1 > 0.1 else float('nan')

print(f"\n{'Peer mean deceleration (g2/g1):':<40} {np.mean(decels):.2f}")
print(f"{'Peer median deceleration (g2/g1):':<40} {np.median(decels):.2f}")
print(f"{'China deceleration (g2/g1):':<40} {china_decel:.2f}")
print(f"\nIf barefoot doctor dismantlement caused a discontinuity unique to China,")
print(f"China's deceleration should be significantly worse than peers'.")
print(f"China's rank: {sum(1 for d in decels if d < china_decel) + 1} of {len(decels) + 1}")
