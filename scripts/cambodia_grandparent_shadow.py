# =============================================================================
# PAPER REFERENCE
# Script:  scripts/cambodia_grandparent_shadow.py
# Paper:   "Education of Nations"
#
# Produces:
#   Test of the "grandparent shadow" prediction for Cambodia. The Khmer Rouge
#   froze education at ~10% during 1975-1985. Those frozen cohorts became
#   *parents* of 1996-2010 school-age children (the first PT shadow, already
#   in the paper). They also become *grandparents* of 2015-2025 school-age
#   children — predicting a second, deeper drag on recovery visible as a
#   slower expansion rate than peer countries with similar mother-level
#   education but undamaged grandparent baselines.
#
# Inputs:
#   wcde/data/processed/completion_both_long.csv
#
# Outputs:
#   checkin/cambodia_grandparent_shadow.json
# =============================================================================
"""
cambodia_grandparent_shadow.py

Test whether Cambodia's post-2010 recovery is slower than predicted by
mother-level education alone, consistent with a grandparent shadow from
the Khmer Rouge disruption.

Design:
  1. Identify countries at a similar lower-secondary completion level to
     Cambodia in 2010 (~36%) but whose grandmother cohort (T-50) was NOT
     disrupted — i.e., grandmother education tracks normal expansion.
  2. Compare 2010→2025 expansion rates: Cambodia vs. peers.
  3. Show that Cambodia's grandmother education (T-50 from 2020 = 1970,
     T-50 from 2025 = 1975) is abnormally low relative to peers, and
     that the expansion gap is consistent with the grandmother regression
     coefficient from the paper.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from _shared import PROC, CHECKIN, REGIONS, write_checkin

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
df = df[~df["country"].isin(REGIONS)].copy()

# ── Cambodia trajectory ──────────────────────────────────────────────────
cam = df[df["country"] == "Cambodia"].sort_values("year")

print("=" * 70)
print("CAMBODIA EDUCATION TRAJECTORY (lower secondary completion, %)")
print("=" * 70)
for _, row in cam.iterrows():
    yr = int(row["year"])
    ls = row["lower_sec"]
    if pd.notna(ls):
        print(f"  {yr}: {ls:5.1f}%")

# ── Three-generation structure for Cambodia ──────────────────────────────
print("\n" + "=" * 70)
print("THREE-GENERATION STRUCTURE: CAMBODIA")
print("=" * 70)
print("\nFor children reaching age 20-24 at time T:")
print("  Mother's education = completion at T-25")
print("  Grandmother's education = completion at T-50\n")

for outcome_year in [2010, 2015, 2020, 2025]:
    mother_year = outcome_year - 25
    gm_year = outcome_year - 50
    child_val = cam[cam["year"] == outcome_year]["lower_sec"].values
    mother_val = cam[cam["year"] == mother_year]["lower_sec"].values
    gm_val = cam[cam["year"] == gm_year]["lower_sec"].values

    child_str = f"{child_val[0]:.1f}%" if len(child_val) > 0 and pd.notna(child_val[0]) else "n/a"
    mother_str = f"{mother_val[0]:.1f}%" if len(mother_val) > 0 and pd.notna(mother_val[0]) else "n/a"
    gm_str = f"{gm_val[0]:.1f}%" if len(gm_val) > 0 and pd.notna(gm_val[0]) else "n/a"

    print(f"  T={outcome_year}: child={child_str}  mother(T-25={mother_year})={mother_str}  "
          f"grandmother(T-50={gm_year})={gm_str}")

# ── Find peer countries: similar completion in 2010, no disruption ───────
# Cambodia in 2010: ~36% lower secondary completion
# Peers: countries at 30-42% in 2010 (±6pp window)
CAM_2010 = float(cam[cam["year"] == 2010]["lower_sec"].values[0])
PEER_WINDOW = 8  # ±8pp

peers_2010 = df[df["year"] == 2010].copy()
peers_2010 = peers_2010[
    (peers_2010["lower_sec"] >= CAM_2010 - PEER_WINDOW) &
    (peers_2010["lower_sec"] <= CAM_2010 + PEER_WINDOW) &
    (peers_2010["country"] != "Cambodia")
].copy()

# For each peer, get the full trajectory and grandmother education
peer_countries = sorted(peers_2010["country"].unique())

print(f"\n{'=' * 70}")
print(f"PEER COUNTRIES (lower_sec within ±{PEER_WINDOW}pp of Cambodia's "
      f"{CAM_2010:.1f}% in 2010)")
print(f"{'=' * 70}")

rows = []
for c in peer_countries:
    cdata = df[df["country"] == c]
    val_2010 = cdata[cdata["year"] == 2010]["lower_sec"].values
    val_2025 = cdata[cdata["year"] == 2025]["lower_sec"].values
    # Grandmother education for the 2020 outcome cohort = education in 1970
    gm_1970 = cdata[cdata["year"] == 1970]["lower_sec"].values
    # Grandmother education for the 2025 outcome cohort = education in 1975
    gm_1975 = cdata[cdata["year"] == 1975]["lower_sec"].values

    if len(val_2010) > 0 and len(val_2025) > 0:
        e2010 = float(val_2010[0])
        e2025 = float(val_2025[0])
        gain = e2025 - e2010
        gm70 = float(gm_1970[0]) if len(gm_1970) > 0 and pd.notna(gm_1970[0]) else np.nan
        gm75 = float(gm_1975[0]) if len(gm_1975) > 0 and pd.notna(gm_1975[0]) else np.nan
        rows.append({
            "country": c, "edu_2010": e2010, "edu_2025": e2025,
            "gain_2010_2025": gain, "gm_1970": gm70, "gm_1975": gm75
        })

# Cambodia row
cam_2025 = float(cam[cam["year"] == 2025]["lower_sec"].values[0])
cam_gm_1970 = cam[cam["year"] == 1970]["lower_sec"].values
cam_gm_1975 = cam[cam["year"] == 1975]["lower_sec"].values
cam_gm70 = float(cam_gm_1970[0]) if len(cam_gm_1970) > 0 else np.nan
cam_gm75 = float(cam_gm_1975[0]) if len(cam_gm_1975) > 0 else np.nan
cam_gain = cam_2025 - CAM_2010

peers_df = pd.DataFrame(rows)

# Print comparison table
print(f"\n{'Country':<35} {'Edu 2010':>8} {'Edu 2025':>8} {'Gain':>8} "
      f"{'GM 1970':>8} {'GM 1975':>8}")
print("-" * 85)
print(f"{'*** CAMBODIA ***':<35} {CAM_2010:>7.1f}% {cam_2025:>7.1f}% "
      f"{cam_gain:>+7.1f}  {cam_gm70:>7.1f}% {cam_gm75:>7.1f}%")
print("-" * 85)
for _, r in peers_df.sort_values("gain_2010_2025").iterrows():
    gm70_str = f"{r['gm_1970']:>7.1f}%" if pd.notna(r["gm_1970"]) else "    n/a "
    gm75_str = f"{r['gm_1975']:>7.1f}%" if pd.notna(r["gm_1975"]) else "    n/a "
    print(f"{r['country']:<35} {r['edu_2010']:>7.1f}% {r['edu_2025']:>7.1f}% "
          f"{r['gain_2010_2025']:>+7.1f}  {gm70_str} {gm75_str}")

# ── Summary statistics ───────────────────────────────────────────────────
peer_mean_gain = peers_df["gain_2010_2025"].mean()
peer_median_gain = peers_df["gain_2010_2025"].median()
peer_mean_gm70 = peers_df["gm_1970"].mean()
peer_mean_gm75 = peers_df["gm_1975"].mean()

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  Cambodia gain 2010→2025:        {cam_gain:+.1f} pp")
print(f"  Peer mean gain 2010→2025:       {peer_mean_gain:+.1f} pp")
print(f"  Peer median gain 2010→2025:     {peer_median_gain:+.1f} pp")
print(f"  Cambodia shortfall vs mean:     {cam_gain - peer_mean_gain:+.1f} pp")
print(f"  Cambodia shortfall vs median:   {cam_gain - peer_median_gain:+.1f} pp")
print(f"\n  Cambodia GM edu (1970):         {cam_gm70:.1f}%")
print(f"  Cambodia GM edu (1975):         {cam_gm75:.1f}%")
print(f"  Peer mean GM edu (1970):        {peer_mean_gm70:.1f}%")
print(f"  Peer mean GM edu (1975):        {peer_mean_gm75:.1f}%")
print(f"  GM deficit (1970):              {cam_gm70 - peer_mean_gm70:+.1f} pp")
print(f"  GM deficit (1975):              {cam_gm75 - peer_mean_gm75:+.1f} pp")

# ── Grandmother shadow interpretation ────────────────────────────────────
# The grandmother regression (paper §4.6) shows β_gm ≈ -0.059 for TFR
# and β_gm ≈ +0.271 for child education at low baselines.
# If GM education is depressed by X pp, the predicted drag on child
# education ≈ 0.271 * X pp.
gm_deficit_75 = cam_gm75 - peer_mean_gm75
predicted_drag = 0.271 * gm_deficit_75
actual_shortfall = cam_gain - peer_median_gain

print(f"\n{'=' * 70}")
print("GRANDMOTHER SHADOW PREDICTION")
print(f"{'=' * 70}")
print(f"  GM deficit (1975 vs peers):     {gm_deficit_75:+.1f} pp")
print(f"  β_gm for child edu (paper):     0.271")
print(f"  Predicted drag (β_gm × deficit):{predicted_drag:+.1f} pp")
print(f"  Actual shortfall vs peer median: {actual_shortfall:+.1f} pp")
if predicted_drag < 0 and actual_shortfall < 0:
    ratio = actual_shortfall / predicted_drag
    print(f"  Ratio actual/predicted:          {ratio:.2f}")
    print(f"  → Grandmother shadow accounts for ~{min(ratio, 1.0)*100:.0f}% of Cambodia's shortfall")

# ── Checkin ──────────────────────────────────────────────────────────────
checkin = {
    "test": "Cambodia grandparent shadow",
    "cambodia_2010": round(CAM_2010, 1),
    "cambodia_2025": round(cam_2025, 1),
    "cambodia_gain_2010_2025": round(cam_gain, 1),
    "peer_median_gain_2010_2025": round(peer_median_gain, 1),
    "cambodia_gm_1970": round(cam_gm70, 1),
    "cambodia_gm_1975": round(cam_gm75, 1),
    "peer_mean_gm_1970": round(peer_mean_gm70, 1),
    "peer_mean_gm_1975": round(peer_mean_gm75, 1),
    "n_peers": len(peers_df),
    "peers": sorted(peers_df["country"].tolist()),
}
write_checkin("cambodia_grandparent_shadow.json", checkin,
              script_path="scripts/cambodia_grandparent_shadow.py")
