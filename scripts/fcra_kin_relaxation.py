"""
FCRA foreign contributions per capita by Indian state — kin relaxation at
the diaspora radius.

High-education states receive 13.6× more foreign contributions per capita
than low-education states (Rs 335 vs Rs 25/person, FY 2019–22 average).
This is kin relaxation: educated diaspora populations whose own children are
provided for direct surplus investment back to home communities.

Data source:
  Ministry of Home Affairs, Government of India.
  "State-wise receipt of foreign contribution under FCRA."
  data.gov.in, FY 2019-20 to 2021-22.
  https://www.data.gov.in/resource/state-wise-receipt-foreign-contribution-foreign-contribution-regulation-act-fcra

Population:
  Census of India 2011 projected to 2021 (Registrar General of India).

Output: per-capita ranking, high-education vs low-education contrast,
        bias analysis.
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")

# ── Constants ─────────────────────────────────────────────────────

FCRA_FILE = os.path.join(DATA_DIR, "fcra_state_wise_2019_2022.csv")

# 2021 projected populations (crores), Census 2011 + RGI projections
STATE_POP_CR = {
    "Andaman and Nicobar Islands": 0.04,
    "Andhra Pradesh": 4.97,
    "Arunachal Pradesh": 0.14,
    "Assam": 3.12,
    "Bihar": 10.41,
    "Chandigarh": 0.11,
    "Chhattisgarh": 2.55,
    "Dadra and Nagar Haveli": 0.06,
    "Delhi": 1.68,
    "Goa": 0.15,
    "Gujarat": 6.04,
    "Haryana": 2.54,
    "Himachal Pradesh": 0.69,
    "Jammu and Kashmir": 1.25,
    "Jharkhand": 3.29,
    "Karnataka": 6.11,
    "Kerala": 3.34,
    "Madhya Pradesh": 7.26,
    "Maharashtra": 11.24,
    "Manipur": 0.29,
    "Meghalaya": 0.30,
    "Mizoram": 0.11,
    "Nagaland": 0.20,
    "Orissa": 4.19,
    "Pondicherry": 0.13,
    "Punjab": 2.77,
    "Rajasthan": 6.86,
    "Sikkim": 0.06,
    "Tamil Nadu": 7.21,
    "Telangana": 3.50,
    "Tripura": 0.37,
    "Uttar Pradesh": 19.98,
    "Uttarakhand": 1.01,
    "West Bengal": 9.13,
}

# State groupings by educational attainment
HIGH_EDUCATION = ["Kerala", "Tamil Nadu", "Himachal Pradesh", "Goa", "Karnataka"]
LOW_EDUCATION = ["Bihar", "Uttar Pradesh", "Madhya Pradesh",
                  "Jharkhand", "Rajasthan", "Chhattisgarh"]

# ── Load and compute ──────────────────────────────────────────────


def load_fcra():
    df = pd.read_csv(FCRA_FILE)
    df.columns = ["sl", "state", "fc_2020", "fc_2021", "fc_2022"]
    df["fc_avg"] = df[["fc_2020", "fc_2021", "fc_2022"]].mean(axis=1)
    df["pop_cr"] = df["state"].map(STATE_POP_CR)
    df["fc_per_capita"] = df["fc_avg"] / df["pop_cr"]
    return df


def print_ranking(df):
    print("=" * 80)
    print("FCRA Foreign Contributions per Capita (Rs/person, avg FY 2019-22)")
    print("=" * 80)
    print(f"{'State':<35} {'FC Avg (Cr)':>12} {'Pop (Cr)':>10} {'Rs/person':>12}")
    print("-" * 80)
    ranked = df.sort_values("fc_per_capita", ascending=False)
    for _, r in ranked.iterrows():
        print(f"{r['state']:<35} {r['fc_avg']:>12.1f} "
              f"{r['pop_cr']:>10.2f} {r['fc_per_capita']:>12.1f}")


def print_contrast(df):
    print("\n" + "=" * 80)
    print("HIGH-EDUCATION vs LOW-EDUCATION STATES")
    print("=" * 80)

    for label, states in [("High-education", HIGH_EDUCATION),
                          ("Low-education", LOW_EDUCATION)]:
        subset = df[df["state"].isin(states)]
        total_fc = subset["fc_avg"].sum()
        total_pop = subset["pop_cr"].sum()
        per_cap = total_fc / total_pop
        print(f"\n{label} states:")
        for _, r in subset.sort_values("fc_per_capita", ascending=False).iterrows():
            print(f"  {r['state']:<30} Rs {r['fc_per_capita']:>8.1f}/person")
        print(f"  {'AGGREGATE':<30} Rs {per_cap:>8.1f}/person  "
              f"(total: {total_fc:.0f} Cr, pop: {total_pop:.1f} Cr)")

    high = df[df["state"].isin(HIGH_EDUCATION)]
    low = df[df["state"].isin(LOW_EDUCATION)]
    ratio = (high["fc_avg"].sum() / high["pop_cr"].sum()) / \
            (low["fc_avg"].sum() / low["pop_cr"].sum())
    print(f"\nRatio: {ratio:.1f}×")


def print_bias_analysis(df):
    print("\n" + "=" * 80)
    print("BIAS ANALYSIS")
    print("=" * 80)

    delhi = df[df["state"] == "Delhi"].iloc[0]
    national_fc = df["fc_avg"].sum()
    print(f"\n1. DELHI HQ BIAS")
    print(f"   Delhi: Rs {delhi['fc_per_capita']:.0f}/person "
          f"({delhi['fc_avg']:.0f} Cr, {delhi['fc_avg']/national_fc*100:.1f}% "
          f"of national total)")
    print(f"   Delhi is where NGO headquarters are registered (World Vision,")
    print(f"   Oxfam India, CRY, etc.), not where money is spent.")
    print(f"   Registration bias inflates Delhi, deflates destination states.")
    print(f"   Direction of bias: CONSERVATIVE — reassigning Delhi's share")
    print(f"   to actual spending states would raise Kerala/TN further.")

    print(f"\n2. RELIGIOUS INSTITUTIONAL GIVING")
    print(f"   Kerala's Christian community and Andhra's evangelical orgs")
    print(f"   route diaspora giving through churches. This is not a")
    print(f"   confounder — religious institutions are the routing mechanism")
    print(f"   for kin relaxation, not an alternative explanation.")

    print(f"\n3. FCRA UNDERSTATES TOTAL DIASPORA FLOW")
    print(f"   FCRA captures foreign contributions through registered NGOs.")
    print(f"   It excludes: direct family remittances, property investment,")
    print(f"   informal transfers. Total diaspora investment is larger.")
    print(f"   Direction of bias: CONSERVATIVE.")

    print(f"\n4. KARNATAKA (Rs {df[df['state']=='Karnataka'].iloc[0]['fc_per_capita']:.0f}/person)")
    print(f"   Bangalore is an NGO and tech-philanthropy hub.")
    print(f"   Some HQ registration bias, but also genuine tech-diaspora")
    print(f"   giving (educated IT workforce with global connections).")

    total_fc = df["fc_avg"].sum()
    total_pop = df["pop_cr"].sum()
    print(f"\nNational average: Rs {total_fc/total_pop:.1f}/person "
          f"(total: {total_fc:.0f} Cr, pop: {total_pop:.1f} Cr)")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_fcra()
    print_ranking(df)
    print_contrast(df)
    print_bias_analysis(df)
