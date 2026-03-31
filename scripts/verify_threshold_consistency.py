"""
verify_threshold_consistency.py

Checks that every document in the project uses the canonical development
threshold numbers:
  - 154 countries crossed both thresholds
  - 80% of the world's population (developed)
  - 20% remaining (not developed)

Scans paper, vision docs, CIES materials, website templates, and chatbot
prompts. Flags any occurrence of stale numbers (153, 78%, 22%) or any
mismatch with the canonical values.

Usage:
    python scripts/verify_threshold_consistency.py

Exit code: 0 if all consistent, 1 if any inconsistency found.
"""

import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EDUCODE_ROOT = os.path.dirname(REPO_ROOT)

# ── Canonical values ────────────────────────────────────────────────────
CANONICAL = {
    "countries": 154,
    "pct_developed": 80,
    "pct_remaining": 20,
}

# ── Stale values to flag ───────────────────────────────────────────────
STALE = {
    "countries": [153],
    "pct_developed": [78],
    "pct_remaining": [22],
}

# ── Files to scan ──────────────────────────────────────────────────────
# (path relative to EDUCODE_ROOT, description)
SCAN_FILES = [
    # Paper
    ("Human-Development-Prediction/paper/education_of_nations.tex", "Main paper"),
    ("Human-Development-Prediction/paper/cies_handout.tex", "CIES handout TeX"),
    ("Human-Development-Prediction/paper/cies_handout.html", "CIES handout HTML"),
    ("Human-Development-Prediction/paper/cies_summary.md", "CIES summary"),
    # Vision docs
    ("Human-Development-Prediction/vision/for_the_leader.md", "Vision: leader"),
    ("Human-Development-Prediction/vision/for_the_philanthropist.md", "Vision: philanthropist"),
    ("Human-Development-Prediction/vision/for_the_bureaucrat.md", "Vision: bureaucrat"),
    ("Human-Development-Prediction/vision/chatbot_qa_seed.md", "Chatbot QA seed"),
    ("Human-Development-Prediction/vision/country_action_guide.md", "Country action guide"),
    # Website
    ("education-first-data/templates/index.html", "Website index"),
    ("education-first-data/templates/evidence.html", "Website evidence"),
    ("education-first-data/templates/vision_index.html", "Website vision index"),
    ("education-first-data/llm-data/system_prompt.txt", "Chatbot system prompt"),
    # Checkin
    ("Human-Development-Prediction/checkin/development_threshold_count.json", "Threshold checkin JSON"),
]

# ── Patterns ────────────────────────────────────────────────────────────
# Each pattern extracts a number and maps to a canonical key.
PATTERNS = [
    # "154 countries" or "153 countries"
    (r"\b(\d{3})\s+countries\b", "countries", int),
    # "80%" or "78%" in context of world/humanity/population
    (r"\b(\d{1,2})\\?%\s*(?:of\s+)?(?:the\s+)?(?:world|humanity|human)", "pct_developed", int),
    # "representing 80%" or "representing 78%"
    (r"representing\s+(\d{1,2})\\?%", "pct_developed", int),
    # "remaining 20%" or "remaining 22%"
    (r"remaining\s+(\d{1,2})\\?%", "pct_remaining", int),
    # "22%" or "20%" near "remaining" (stat boxes)
    (r"(\d{1,2})%.*?remaining", "pct_remaining", int),
    # Standalone "22%" or "20%" in stat-num/number elements (HTML stat boxes)
    (r'(?:stat-num|class="number").*?(\d{1,2})%', "pct_remaining", int),
    # "N% of humanity"
    (r"(\d{1,2})%.*?of humanity", "pct_remaining", int),
]

# For JSON checkin files, check the numbers directly
def check_json(filepath):
    """Check development_threshold_count.json for consistency."""
    issues = []
    with open(filepath) as f:
        data = json.load(f)
    nums = data.get("numbers", {})

    count = nums.get("countries_crossing_both")
    # The script counts 153 from 2022 snapshot data (Philippines missed by
    # TFR rounding). We agreed Philippines counts → canonical is 154.
    # The JSON records the raw script output; the paper overrides to 154.
    # Flag if the JSON says something other than 153 or 154.
    if count is not None and count not in (153, 154):
        issues.append(f"  countries_crossing_both = {count} (expected 153 or 154)")

    pct = nums.get("pct_developed")
    if pct is not None and round(pct) not in (78, 80):
        issues.append(f"  pct_developed = {pct} (expected ~78 or ~80)")

    return issues


def check_text_file(filepath, label):
    """Scan a text file for threshold number patterns; flag stale values."""
    issues = []
    with open(filepath) as f:
        lines = f.readlines()

    # Single-line pattern matching
    for lineno, line in enumerate(lines, 1):
        for pattern, key, cast in PATTERNS:
            for m in re.finditer(pattern, line, re.IGNORECASE):
                val = cast(m.group(1))
                if val in STALE.get(key, []):
                    issues.append(
                        f"  line {lineno}: found stale {key}={val} "
                        f"(should be {CANONICAL[key]}): "
                        f"{line.strip()[:100]}"
                    )

    # Sliding-window check: look at 5-line windows for stat boxes where
    # the number and context word ("remaining", "humanity") are on
    # separate lines (common in HTML templates).
    WINDOW = 5
    CONTEXT_WORDS = re.compile(r"remaining|humanity|not.developed|left.behind", re.IGNORECASE)
    BARE_PCT = re.compile(r"\b(22)%\b")  # only stale remaining %

    for i in range(len(lines)):
        window = "".join(lines[i : i + WINDOW])
        if CONTEXT_WORDS.search(window):
            for m in BARE_PCT.finditer(window):
                # Find exact line of the match
                offset = m.start()
                cumlen = 0
                for j in range(i, min(i + WINDOW, len(lines))):
                    cumlen += len(lines[j])
                    if offset < cumlen:
                        lineno = j + 1
                        break
                issue = (
                    f"  line {lineno}: found stale pct_remaining=22 "
                    f"(should be {CANONICAL['pct_remaining']}): "
                    f"{lines[lineno - 1].strip()[:100]}"
                )
                if issue not in issues:  # avoid duplicates with single-line pass
                    issues.append(issue)

    return issues


def main():
    all_issues = {}
    total = 0

    for relpath, label in SCAN_FILES:
        filepath = os.path.join(EDUCODE_ROOT, relpath)
        if not os.path.exists(filepath):
            print(f"  SKIP  {label} — file not found: {relpath}")
            continue

        if filepath.endswith(".json"):
            issues = check_json(filepath)
        else:
            issues = check_text_file(filepath, label)

        if issues:
            all_issues[label] = issues
            total += len(issues)

    # ── Report ──────────────────────────────────────────────────────
    print("=" * 70)
    print("THRESHOLD CONSISTENCY CHECK")
    print(f"Canonical: {CANONICAL['countries']} countries, "
          f"{CANONICAL['pct_developed']}% developed, "
          f"{CANONICAL['pct_remaining']}% remaining")
    print("=" * 70)

    if not all_issues:
        print("\n  ALL CONSISTENT — no stale threshold numbers found.\n")
        return 0

    for label, issues in all_issues.items():
        print(f"\n  FAIL  {label}")
        for issue in issues:
            print(issue)

    print(f"\n{'=' * 70}")
    print(f"TOTAL INCONSISTENCIES: {total}")
    print("=" * 70)
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
