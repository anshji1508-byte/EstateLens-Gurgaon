from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Cleaned" / "property_outlier_treated.csv"

# These are unambiguous Delhi/NCR/other-city locality markers.  Generic names
# such as Gandhi Nagar or Patel Nagar are intentionally not included because
# they also occur as valid Gurgaon localities.
NON_GURGAON_MARKERS = (
    "sector 21a faridabad",
    "sonipat",
    "lucknow",
    "delhi ncr",
    "rajendra place",
    "lajpat nagar",
    "vishnu garden",
    "alipur",
    "mohan nagar",
)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    locality = data["sector"].fillna("").astype(str).str.strip().str.lower()
    bad = locality.str.contains("|".join(NON_GURGAON_MARKERS), regex=True, na=False)
    removed = data.loc[bad, "sector"].value_counts().to_dict()
    data = data.loc[~bad].reset_index(drop=True)
    data.to_csv(DATA_PATH, index=False)
    print(f"Removed {int(bad.sum())} non-Gurgaon rows")
    print("Removed localities:", removed)
    print(f"Remaining rows: {len(data)}")


if __name__ == "__main__":
    main()
