from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CLEAN_PATH = ROOT / "data" / "Cleaned" / "property_outlier_treated.csv"
SOURCE_PATH = ROOT / "data" / "Cleaned" / "property.csv"
REPORT_PATH = ROOT / "data" / "Cleaned" / "sector_normalization_report.csv"
SECTOR_RE = re.compile(r"\bsector[- ]?(\d{1,3}[a-z]?)\b", re.I)

# Verified/common canonical labels for broad Gurgaon localities.  A broad
# locality may cover multiple sectors; this gives the model one stable label.
CANONICAL = {
    "sohna road": "sector 33", "sohna road road": "sector 33", "sohna  road": "sector 33",
    "sohna": "sector 36", "dlf phase 1": "sector 26", "dlf phase 2": "sector 25",
    "dlf phase 3": "sector 24", "dlf phase 4": "sector 28", "dlf phase 5": "sector 54",
    "nirvana country": "sector 50", "palam vihar": "sector 2", "palam vihar extension": "sector 2",
    "new palam vihar": "sector 2", "new palam vihar phase 1": "sector 2",
    "new palam vihar phase 2": "sector 2", "ansal's palam vihar": "sector 2",
    "sushant lok phase 1": "sector 43", "c block sushant lok phase 1": "sector 43",
    "a block sushant lok phase 1": "sector 43", "b block sushant lok phase 1": "sector 43",
    "sushant lok phase 2": "sector 55", "sushant lok phase 3": "sector 57",
    "gwal pahari": "sector 54", "laxman vihar": "sector 3a", "laxman vihar phase 2": "sector 3a",
    "laxmna vihar, phase 2": "sector 3a", "rajendra park": "sector 105",
    "uppals southend": "sector 49", "south city 1": "sector 41", "south city": "sector 41",
    "mianwali colony": "sector 12a", "manesar": "sector 1", "hsiidc plot sector 1 imt manesar": "sector 1",
    "ashok vihar": "sector 5", "ashok vihar phase 2": "sector 5", "ashok vihar phase 3": "sector 5",
    "ashok vihar phase 3 extension": "sector 5", "ashok vihar phase i": "sector 5",
    "block h ashok vihar phase iii extension": "sector 5", "ansal plaza": "sector 23",
    "greenwood city": "sector 45", "jacobpura": "sector 12", "jacobpura near sadar bazar": "sector 12",
    "rajiv nagar": "sector 13", "devilal colony": "sector 9", "new colony": "sector 7",
    "jyoti park": "sector 7", "arjun nagar": "sector 8", "baldev nagar": "sector 9",
    "ravi nagar": "sector 9", "ravi nager basai road": "sector 9", "prem nagar": "sector 11",
    "surat nagar": "sector 104", "surat nagar 1": "sector 104", "surat nagar phase 2": "sector 104",
    "subhash nagar": "sector 12a", "gandhi nagar": "sector 11", "patel nagar": "sector 15",
    "hira nagar": "sector 9", "vir nagar": "sector 9", "shakti nagar": "sector 11",
    "park view": "sector 67", "urban estate, sector 5": "sector 5", "part 3 sector 5": "sector 5",
    "b1 block sector 57": "sector 57", "block g sector-57": "sector 57",
    "anand garden, sector-105": "sector 105", "bhim nagar": "sector 6",
    "saraswati vihar": "sector 28", "huda saraswati vihar": "sector 28",
    "new sector 2": "sector 2", "farukhnagar": "sector 3", "suncity": "sector 54",
    "maruti kunj": "sector 55", "bhawani enclave": "sector 55", "sheetla colony": "sector 5",
    "golf course road": "sector 42", "rajiv chowk": "sector 12", "near euro international school": "sector 10",
    # Remaining localities were checked against Gurgaon address listings and
    # normalized to their commonly used nearest sector.
    "dwarka expressway": "sector 102", "new": "sector 90", "madanpuri": "sector 7",
    "surya vihar": "sector 21", "malibu town": "sector 47", "shivpuri": "sector 7",
    "bhondsi": "sector 68", "dharam colony": "sector 39", "new basti": "sector 8",
    "krishna colony": "sector 7", "acharya puri": "sector 7", "laxmi garden": "sector 11",
    "rosewood": "sector 50", "dayanand colony": "sector 4", "garhi harsaru": "sector 88",
    "mayfield garden": "sector 47", "bissar": "sector 70", "pratap nagar": "sector 8",
    "narsinghpur": "sector 36", "rattan garden": "sector 7", "bhimgarh kheri phase 3": "sector 3",
    "sec 3 bhim vihar part 1 bhimgher kheri": "sector 3", "shivji park colony": "sector 7",
    "damdma": "sector 59", "dhankot": "sector 104", "shivaji nagar": "sector 11",
    "g block dlf city phase 1": "sector 26", "v block dlf phase 3": "sector 24",
    "begampur khatola": "sector 74", "west rajiv nagar": "sector 13", "sadar bazar": "sector 12",
    "tekchand nagar": "sector 104", "mahavir pura": "sector 5", "new jyoti park": "sector 7",
    "devi lal colony gali no 12 gurgaon nr omkar public school": "sector 9",
    "saraswati enclave": "sector 5", "manohar nagar": "sector 7", "hno 13 sagar enclave sura": "sector 6",
    "om nagar": "sector 5", "ram nagar": "sector 7", "daulatabad": "sector 104", "jat colony,fazilpur/taj nagar road": "sector 75",
    "laxman vihar, railway road,": "sector 3a", "kheri": "sector 3", "bhora kalan": "sector 90",
}


def address_sector_candidates(source: pd.DataFrame, row: pd.Series) -> list[str]:
    q = source[
        (source["sector"].fillna("").astype(str).str.lower() == str(row["sector"]).lower())
        & (source["property_type"] == row["property_type"])
        & np.isclose(source["price"], row["price"])
    ].copy()
    if q.empty:
        return []
    q["distance"] = (q["area"] - row["area"]).abs()
    addresses = q.sort_values("distance").head(8)["address"].fillna("")
    found = []
    for address in addresses:
        found.extend(f"sector {value.lower()}" for value in SECTOR_RE.findall(address))
    return list(dict.fromkeys(found))


def main() -> None:
    data = pd.read_csv(CLEAN_PATH)
    source = pd.read_csv(SOURCE_PATH)
    original = data["sector"].fillna("").astype(str).str.strip().str.lower()
    report = []
    changed = 0
    for index, value in original.items():
        if re.match(r"^sector[- ]", value):
            continue
        candidates = address_sector_candidates(source, data.loc[index])
        mapped = candidates[0] if len(candidates) == 1 else CANONICAL.get(value)
        if mapped:
            report.append({"row": index, "original": data.loc[index, "sector"], "replacement": mapped,
                           "source": "listing address" if len(candidates) == 1 else "canonical locality map"})
            data.loc[index, "sector"] = mapped
            changed += 1
    data.to_csv(CLEAN_PATH, index=False)
    pd.DataFrame(report).to_csv(REPORT_PATH, index=False)
    print(f"Normalized {changed} rows")
    print(f"Still non-sector rows: {int((~data.sector.astype(str).str.match(r'^sector[- ]', case=False)).sum())}")
    print(f"Audit report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
