from pathlib import Path

import numpy as np
import pandas as pd


def clean_data(source, dataset_type=None, sep=None, min_sector_frequency=3,
               furnishing_clusters=3):
    """Return a cleaned DataFrame from a CSV/TSV path or DataFrame.

    ``dataset_type`` may be ``"flats"`` or ``"houses"``. When omitted it is
    inferred from the columns, so the same call works for another matching CSV.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if sep is None:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)

    if dataset_type is None:
        dataset_type = "houses" if "noOfFloor" in df.columns else "flats"
    dataset_type = dataset_type.lower().rstrip("s")
    if dataset_type not in {"flat", "house"}:
        raise ValueError("dataset_type must be 'flats' or 'houses'")

    # Shared parsing helpers kept inline so this is the only function users
    # need to call.
    price_text = df.get("price", pd.Series(index=df.index, dtype="string")).astype("string").str.strip()
    price = pd.to_numeric(price_text.str.extract(r"([\d.]+)")[0], errors="coerce")
    price = price.where(~price_text.str.contains(r"\b(?:lac|lakh)\b", case=False, na=False), price / 100)
    df["price"] = price.mask(price_text.str.casefold().isin({"price on request", "price", ""}))

    if dataset_type == "house":
        df = df.drop_duplicates().copy()
        df.drop(columns=["link", "property_id"], errors="ignore", inplace=True)
        df.rename(columns={"rate": "price_per_sqft", "noOfFloor": "floorNum"}, inplace=True)
        if "society" in df:
            df["society"] = (df["society"].astype("string")
                .str.replace(r"\d+(\.\d+)?\s?★", "", regex=True).str.strip().str.lower()
                .replace({"nan": "independent", "<NA>": "independent"}).fillna("independent"))
        if "price_per_sqft" in df:
            df["price_per_sqft"] = pd.to_numeric(
                df["price_per_sqft"].astype("string").str.split("/").str[0]
                .str.replace("₹", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
        if "bedRoom" in df:
            df = df[df["bedRoom"].notna()].copy()
            df["bedRoom"] = pd.to_numeric(df["bedRoom"].astype("string").str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
        for column in ["bathroom", "balcony"]:
            if column in df:
                values = df[column].astype("string").str.replace("No", "0", regex=False)
                df[column] = pd.to_numeric(values.str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
        if "additionalRoom" in df:
            df["additionalRoom"] = df["additionalRoom"].fillna("not available").astype("string").str.lower()
        if "floorNum" in df:
            df["floorNum"] = df["floorNum"].astype("string").str.extract(r"(\d+)")[0]
        if "facing" in df:
            df["facing"] = df["facing"].fillna("NA")
        if {"price", "price_per_sqft"}.issubset(df.columns):
            df["area"] = ((df["price"] * 10_000_000) / df["price_per_sqft"]).round()
        df.insert(1, "property_type", "house")
        if "additionalRoom" in df:
            for room in ["Pooja Room", "Study Room", "Store Room", "Servant Room", "Others"]:
                df[room] = df["additionalRoom"].str.contains(room, case=False, na=False).astype("int8")
            df.drop(columns="additionalRoom", inplace=True)
        if "property_name" in df:
            parts = df["property_name"].astype("string").str.extract(r"^(.*?)\s+in\s+(.*)$")
            prefix = parts[0].fillna("").str.strip()
            df["Size"] = pd.to_numeric(prefix.str.extract(r"^(\d+)")[0], errors="coerce").astype("Int64")
            df["Type"] = prefix.str.replace(r"^\d+\s*Bedroom\s*", "", regex=True).str.replace(r"\s+for sale$", "", regex=True).str.strip()
            df["sector"] = parts[1].fillna("").str.replace("Gurgaon", "", regex=False).str.strip().str.lower()
            df.drop(columns="property_name", inplace=True)
        if "agePossession" in df:
            age_map = {"1 to 5 Year Old":"Medium Aged", "0 to 1 Year Old":"New", "5 to 10 Year Old":"Old", "10+ Year Old":"Very Old"}
            mapped = df["agePossession"].map(age_map)
            fallback = pd.Series(np.where(df["agePossession"].astype("string").str.contains(r"Under Construction|Within 6 months|Within 3 months|202", na=False), "Under Construction", "Undefined"), index=df.index)
            df["agePossession"] = mapped.where(mapped.notna(), fallback)
        if "furnishDetails" in df:
            details = ["Fan", "Light", "Wardrobe", "AC", "Bed", "Chimney", "Curtains", "Dining Table", "Exhaust Fan", "Geyser", "Modular Kitchen", "Microwave", "Fridge", "Sofa", "Stove", "TV", "Washing Machine", "Water Purifier"]
            for detail in details:
                df[detail] = (~df["furnishDetails"].astype("string").str.contains("No " + detail, na=False)).astype("int8")
            if furnishing_clusters is not None and len(df) >= furnishing_clusters:
                from sklearn.cluster import KMeans
                df["furnishstatus"] = KMeans(n_clusters=furnishing_clusters, random_state=42, n_init=10).fit_predict(df[details])
            df.drop(columns=["furnishDetails", *details], inplace=True, errors="ignore")
        common_columns = ["property_type", "society", "price", "price_per_sqft", "sector", "area", "areaWithType", "bedRoom", "bathroom", "balcony", "address", "floorNum", "facing", "agePossession", "nearbyLocations", "description", "features", "rating", "Pooja Room", "Study Room", "Store Room", "Servant Room", "Others", "Size", "Type", "furnishstatus"]
        for column in common_columns:
            if column not in df:
                df[column] = pd.NA
        return df[common_columns].reset_index(drop=True)

    # Flats-specific cleaning.
    df.drop(columns=["link", "property_id"], errors="ignore", inplace=True)
    df.insert(1, "property_type", "flat")
    if "area" in df:
        area = df["area"].astype("string").str.split("/").str[0]
        df["area"] = pd.to_numeric(area.str.replace("₹", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
        df = df[df["area"].notna()].copy()
    if "bedRoom" in df:
        df = df[df["bedRoom"].notna()].copy()
        df["bedRoom"] = pd.to_numeric(df["bedRoom"].astype("string").str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    for column in ["bathroom", "balcony"]:
        if column in df:
            df[column] = pd.to_numeric(df[column].astype("string").str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    if "additionalRoom" in df:
        df["additionalRoom"] = df["additionalRoom"].fillna("Not Available").astype("string")
        for room in ["Pooja Room", "Study Room", "Store Room", "Servant Room", "Others"]:
            df[room] = df["additionalRoom"].str.contains(room, case=False, na=False).astype("int8")
        df.drop(columns="additionalRoom", inplace=True)

    if "property_name" in df:
        parts = df["property_name"].astype("string").str.extract(r"^(.*?)\s+in\s+(.*)$")
        prefix = parts[0].fillna("").str.strip()
        df["Size"] = pd.to_numeric(prefix.str.extract(r"^(\d+)")[0], errors="coerce").astype("Int64")
        df["Type"] = prefix.str.replace(r"^\d+\s*BHK\s*", "", regex=True).str.strip()
        df.insert(3, "sector", parts[1].fillna("").str.replace("Gurgaon", "", regex=False).str.strip().str.lower())
        sector_map = {
            "dharam colony":"sector 12", "krishna colony":"sector 7", "suncity":"sector 54", "prem nagar":"sector 13", "mg road":"sector 28", "gandhi nagar":"sector 28", "laxmi garden":"sector 11", "shakti nagar":"sector 11", "baldev nagar":"sector 7", "shivpuri":"sector 7", "garhi harsaru":"sector 17", "imt manesar":"manesar", "adarsh nagar":"sector 12", "shivaji nagar":"sector 11", "bhim nagar":"sector 6", "madanpuri":"sector 7", "saraswati vihar":"sector 28", "arjun nagar":"sector 8", "ravi nagar":"sector 9", "vishnu garden":"sector 105", "bhondsi":"sector 11", "surya vihar":"sector 21", "devilal colony":"sector 9", "valley view estate":"gwal pahari", "mehrauli  road":"sector 14", "jyoti park":"sector 7", "ansal plaza":"sector 23", "dayanand colony":"sector 6", "sushant lok phase 2":"sector 55", "chakkarpur":"sector 28", "greenwood city":"sector 45", "subhash nagar":"sector 12", "sohna road road":"sohna road", "malibu town":"sector 47", "surat nagar 1":"sector 104", "new colony":"sector 7", "mianwali colony":"sector 12", "jacobpura":"sector 12", "rajiv nagar":"sector 13", "ashok vihar":"sector 3", "dlf phase 1":"sector 26", "nirvana country":"sector 50", "palam vihar":"sector 2", "dlf phase 2":"sector 25", "sushant lok phase 1":"sector 43", "laxman vihar":"sector 4", "dlf phase 4":"sector 28", "dlf phase 3":"sector 24", "sushant lok phase 3":"sector 57", "dlf phase 5":"sector 43", "rajendra park":"sector 105", "uppals southend":"sector 49", "sohna":"sohna road", "ashok vihar phase 3 extension":"sector 5", "south city 1":"sector 41", "ashok vihar phase 2":"sector 5", "sector 95a":"sector 95", "sector 23a":"sector 23", "sector 12a":"sector 12", "sector 3a":"sector 3", "sector 110 a":"sector 110", "patel nagar":"sector 15", "a block sector 43":"sector 43", "maruti kunj":"sector 12", "b block sector 43":"sector 43", "sector-33 sohna road":"sector 33", "sector 1 manesar":"manesar", "sector 4 phase 2":"sector 4", "sector 1a manesar":"manesar", "c block sector 43":"sector 43", "sector 89 a":"sector 89", "sector 2 extension":"sector 2", "sector 36 sohna road":"sector 36"
        }
        for old, new in sorted(sector_map.items(), key=lambda item: len(item[0]), reverse=True):
            df["sector"] = df["sector"].str.replace(old, new, regex=False)
        counts = df["sector"].value_counts()
        df = df[df["sector"].isin(counts[counts >= min_sector_frequency].index)].copy()
        df.drop(columns="property_name", inplace=True)

    if "society" in df:
        df["society"] = df["society"].astype("string").str.split(r"\d").str[0].str.strip()
    if "agePossession" in df:
        age_map = {"1 to 5 Year Old":"Medium Aged", "0 to 1 Year Old":"New", "5 to 10 Year Old":"Old", "10+ Year Old":"Very Old"}
        mapped = df["agePossession"].map(age_map)
        fallback = pd.Series(np.where(df["agePossession"].astype("string").str.contains(r"Under Construction|Within 6 months|Within 3 months|202", na=False), "Under Construction", "Undefined"), index=df.index)
        df["agePossession"] = mapped.where(mapped.notna(), fallback)
    if "furnishDetails" in df:
        details = ["Fan", "Light", "Wardrobe", "AC", "Bed", "Chimney", "Curtains", "Dining Table", "Exhaust Fan", "Geyser", "Modular Kitchen", "Microwave", "Fridge", "Sofa", "Stove", "TV", "Washing Machine", "Water Purifier"]
        for detail in details:
            df[detail] = (~df["furnishDetails"].astype("string").str.contains("No " + detail, na=False)).astype("int8")
        if furnishing_clusters is not None and len(df) >= furnishing_clusters:
            from sklearn.cluster import KMeans
            df["furnishstatus"] = KMeans(n_clusters=furnishing_clusters, random_state=42, n_init=10).fit_predict(df[details])
        # Detail flags are temporary KMeans inputs; keep only the cluster label.
        df.drop(columns=["furnishDetails", *details], inplace=True, errors="ignore")
    if {"price", "area"}.issubset(df.columns):
        df["area"] = (df["price"] * 10_000_000 / df["area"]).round(2)
    if "floorNum" in df:
        address = df.get("address", pd.Series(index=df.index, dtype="string")).astype("string")
        missing = df["floorNum"].isna()
        extracted = address.str.extract(r"(\d+)(?:st|nd|rd|th)")[0]
        df.loc[missing, "floorNum"] = extracted[missing]
        floor = df["floorNum"].astype("string").str.replace("Ground", "0", regex=False).str.replace("Lower", "0", regex=False).str.replace("Basement", "-1", regex=False)
        df["floorNum"] = pd.to_numeric(floor.str.extract(r"(-?\d+)")[0], errors="coerce")
    if "facing" in df:
        df["facing"] = df["facing"].fillna("NA")
    common_columns = ["property_type", "society", "price", "price_per_sqft", "sector", "area", "areaWithType", "bedRoom", "bathroom", "balcony", "address", "floorNum", "facing", "agePossession", "nearbyLocations", "description", "features", "rating", "Pooja Room", "Study Room", "Store Room", "Servant Room", "Others", "Size", "Type", "furnishstatus"]
    for column in common_columns:
        if column not in df:
            df[column] = pd.NA
    return df[common_columns].reset_index(drop=True)


df=clean_data(Path('data/raw/house.csv'))
df1=clean_data(Path('data/raw/flats - flats.csv'))

final_df=pd.concat((df,df1),ignore_index=True)

final_df=final_df.sample(final_df.shape[0],ignore_index=True)
final_df.to_csv('data/Cleaned/gurgaon_Properties.csv',index=False)