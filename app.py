from __future__ import annotations

import pickle
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
try:
    from wordcloud import WordCloud
except ModuleNotFoundError:  # Optional visual enhancement; app still works without it.
    WordCloud = None
from train_model import make_features


ROOT = Path(__file__).resolve().parent
with (ROOT / "Pkles" / "property_price_model.pkl").open("rb") as file:
    bundle = pickle.load(file)
model = bundle["model"]
metrics = bundle["metrics"]
data = pd.read_csv(ROOT / "data" / "Cleaned" / "property_outlier_treated.csv")
raw_data = pd.read_csv(ROOT / "data" / "Cleaned" / "property.csv")
estate_data = pd.read_csv(ROOT / "data" / "raw" / "estate.csv")

st.set_page_config(page_title="EstateLens Gurgaon", page_icon="🏢", layout="wide")
st.markdown("""
<style>
.stApp {background:radial-gradient(circle at 0% 0%, #2a1015 0, #120b12 35%, #080709 100%); color:#fff7ed;}
.block-container {padding:2.5rem 3.2rem 4rem; max-width:1500px; animation:pageIn .55s ease-out both;}
@keyframes pageIn {from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);}}
.hero {position:relative; overflow:hidden; padding:2.8rem 3.2rem; border-radius:30px; background:linear-gradient(125deg,#160708 0%,#5b1018 48%,#d97706 100%); color:white; margin-bottom:1.8rem; box-shadow:0 20px 55px rgba(220,38,38,.26);}
.hero:after {content:""; position:absolute; width:260px; height:260px; border-radius:50%; right:-65px; top:-95px; background:rgba(255,255,255,.10); box-shadow:-80px 150px 0 rgba(255,255,255,.05);}
.hero:after {animation:floatOrb 7s ease-in-out infinite alternate;}
@keyframes floatOrb {from {transform:translate(0,0) scale(1);} to {transform:translate(-25px,18px) scale(1.12);}}
.hero h1 {font-size:3rem; letter-spacing:-.05em; margin:0 0 .55rem 0; position:relative; z-index:1;}
.hero p {font-size:1.12rem; opacity:.86; margin:0; position:relative; z-index:1;}
.hero:before {content:"LIVE MARKET ESTIMATE  •  GURUGRAM"; display:inline-block; font-size:.7rem; letter-spacing:.14em; font-weight:700; color:#fde68a; margin-bottom:1rem; position:relative; z-index:1;}
[data-testid="stTabs"] {background:rgba(255,255,255,.72); border:1px solid rgba(148,163,184,.25); border-radius:24px; padding:1rem 1.2rem 1.8rem; box-shadow:0 12px 35px rgba(15,23,42,.06);}
[data-testid="stTabs"] button {font-weight:700; color:#64748b; border-radius:12px; padding:.7rem 1.25rem;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#f59e0b; background:#451a03;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#171a31,#111426); border:1px solid #2c3153; padding:1.15rem 1.25rem; border-radius:18px; box-shadow:0 8px 22px rgba(0,0,0,.22);}
[data-testid="stMetric"] {transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease;}
[data-testid="stMetric"]:hover {transform:translateY(-5px); border-color:#fbbf24; box-shadow:0 15px 32px rgba(245,158,11,.16);}
[data-testid="stMetricLabel"] {color:#64748b; font-weight:650;}
[data-testid="stMetricValue"] {color:#fbbf24; font-weight:800;}
div[data-testid="stForm"] {background:rgba(17,20,38,.86); border:1px solid #2b3155; border-radius:22px; padding:1.4rem 1.3rem .5rem; box-shadow:0 10px 30px rgba(0,0,0,.24);}
div[data-testid="stForm"] {animation:cardIn .65s .08s ease-out both;}
@keyframes cardIn {from {opacity:0; transform:translateY(14px) scale(.99);} to {opacity:1; transform:translateY(0) scale(1);}}
label {font-weight:650 !important; color:#d8d5ed !important;}
div[data-baseweb="select"] > div, input {border-radius:11px !important; border-color:#373b63 !important; background:#15182a !important; color:#f8fafc !important;}
div[data-baseweb="select"] > div:focus-within, input:focus {border-color:#fbbf24 !important; box-shadow:0 0 0 2px rgba(251,191,36,.14) !important;}
div[data-baseweb="select"] input[aria-label^="Selected "] {width:2px !important; min-width:2px !important; padding:0 !important; border:0 !important; border-radius:0 !important; background:transparent !important; box-shadow:none !important;}
div[data-baseweb="select"] input[aria-label^="Selected "]:focus {border:0 !important; background:transparent !important; box-shadow:none !important;}
button[kind="primaryFormSubmit"], button[kind="primary"] {border-radius:13px !important; background:linear-gradient(100deg,#b91c1c,#f59e0b) !important; border:0 !important; box-shadow:0 8px 18px rgba(220,38,38,.24); font-weight:800 !important;}
button[kind="primaryFormSubmit"]:hover, button[kind="primary"]:hover {background:linear-gradient(100deg,#991b1b,#d97706) !important; transform:translateY(-1px);}
.stAlert {border-radius:15px; border:0; box-shadow:0 8px 20px rgba(0,0,0,.2);}
.stPlotlyChart {background:#111426; border-radius:18px; padding:.35rem; box-shadow:0 8px 25px rgba(0,0,0,.22);}
.stPlotlyChart {transition:transform .3s ease, box-shadow .3s ease;}
.stPlotlyChart:hover {transform:translateY(-3px); box-shadow:0 18px 38px rgba(220,38,38,.14);}
h2, h3 {letter-spacing:-.025em; color:#f5f3ff;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#160708 0%,#3b0d16 55%,#65151d 100%); border-right:1px solid rgba(255,255,255,.08);}
[data-testid="stSidebar"] * {color:#e5f7f5 !important;}
[data-testid="stSidebar"] .stRadio label {padding:.65rem .75rem; border-radius:10px; font-weight:650;}
[data-testid="stSidebar"] .stRadio label:hover {background:rgba(255,255,255,.10);}
.side-brand {animation:brandGlow 3s ease-in-out infinite alternate;}
@keyframes brandGlow {from {text-shadow:0 0 0 rgba(251,191,36,0);} to {text-shadow:0 0 18px rgba(251,191,36,.45);}}
.side-brand {display:flex; align-items:center; gap:.65rem; font-size:1.25rem; font-weight:850; letter-spacing:-.04em; margin:.5rem 0 .2rem;}
.brand-mark {display:inline-flex; align-items:center; justify-content:center; width:36px; height:36px; border-radius:11px; background:linear-gradient(145deg,#fbbf24,#b91c1c); box-shadow:0 7px 18px rgba(245,158,11,.28);}
.side-sub {font-size:.78rem; color:#fde68a !important; margin-bottom:2rem;}
.section-kicker {color:#fbbf24; font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.5rem;}
.welcome-card {background:linear-gradient(120deg,#211015,#301017); border:1px solid #6b2732; border-radius:22px; padding:1.4rem 1.6rem; margin-bottom:1.2rem; box-shadow:0 10px 30px rgba(0,0,0,.22);}
.welcome-card p {color:#aaa7c3 !important;}
.welcome-card {transition:transform .3s ease, border-color .3s ease, box-shadow .3s ease;}
.welcome-card:hover {transform:translateY(-3px); border-color:#fbbf24; box-shadow:0 16px 34px rgba(245,158,11,.14);}
@media (prefers-reduced-motion: reduce) {*, *:before, *:after {animation:none !important; transition:none !important;}}
</style>
<div class="hero"><h1>EstateLens Gurgaon</h1><p>Professional property intelligence and explainable valuation for Gurugram.</p></div>
""", unsafe_allow_html=True)


def options(column, fallback):
    values = sorted(data[column].dropna().astype(str).unique().tolist())
    return values or fallback


def form_frame(values):
    return pd.DataFrame([values])


def sector_map_key(value):
    key = re.sub(r"^sector[- ]*", "", str(value).lower()).strip()
    key = re.sub(r"\s+", " ", key)
    aliases = {
        "15-i": "15 part 1",
        "15-ii": "15 part 2",
        "1 manesar": "m1",
        "1a manesar": "m1a",
    }
    return aliases.get(key, key)


def sector_boundary_geojson(path):
    """Load combined GMDA/DTCP and OSM boundaries as sector multipolygons."""
    with path.open(encoding="utf-8") as file:
        source = json.load(file)

    grouped = {}
    for feature in source["features"]:
        properties = feature["properties"]
        name = properties.get("sector") or properties.get("name") or properties.get("displayName", "")
        key = sector_map_key(name)
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry.get("type") == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        grouped.setdefault(key, []).extend(polygons)

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": key,
                "properties": {"sector_key": key},
                "geometry": {"type": "MultiPolygon", "coordinates": polygons},
            }
            for key, polygons in grouped.items()
        ],
    }


def amenity_text(property_type="All types", sector="All sectors", bedrooms=None):
    source = raw_data.copy()
    amenity_labels = {
        "Club house / Community Center": "Club House",
        "Feng Shui / Vaastu Compliant": "Feng Shui and Vaastu Compliant",
        "Fitness Centre / GYM": "Fitness Centre",
        "Internet/wi-fi connectivity": "Internet and Wi-Fi Connectivity",
        "Private Garden / Terrace": "Private Garden or Terrace",
        "Security / Fire Alarm": "Security and Fire Alarm",
    }
    if property_type != "All types":
        source = source[source.property_type == property_type]
    if sector != "All sectors":
        source = source[source.sector == sector]
    if bedrooms:
        source = source[source.bedRoom.isin(bedrooms)]
    values = source.get("features", pd.Series(dtype=str)).dropna().astype(str)
    frequencies = {}
    for value in values:
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                for item in parsed:
                    amenity = amenity_labels.get(str(item), str(item).replace("/", " ")).strip()
                    if amenity:
                        frequencies[amenity] = frequencies.get(amenity, 0) + 1
        except (ValueError, SyntaxError):
            continue
    return frequencies


def shap_explanation(row):
    """Return top SHAP contributions for the current prediction."""
    try:
        pre = model.named_steps["preprocessor"]
        estimator = model.named_steps["regressor"]
        transformed = pre.transform(make_features(form_frame(row)))
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)[0]
        names = pre.get_feature_names_out()
        result = pd.DataFrame({"feature": names, "impact": values})
        result["feature"] = result["feature"].str.replace(r"^(categorical|numeric)__", "", regex=True)
        return result.reindex(result.impact.abs().sort_values(ascending=False).index).head(8)
    except Exception:
        return pd.DataFrame(columns=["feature", "impact"])


def parse_estate_value(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def estate_proximity(value):
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*(km|kms|kilometers?|m|meters?|mins?|minutes?)\b", text)
    if not match:
        return ("text", text) if text else None
    amount = float(match.group(1).replace(",", ""))
    unit = match.group(2)
    if unit.startswith("m") and not unit.startswith("min") and not unit.startswith("minute"):
        return "distance", amount / 1000
    if unit.startswith("km") or unit.startswith("kilometer"):
        return "distance", amount
    return "time", amount


def estate_profile(listing):
    price_details = parse_estate_value(listing.get("PriceDetails"))
    bedrooms = set()
    building_types = set()
    locations_value = parse_estate_value(listing.get("LocationAdvantages"))
    locations = {}
    if isinstance(price_details, dict):
        for label, details in price_details.items():
            bedrooms.update(int(value) for value in re.findall(r"\d+", str(label)))
            if isinstance(details, dict) and details.get("building_type"):
                building_types.add(str(details["building_type"]).strip().lower())
    if isinstance(locations_value, dict):
        for place, distance in locations_value.items():
            place_key = re.sub(r"\W+", " ", str(place).lower()).strip()
            proximity = estate_proximity(distance)
            if place_key and proximity:
                locations[place_key] = proximity

    areas = []
    prices = []
    if isinstance(price_details, dict):
        for details in price_details.values():
            if not isinstance(details, dict):
                continue
            area_values = re.findall(r"\d[\d,]*(?:\.\d+)?", str(details.get("area", "")))
            areas.extend(float(value.replace(",", "")) for value in area_values)
            price_text = str(details.get("price-range", ""))
            price_values = re.findall(r"([\d,]+(?:\.\d+)?)\s*(Cr|Crore|L|Lakhs?)?", price_text, re.I)
            default_unit = next((unit for _, unit in reversed(price_values) if unit), "Cr")
            for amount, unit in price_values:
                price = float(amount.replace(",", ""))
                if (unit or default_unit).lower().startswith("l"):
                    price /= 100
                prices.append(price)

    facilities_value = parse_estate_value(listing.get("TopFacilities"))
    facilities = {
        re.sub(r"\s+", " ", str(value).strip().lower())
        for value in facilities_value
    } if isinstance(facilities_value, list) else set()
    return {
        "bedrooms": bedrooms,
        "building_types": building_types,
        "locations": locations,
        "area": float(np.median(areas)) if areas else None,
        "price": float(np.median(prices)) if prices else None,
        "facilities": facilities,
    }


def estate_set_similarity(first, second):
    union = first | second
    return len(first & second) / len(union) if union else None


def estate_value_similarity(first, second, ratio_limit):
    if first is None or second is None or first <= 0 or second <= 0:
        return None
    return max(0, 1 - abs(np.log(first / second)) / np.log(ratio_limit))


def estate_location_similarity(first, second):
    first_locations, second_locations = first["locations"], second["locations"]
    union = set(first_locations) | set(second_locations)
    if not union:
        return None
    shared = set(first_locations) & set(second_locations)
    if not shared:
        return 0.0
    proximity_scores = []
    for place in shared:
        first_value, second_value = first_locations[place], second_locations[place]
        if first_value[0] == second_value[0] and first_value[0] != "text":
            value_similarity = max(0, 1 - abs(first_value[1] - second_value[1]) / max(first_value[1], second_value[1], 1))
        else:
            value_similarity = float(first_value == second_value)
        proximity_scores.append(value_similarity)
    overlap = len(shared) / len(union)
    return .7 * overlap + .3 * float(np.mean(proximity_scores))


def estate_similarity(first, second, bases):
    components = []
    if "LocationAdvantages" in bases:
        location_score = estate_location_similarity(first, second)
        if location_score is not None:
            components.append(location_score)
    if "PriceDetails" in bases:
        price_scores = [
            estate_set_similarity(first["bedrooms"], second["bedrooms"]),
            estate_set_similarity(first["building_types"], second["building_types"]),
            estate_value_similarity(first["price"], second["price"], 5),
            estate_value_similarity(first["area"], second["area"], 4),
        ]
        available_scores = [score for score in price_scores if score is not None]
        if available_scores:
            components.append(float(np.mean(available_scores)))
    if "TopFacilities" in bases:
        facility_score = estate_set_similarity(first["facilities"], second["facilities"])
        if facility_score is not None:
            components.append(facility_score)
    if not components:
        return 0.0
    return 100 * float(np.mean(components))


def recommend_estate_properties(property_name, bases, count=5):
    name_column = estate_data.columns[0]
    selected_rows = estate_data[estate_data[name_column].astype(str) == str(property_name)]
    if selected_rows.empty:
        return estate_data.iloc[0:0].assign(similarity=pd.Series(dtype=float))

    selected_index = selected_rows.index[0]
    selected_profile = estate_profile(estate_data.loc[selected_index])
    scored = []
    for index, listing in estate_data.iterrows():
        if index == selected_index:
            continue
        scored.append((estate_similarity(selected_profile, estate_profile(listing), bases), index))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    recommendations = estate_data.loc[[index for _, index in scored[:count]]].copy()
    recommendations["similarity"] = [score for score, _ in scored[:count]]
    return recommendations


with st.sidebar:
    st.markdown('<div class="side-brand"><span class="brand-mark"><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#180b08" stroke-width="2"><path d="M3 21h18M5 21V9l7-5 7 5v12M9 21v-6h6v6M8 10h.01M12 10h.01M16 10h.01" stroke-linecap="round" stroke-linejoin="round"/></svg></span>EstateLens</div><div class="side-sub">GURGAON PROPERTY INTELLIGENCE</div>', unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Overview", "Price Predictor", "Recommendations", "Analytical Lab", "Regression Analysis", "Methodology"], label_visibility="visible")
    st.markdown("---")
    st.caption("Model status")
    st.success("● Online · leakage-safe")
    st.caption(f"{len(data):,} verified listings")

if page == "Overview":
    st.markdown('<div class="section-kicker">Command center</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-card"><h2 style="margin:0">Welcome to EstateLens</h2><p style="margin:.35rem 0 0;color:#527078">Explore Gurgaon property values with a clean, explainable regression model.</p></div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Listings analysed", f"{len(data):,}")
    b.metric("Median property", f"₹{data.price.median():.2f} Cr")
    c.metric("Model CV R²", f"{metrics['cv_r2_mean']:.3f}")
    d.metric("Holdout MAE", f"₹{metrics['mae_crore']:.2f} Cr")
    left, right = st.columns([1.25, .75])
    with left:
        st.plotly_chart(px.histogram(data, x="price", nbins=45, title="How property prices are distributed", color_discrete_sequence=["#f59e0b"], template="plotly_dark"), use_container_width=True)
    with right:
        sector_prices = data.groupby("sector", as_index=False)["price"].median().sort_values("price", ascending=False).head(10)
        st.plotly_chart(px.bar(sector_prices, x="price", y="sector", orientation="h", title="Premium sectors", color="price", color_continuous_scale="YlOrRd", template="plotly_dark"), use_container_width=True)
    st.info("Use Price Predictor from the sidebar to estimate a property's market value.")

elif page == "Price Predictor":
    st.markdown('<div class="section-kicker">Smart valuation</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-card"><h2 style="margin:0">Find your property value</h2><p style="margin:.35rem 0 0;color:#527078">Enter the property profile and get an instant estimate in crore INR.</p></div>', unsafe_allow_html=True)
    with st.form("valuation_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            property_type = st.selectbox("Property type", options("property_type", ["flat", "house"]))
            sector = st.selectbox("Sector / locality", options("sector", ["sector 57"]))
            area = st.number_input("Area (sq ft)", min_value=100.0, max_value=100000.0, value=1200.0, step=25.0)
            built_up = st.number_input("Built-up area (sq ft)", min_value=0.0, value=0.0, step=25.0)
        with c2:
            bedrooms = st.slider("Bedrooms", 1, 10, 3)
            bathrooms = st.slider("Bathrooms", 1, 10, 2)
            balcony = st.slider("Balconies", 0, 5, 2)
            floor = st.number_input("Floor number", min_value=0.0, max_value=60.0, value=3.0, step=1.0)
        with c3:
            age = st.selectbox("Property age", options("agePossession", ["Undefined"]))
            furnishing = st.selectbox("Furnishing", [0, 1, 2], format_func=lambda x: {0:"Unfurnished",1:"Semi-furnished",2:"Furnished"}[x])
            carpet = st.number_input("Carpet area (sq ft)", min_value=0.0, value=0.0, step=25.0)
            predict = st.form_submit_button("Estimate market value", use_container_width=True, type="primary")
    if predict:
        row = {
            "property_type": property_type, "sector": sector,
            "area": area, "bedRoom": bedrooms, "bathroom": bathrooms,
            "balcony": balcony, "floorNum": floor,
            "agePossession": age, "Pooja Room": 0, "Study Room": 0,
            "Store Room": 0, "Servant Room": 0, "Others": 0,
            "furnishstatus": furnishing,
            "super_built_up_area": np.nan, "built_up_area": built_up or np.nan,
            "carpet_area": carpet or np.nan,
        }
        prediction = float(np.clip(np.expm1(model.predict(make_features(form_frame(row)))[0]), 0, None))
        error_margin = float(metrics.get("mae_crore", 0.5))
        lower = max(0.0, prediction - error_margin)
        upper = prediction + error_margin
        st.success(f"Estimated range: ₹{lower:.2f} – ₹{upper:.2f} crore")
        st.metric("Best estimate", f"₹{prediction:.2f} crore", f"± ₹{error_margin:.2f} crore model MAE")
        st.caption("Use this as an analytical estimate, not a certified valuation. Locality and listing quality can move the actual price materially.")
        explanation = shap_explanation(row)
        if not explanation.empty:
            st.markdown("### Why this estimate?")
            st.caption("Positive bars push the estimate higher; negative bars push it lower. SHAP values are shown on the model's log-price scale.")
            fig = px.bar(explanation.sort_values("impact"), x="impact", y="feature", orientation="h",
                         color="impact", color_continuous_scale="RdBu", template="plotly_dark",
                         title="Top contributors to this prediction")
            fig.update_layout(height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

elif page == "Recommendations":
    st.markdown('<div class="section-kicker">Smart discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-card"><h2 style="margin:0">Find similar properties</h2><p style="margin:.35rem 0 0;color:#aaa7c3">Choose a listed property to see five similar options.</p></div>', unsafe_allow_html=True)
    name_column = estate_data.columns[0]
    property_names = estate_data[name_column].dropna().astype(str).tolist()
    selected_name = st.selectbox("Property name", property_names)
    selected_listing = estate_data.loc[estate_data[name_column].astype(str) == selected_name].iloc[0]
    st.caption(str(selected_listing.get("PropertySubName", "")))
    selected_link = str(selected_listing.get("Link", ""))
    if selected_link.startswith("http"):
        st.link_button("Open selected property", selected_link)

    recommendation_bases = st.multiselect(
        "Compare properties using",
        ["LocationAdvantages", "PriceDetails", "TopFacilities"],
        default=["LocationAdvantages", "PriceDetails", "TopFacilities"],
        format_func={
            "LocationAdvantages": "Location advantages",
            "PriceDetails": "Price details",
            "TopFacilities": "Top facilities",
        }.get,
    )
    st.markdown("### Top 5 similar properties")
    if not recommendation_bases:
        st.info("Choose at least one similarity basis.")
    else:
        recommendations = recommend_estate_properties(selected_name, recommendation_bases)
    if recommendation_bases and recommendations.empty:
        st.info("No similar properties were found for this listing.")
    elif recommendation_bases:
        cards = st.columns(3)
        for card_index, (listing_index, listing) in enumerate(recommendations.iterrows()):
            with cards[card_index % len(cards)]:
                listing_name = str(listing.iloc[0])
                listing_link = str(listing.get("Link", ""))
                if listing_link.startswith("http"):
                    st.link_button(listing_name, listing_link, use_container_width=True)
                else:
                    st.markdown(f"**{listing_name}**")
                st.caption(str(listing.get("PropertySubName", "")))
                st.progress(float(listing.similarity) / 100, text=f"{listing.similarity:.0f}% similar")

elif page == "Analytical Lab":
    st.markdown('<div class="section-kicker">Market analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-card"><h2 style="margin:0">Gurgaon analytical lab</h2><p style="margin:.35rem 0 0;color:#aaa7c3">Explore sector geography, amenities, demand mix and price relationships.</p></div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1, 1.3, 1])
    with f1:
        selected_type = st.selectbox("Property type", ["All types"] + sorted(data.property_type.dropna().unique().tolist()))
    with f2:
        selected_sector = st.selectbox("Sector", ["All sectors"] + sorted(data.sector.unique().tolist()))
    with f3:
        selected_bhk = st.multiselect("BHK", sorted(data.bedRoom.dropna().astype(int).unique().tolist()), default=[])
    filtered = data.copy()
    if selected_type != "All types":
        filtered = filtered[filtered.property_type == selected_type]
    if selected_sector != "All sectors":
        filtered = filtered[filtered.sector == selected_sector]
    if selected_bhk:
        filtered = filtered[filtered.bedRoom.isin(selected_bhk)]

    st.caption(f"Showing {len(filtered):,} matching listings")
    if filtered.empty:
        st.warning("No listings match these filters. Try a broader sector or society selection.")
        st.stop()

    # Filter-aware observations make the charts actionable instead of purely
    # descriptive.
    top_type = filtered.property_type.mode().iat[0]
    top_bhk = int(filtered.bedRoom.mode().iat[0])
    top_sector = filtered.groupby("sector").price.median().idxmax()
    st.markdown("#### Key observations")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Median price", f"₹{filtered.price.median():.2f} Cr")
    o2.metric("Typical home", f"{top_bhk} BHK")
    o3.metric("Most common type", top_type.title())
    o4.metric("Highest median sector", top_sector.title())
    st.info(
        f"Among the selected listings, {top_type.title()} properties and {top_bhk} BHK homes are most common. "
        f"Prices range from ₹{filtered.price.min():.2f} Cr to ₹{filtered.price.max():.2f} Cr; "
        f"{top_sector.title()} has the highest median price within this filtered view."
    )

    st.markdown("#### Sector geomap")
    map_metrics = filtered.groupby("sector", as_index=False).agg(
        built_up_area=("built_up_area", "median"),
        price_per_sqft=("price_per_sqft", "median"),
        median_price=("price", "median"),
        listings=("sector", "size"),
    )
    map_metrics["sector_key"] = map_metrics.sector.map(sector_map_key)
    sector_geojson = sector_boundary_geojson(ROOT / "data" / "Cleaned" / "gurgaon_sector_boundaries.geojson")
    boundary_keys = {feature["properties"]["sector_key"] for feature in sector_geojson["features"]}
    map_data = map_metrics[map_metrics.sector_key.isin(boundary_keys)]
    missing_boundaries = sorted(map_metrics.loc[~map_metrics.sector_key.isin(boundary_keys), "sector"].tolist())
    if map_data.empty:
        st.warning("No matching boundary is available from GMDA/DTCP or OSM for these filters.")
    else:
        sector_map = px.choropleth_map(
            map_data,
            geojson=sector_geojson,
            locations="sector_key",
            featureidkey="properties.sector_key",
            color="price_per_sqft",
            hover_name="sector",
            hover_data={
                "built_up_area": ":,.0f",
                "price_per_sqft": ":,.0f",
                "median_price": ":.2f",
                "listings": True,
            },
            zoom=10,
            center={"lat": 28.4595, "lon": 77.0266},
            map_style="open-street-map",
            color_continuous_scale="YlOrRd",
            opacity=0.78,
            labels={
                "built_up_area": "Median built-up area (sq ft)",
                "price_per_sqft": "Median price/sq ft",
                "median_price": "Median price (Cr)",
                "listings": "Listings",
            },
            title="Price per sq ft by sector",
        )
        sector_map.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=45, b=0),
            paper_bgcolor="#111426",
            font_color="#f8fafc",
        )
        st.caption("Boundaries: [GMDA OneMap / DTCP GIS](https://expressway-wheat.vercel.app/) and © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). Hover for built-up area, median price, and listings.")
        st.plotly_chart(sector_map, use_container_width=True)
    if missing_boundaries and not map_data.empty:
        st.warning(f"No GMDA/DTCP or OSM boundary found for {len(missing_boundaries)} of {len(map_metrics)} sectors: {', '.join(missing_boundaries)}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Amenities word cloud")
        amenity_frequencies = amenity_text(selected_type, selected_sector, selected_bhk)
        if amenity_frequencies and WordCloud is not None:
            wc = WordCloud(width=900, height=800, background_color="#111426", colormap="YlOrRd", prefer_horizontal=.8).generate_from_frequencies(amenity_frequencies)
            fig, ax = plt.subplots(figsize=(10, 8), facecolor="#111426")
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); fig.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        elif amenity_frequencies:
            st.error("Word cloud unavailable. Install the declared `wordcloud` package.")
        else:
            st.info("No amenity data is available for these filters.")
    with c2:
        st.markdown("#### Area vs price")
        fig = px.scatter(filtered, x="area", y="price", color="property_type", size="bedRoom",
                         hover_data=["sector", "bedRoom"], template="plotly_dark", height=430,
                         color_discrete_sequence=["#ef4444", "#fbbf24"], title="Larger homes vs asking price")
        fig.update_layout(paper_bgcolor="#111426", plot_bgcolor="#111426")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 3D property landscape")
    st.caption("Rotate, zoom and use the BHK slider to explore area, bedrooms and price together.")
    plot3d = filtered.copy()
    plot3d["bedroom_frame"] = plot3d["bedRoom"].fillna(0).astype(int).astype(str)
    plot3d["listing_id"] = np.arange(len(plot3d))
    fig3d = px.scatter_3d(
        plot3d, x="area", y="bathroom", z="price", color="property_type",
        size="bedRoom", hover_name="sector", hover_data=["bedRoom", "price"],
        animation_frame="bedroom_frame", animation_group="listing_id",
        color_discrete_sequence=["#ef4444", "#fbbf24"],
        template="plotly_dark", title="Interactive 3D price topology",
    )
    fig3d.update_layout(height=650, paper_bgcolor="#111426", scene_bgcolor="#0b1020",
                        margin=dict(l=0, r=0, t=55, b=0), scene=dict(
                            xaxis_title="Area (sq ft)", yaxis_title="Bathrooms", zaxis_title="Price (Cr)"))
    st.plotly_chart(fig3d, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### BHK mix by sector")
        bhk = filtered.bedRoom.value_counts().sort_index().reset_index()
        bhk.columns = ["BHK", "listings"]
        st.plotly_chart(px.pie(bhk, names="BHK", values="listings", hole=.52, template="plotly_dark",
                               color_discrete_sequence=["#ef4444", "#f97316", "#fbbf24", "#fde68a"], title="Bedroom mix"), use_container_width=True)
    with c2:
        st.markdown("#### Bedrooms vs price")
        fig = px.box(filtered, x="bedRoom", y="price", color="property_type", points=False,
                     template="plotly_dark", color_discrete_sequence=["#ef4444", "#fbbf24"],
                     title="Price spread by bedroom count")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Flat vs house price distribution")
    fig = px.histogram(filtered, x="price", color="property_type", nbins=35, barmode="overlay",
                       opacity=.72, template="plotly_dark", color_discrete_sequence=["#ef4444", "#fbbf24"],
                       title="Price distribution by property type")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Regression Analysis":
    st.markdown('<div class="section-kicker">Model observatory</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-card"><h2 style="margin:0">Regression analysis</h2><p style="margin:.35rem 0 0;color:#527078">Performance, distribution and locality-level market signals.</p></div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Holdout MAE", f"₹{metrics['mae_crore']:.2f} Cr")
    b.metric("Holdout R²", f"{metrics['r2']:.3f}")
    c.metric("CV R²", f"{metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
    d.metric("Training rows", f"{len(data):,}")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(data, x="price", nbins=45, title="Price distribution (crore INR)", color_discrete_sequence=["#f59e0b"], template="plotly_dark"), use_container_width=True)
    with right:
        sector_prices = data.groupby("sector", as_index=False)["price"].median().sort_values("price", ascending=False).head(15)
        st.plotly_chart(px.bar(sector_prices, x="price", y="sector", orientation="h", title="Top localities by median price", color="price", color_continuous_scale="YlOrRd", template="plotly_dark"), use_container_width=True)
    st.info("The displayed score is from a fixed, unseen 20% holdout. `price_per_sqft` is excluded because it is derived from the target and would create leakage.")

elif page == "Methodology":
    st.markdown('<div class="section-kicker">Inside the model</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-card"><h2 style="margin:0">Why you can trust the estimate</h2><p style="margin:.35rem 0 0;color:#527078">A transparent workflow built to avoid target leakage and overfitting.</p></div>', unsafe_allow_html=True)
    st.markdown("""
    - **Leakage prevention:** removed `price_per_sqft`, which contains the answer.
    - **Better signal:** retained society and sector locality information instead of ordinal-encoding them as arbitrary numbers.
    - **Feature engineering:** log area, area per bedroom, bathrooms per bedroom, and missing-area indicators.
    - **Stable target:** trained on `log1p(price)` so expensive listings do not dominate the fit.
    - **Regularisation:** shallow-ish boosted trees, row/column subsampling, minimum child weight and L1/L2 penalties reduce overfitting.
    - **Validation:** fixed holdout plus shuffled 5-fold cross-validation.
    """)
    st.code("streamlit run app.py", language="bash")
