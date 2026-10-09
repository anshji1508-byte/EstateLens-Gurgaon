# Gurgaon Price Intelligence

## Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The model uses scikit-learn's built-in `TargetEncoder`; no separate `category_encoders` installation is required.

The trained bundle is written to `Pkles/property_price_model.pkl`.

## Sector map boundaries

`data/Cleaned/gurgaon_sector_boundaries.geojson` combines polygons from the GMDA OneMap / Haryana DTCP sector layer (republished at [Gurugram Sector Map](https://expressway-wheat.vercel.app/)) with supplemental OpenStreetMap polygons. OpenStreetMap data is © its contributors and licensed under ODbL 1.0. The app reports sectors that have listings but no matching polygon in these sources.

## Validation note

Prices are in crore INR. The model is evaluated on an unseen 20% holdout and 5-fold cross-validation. `price_per_sqft` is intentionally excluded because it is calculated from `price`; using it would make the score look better while making deployment invalid. The app displays the measured MAE and R² rather than hiding uncertainty.
