# Reusable data cleaning

All cleaning is exposed through one function in [`data_cleaning.py`](data_cleaning.py):

```python
from data_cleaning import clean_data

# Dataset type is inferred from the columns when omitted.
cleaned = clean_data("data/raw/flats - flats.csv")
cleaned.to_csv("data/processed/flats_cleaned.csv", index=False)

# The same function can clean another file.
other = clean_data("data/raw/house.csv", dataset_type="houses")
other.to_csv("data/processed/house_cleaned.csv", index=False)
```

`clean_data` also accepts an already loaded pandas DataFrame and never modifies
that input in place. Use `dataset_type="flats"` or `dataset_type="houses"` when
automatic schema detection is not appropriate.
