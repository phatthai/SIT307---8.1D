# SIT307/SIT720 - Task 8.1D
## Sydney Housing Price Prediction and Decision Support System

### Contents

| File | What it is |
|---|---|
| `SIT307_8_1D_parts2to4.ipynb` | Executed notebook covering Parts 2, 3 and 4 |
| `app.py` | Streamlit web application (Part 5) |
| `sydney_housing_data_clean.xlsx` | The collected dataset (110 sales) |
| `housing_model.joblib` | Fitted Gradient Boosting pipeline |
| `model_metadata.json` | Feature layout and CV scores used by the app |
| `model_comparison.csv` | Cross-validated results for all three models |
| `prediction_errors.csv` | Per-property predictions and errors (Part 4) |
| `requirements.txt` | Python dependencies |

### Dataset

110 sold listings collected manually from domain.com.au:
Blacktown 2148 (36), Parramatta 2150 (34), Chatswood 2067 (40).
Price range $320,000 to $6,200,000.

The workbook has four sheets:
- `sydney_housing_data` - the modelling dataset, with shortened agent descriptions
  and 15 binary `f_*` features extracted from the original text
- `Agent descriptions (raw)` - the full original listing text for every property
- `Excluded (price withheld)` - listings dropped because no price was disclosed
- `Parramatta houses 2025` - Parramatta house sales outside the 12-month window

### Reproducing the results

```bash
pip install -r requirements.txt
jupyter notebook SIT307_8_1D_parts2to4.ipynb   # Run All
```

The notebook reads `sydney_housing_data_clean.xlsx` from the same folder and writes
`housing_model.joblib`, `model_metadata.json`, `model_comparison.csv` and
`prediction_errors.csv`. Runtime is under two minutes on a laptop.

### Running the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 and has three tabs:

1. **Single property** - enter suburb, type, bedrooms, bathrooms, car spaces, land size
   and sale method, then either paste the agent's description to fill the feature
   checkboxes automatically or tick them by hand. Returns a point estimate, a range
   based on the model's cross-validated error, context-specific reliability warnings,
   and the closest comparable sales from the dataset.
2. **Upload a file** - upload a CSV or Excel file to predict many properties at once.
   A blank template is downloadable from the tab.
3. **About this model** - how the model was built and where it should not be trusted.

`app.py` loads the pipeline saved by the notebook, so the deployed model is exactly
the one that was cross-validated. Run the notebook before starting the app.

### Results summary

| Model | CV MAPE | CV MAE | R2 (log) | Train MAPE | Gap |
|---|---|---|---|---|---|
| Gradient Boosting | 17.3% | $276,868 | 0.869 | 5.2% | 12.2 |
| Random Forest | 19.3% | $304,942 | 0.823 | 10.0% | 9.3 |
| Ridge regression | 19.8% | $316,419 | 0.839 | 14.6% | 5.2 |

5-fold cross-validation, models trained on log(price), metrics computed on
back-transformed dollars.
