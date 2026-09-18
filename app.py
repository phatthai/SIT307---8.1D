"""
SIT307: Task 8.1D - Part 5
Sydney Housing Price Prediction and Decision Support System

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

The app loads the Gradient Boosting pipeline fitted in
SIT307_8_1D_parts2to4.ipynb (housing_model.joblib) together with the
metadata written alongside it, so the deployed model is exactly the one
that was cross-validated in the notebook.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
MODEL_PATH = HERE / "housing_model.joblib"
META_PATH = HERE / "model_metadata.json"
DATA_PATH = HERE / "sydney_housing_data_clean.xlsx"

st.set_page_config(page_title="Sydney Housing Price Predictor",
                   page_icon="🏠", layout="wide")


# Loading
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. "
                 "Run the notebook first to generate housing_model.joblib.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


@st.cache_data
def load_reference_data():
    try:
        df = pd.read_excel(DATA_PATH, sheet_name="sydney_housing_data")
        type_map = {"House": "House", "Apartment / Unit / Flat": "Unit",
                    "Townhouse": "Attached", "Semi-detached": "Attached",
                    "Villa": "Attached", "Duplex": "Attached"}
        df["type_group"] = df["Property type"].map(type_map)
        return df
    except Exception:
        return None


model, meta = load_model()
ref = load_reference_data()

FLAGS = [
    ("f_renovated", "Renovated or freshly updated"),
    ("f_brand_new", "Brand new or near-new build"),
    ("f_needs_work", "Original condition / scope to improve"),
    ("f_dev_potential", "Development potential (STCA, R4/LMR zoning)"),
    ("f_granny_flat", "Has a granny flat or second dwelling"),
    ("f_view", "Has a view or notable outlook"),
    ("f_private_pool", "Private swimming pool"),
    ("f_complex_facilities", "Building facilities (pool, gym, concierge)"),
    ("f_near_transport", "Walking distance to station or Metro"),
    ("f_premium_finish", "Premium finishes or appliances"),
    ("f_period_character", "Period or character features"),
    ("f_ensuite", "Ensuite bathroom"),
    ("f_ducted_ac", "Ducted air conditioning"),
    ("f_study_extra_living", "Study, rumpus or extra living area"),
    ("f_alfresco", "Alfresco or outdoor entertaining area"),
]

# Keyword rules used in the notebook, reused here so that pasting a listing
# description populates the same flags the model was trained on.
KEYWORDS = {
    "f_renovated": r"renovated|refurbished|fully updated|beautifully updated|freshly painted|"
                   r"brand.?new (?:kitchen|bathroom|carpet|flooring)|new carpet|modernised|move.?in.?ready",
    "f_brand_new": r"brand.?new (?:residence|build|home|townhouse)|newly built|near.?new|custom built",
    "f_needs_work": r"original (?:condition|yet|home)|scope to (?:renovate|enhance|update|improve)|"
                    r"blank canvas|renovator|requires repairs|add your own",
    "f_dev_potential": r"\bstca\b|subject to council approval|redevelop|development potential|"
                       r"duplex|knockdown|zoned r4|\blmr\b|low.?medium density|development site",
    "f_granny_flat": r"granny flat|dual income|second dwelling|self.?contained",
    "f_view": r"\bviews?\b|outlook|vista|glimpses|skyline",
    "f_private_pool": r"in.?ground pool|swimming pool|sparkling pool|poolside",
    "f_complex_facilities": r"gymnasium|\bgym\b|sauna|concierge|resort.?style|lap pool|heated pool",
    "f_near_transport": r"walk to|walking distance|footsteps|stroll to|minutes.{0,10}walk",
    "f_premium_finish": r"miele|gaggenau|smeg|bosch|ilve|marble|granite|stone bench|caesarstone|"
                        r"luxur|gourmet|designer|premium finishes",
    "f_period_character": r"art deco|federation|c19\d\d|period (?:features|detail)|ornate ceilings|"
                          r"character|leadlight|bungalow|weatherboard",
    "f_ensuite": r"ensuite|en.?suite",
    "f_ducted_ac": r"ducted (?:air|reverse)",
    "f_study_extra_living": r"\bstudy\b|home office|study nook|rumpus|theatre room",
    "f_alfresco": r"alfresco|entertaining area|covered (?:deck|patio|terrace)|\bbbq\b|pergola|cabana",
}


def flags_from_text(text: str) -> dict:
    """Apply the notebook's keyword rules to a pasted description."""
    import re
    t = re.sub(r"\s+", " ", text).lower()
    return {k: int(bool(re.search(p, t))) for k, p in KEYWORDS.items()}


def build_row(suburb, type_group, beds, baths, cars, land, months,
              sale_method, flag_values):
    """Assemble a single-row frame matching the training feature layout."""
    has_land = 0 if type_group == "Unit" else 1
    land_size = 0.0 if type_group == "Unit" else float(land)
    row = {
        "Bedrooms": float(beds),
        "Bathrooms": float(baths),
        "car_spaces": float(cars),
        "land_size": land_size,
        "months_before_latest": float(months),
        "has_land": has_land,
        "land_imputed": 0,
        "Suburb": suburb,
        "type_group": type_group,
        "Sale method": sale_method,
    }
    row.update(flag_values)
    order = (meta["numeric_features"] + meta["binary_features"]
             + meta["categorical_features"])
    return pd.DataFrame([row])[order]



# Layout
st.title("🏠 Sydney Housing Price Predictor")
st.caption(
    f"{meta['model_name']} trained on {meta['training_rows']} manually collected sales "
    f"from Blacktown, Parramatta and Chatswood. "
    f"Cross-validated MAPE {meta['cv_mape']:.1f}%."
)

tab_single, tab_batch, tab_about = st.tabs(
    ["Single property", "Upload a file", "About this model"])

# ------------------------------- Single ----------------------------------
with tab_single:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Property details")
        c1, c2 = st.columns(2)
        suburb = c1.selectbox("Suburb", meta["suburbs"])
        type_group = c2.selectbox("Property type", meta["type_groups"], index=1)

        c1, c2, c3 = st.columns(3)
        beds = c1.number_input("Bedrooms", 0, 10, 3, 1)
        baths = c2.number_input("Bathrooms", 1, 8, 2, 1)
        cars = c3.number_input("Car spaces", 0, 8, 1, 1)

        if type_group == "Unit":
            st.caption("Land size does not apply to units and is set to zero.")
            land = 0
        else:
            land = st.number_input("Land size (sqm)", 0, 3000, 550, 10)

        sale_method = st.selectbox("Sale method", meta["sale_methods"])
        months = st.slider("Months before the latest sale in the dataset", 0, 24, 0)

        st.subheader("Property features")
        st.caption("Paste the agent's description to fill these automatically, "
                   "or tick them by hand.")
        desc = st.text_area("Agent description (optional)", height=110,
                            placeholder="Paste the listing description here...")

        auto = flags_from_text(desc) if desc.strip() else {}
        flag_values = {}
        fc1, fc2 = st.columns(2)
        for i, (key, label) in enumerate(FLAGS):
            col = fc1 if i % 2 == 0 else fc2
            flag_values[key] = int(col.checkbox(
                label, value=bool(auto.get(key, 0)), key=f"flag_{key}"))

    with right:
        st.subheader("Estimated sale price")
        row = build_row(suburb, type_group, beds, baths, cars, land,
                        months, sale_method, flag_values)
        pred = float(np.exp(model.predict(row)[0]))
        mape = meta["cv_mape"] / 100
        lo, hi = pred * (1 - mape), pred * (1 + mape)

        st.metric("Point estimate", f"${pred:,.0f}")
        st.info(
            f"**Likely range: \\${lo:,.0f} to \\${hi:,.0f}**\n\n"
            f"The range reflects the model's cross-validated typical error of "
            f"{meta['cv_mape']:.1f}%. Treat the point estimate as the midpoint "
            f"of a range, not as a valuation."
        )

        warnings = []
        if pred > 3_000_000:
            warnings.append(
                "Above roughly \\$3m the model is least reliable. Every one of "
                "the five largest errors in testing was a high-end Chatswood property.")
        if type_group == "Attached":
            warnings.append(
                "Only 11 attached dwellings were in the training data, spanning "
                "\\$700k to \\$4.22m. Predictions for this type are the least reliable.")
        if flag_values.get("f_dev_potential"):
            warnings.append(
                "Development sites are priced on land area and zoning. Without the "
                "actual zoning code and floor space ratio, this estimate may be well off.")
        if pred < meta["price_range"][0] or pred > meta["price_range"][1]:
            warnings.append(
                f"The estimate falls outside the training range of "
                f"\\${meta['price_range'][0]:,} to \\${meta['price_range'][1]:,}. "
                "A tree ensemble cannot extrapolate beyond the prices it has seen.")
        for w in warnings:
            st.warning(w)

        if ref is not None:
            comp = ref[ref["Suburb"].eq(suburb) & ref["type_group"].eq(type_group)]
            if len(comp):
                st.subheader("Comparable sales in the dataset")
                st.caption(f"{len(comp)} {type_group.lower()} sales in {suburb}: "
                           f"median ${comp['Sold price'].median():,.0f}")
                comp = comp.assign(
                    _distance=(comp["Sold price"] - pred).abs())
                show = (comp.nsmallest(6, "_distance")
                        [["Address", "Bedrooms", "Bathrooms", "Sold price", "Sold date"]]
                        .sort_values("Sold price"))
                show["Sold price"] = show["Sold price"].map("${:,.0f}".format)
                show["Sold date"] = pd.to_datetime(show["Sold date"]).dt.date
                st.dataframe(show, hide_index=True, use_container_width=True)

#  Batch 
with tab_batch:
    st.subheader("Predict for many properties at once")
    st.markdown(
        "Upload a CSV or Excel file with the columns below. Any `f_*` column that "
        "is absent is treated as 0."
    )
    required = ["Suburb", "type_group", "Bedrooms", "Bathrooms",
                "car_spaces", "land_size", "months_before_latest", "Sale method"]
    st.code(", ".join(required + [f for f, _ in FLAGS]), language="text")

    template = pd.DataFrame([{
        "Suburb": "Parramatta", "type_group": "House", "Bedrooms": 3,
        "Bathrooms": 2, "car_spaces": 1, "land_size": 550,
        "months_before_latest": 0, "Sale method": "private treaty",
        **{f: 0 for f, _ in FLAGS},
    }])
    st.download_button("Download a blank template",
                       template.to_csv(index=False).encode(),
                       "template.csv", "text/csv")

    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if up is not None:
        try:
            batch = (pd.read_csv(up) if up.name.endswith(".csv")
                     else pd.read_excel(up))
            missing = [c for c in required if c not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                work = batch.copy()
                for f, _ in FLAGS:
                    if f not in work.columns:
                        work[f] = 0
                work["has_land"] = (~work["type_group"].eq("Unit")).astype(int)
                work["land_imputed"] = 0
                work.loc[work["type_group"].eq("Unit"), "land_size"] = 0.0
                order = (meta["numeric_features"] + meta["binary_features"]
                         + meta["categorical_features"])
                preds = np.exp(model.predict(work[order]))
                out = batch.copy()
                out["predicted_price"] = preds.round(0)
                out["range_low"] = (preds * (1 - meta["cv_mape"] / 100)).round(0)
                out["range_high"] = (preds * (1 + meta["cv_mape"] / 100)).round(0)
                st.success(f"Predicted {len(out)} properties.")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download predictions",
                                   out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")

#  About 
with tab_about:
    st.subheader("About this model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", meta["model_name"])
    c2.metric("Typical error (MAPE)", f"{meta['cv_mape']:.1f}%")
    c3.metric("Typical error (MAE)", f"${meta['cv_mae_dollars']:,.0f}")

    st.markdown(f"""
**How it was built.** {meta['training_rows']} sold listings were collected manually from
domain.com.au across Blacktown (2148), Parramatta (2150) and Chatswood (2067). The model
predicts `log(price)` rather than price, because raw prices in the dataset span
\\${meta['price_range'][0]:,} to \\${meta['price_range'][1]:,} and are strongly right-skewed.
Predictions are transformed back to dollars for display.

Three models were compared using 5-fold cross-validation: Ridge regression, Random Forest and
Gradient Boosting. Gradient Boosting performed best and is the model deployed here.

**Limitations you should know about.**

- The model only knows these three suburbs. It cannot price a property anywhere else.
- Roughly one sale in six will be off by more than {meta['cv_mape']:.0f}%.
- Accuracy is worst above \\$3m, where individual properties are effectively one of a kind.
- Internal floor area was unavailable for every listing and is not used.
- Land size does not apply to units and is set to zero for them.
- The dataset contains only disclosed prices. Sales with a withheld price were excluded,
  which biases the sample toward the lower end of each suburb's market.

**This tool is a decision aid, not a valuation.** It was built as a university assignment
and should not be used to make an actual purchase or sale decision.

*Model trained with scikit-learn {meta['sklearn_version']}. Latest sale in the training
data: {meta['latest_sold_date']}.*
""")
