import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import numpy as np

# Load your trained models
with open("catboost_model.pkl", "rb") as f:
    catboost_model = pickle.load(f)

with open("lgb_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

with open("meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

# Load your dataset for reference
data_options = pd.read_csv("concert_data_filtered.csv")

# Define categorical columns used during training
categorical_cols = ['Headliner', 'Support', 'Venue', 'City', 'State', 'Country', 
                    'Market', 'Company Type', 'Promoter', 'Genre', 'month', 'weekday']

# Prepare dropdown options
top_headliners = data_options['Headliner'].value_counts().head(50).index.tolist()
top_support = data_options['Support'].value_counts().head(20).index.tolist()
top_venue = data_options['Venue'].value_counts().head(20).index.tolist()
top_company_types = data_options['Company Type'].dropna().value_counts().head(20).index.tolist()
top_promoters = data_options['Promoter'].dropna().value_counts().head(20).index.tolist()
top_genres = data_options['Genre'].dropna().value_counts().head(20).index.tolist()

# Pre-filter venue-related info
venue_filter = {
    venue: data_options[data_options['Venue'] == venue]
    for venue in top_venue
}

# --- USER INPUT FORM ---
st.title("🎤 Concert Gross Revenue Prediction")

with st.form("input_form"):
    # Venue selection
    venue_options = top_venue + ["Other (type manually)"]
    venue_selected = st.selectbox("Venue", venue_options)
    if venue_selected == "Other (type manually)":
        venue = st.text_input("Enter custom venue")
        venue_data = data_options  # fallback
    else:
        venue = venue_selected
        venue_data = venue_filter[venue]

    # Venue-based suggestions
    cities_for_venue = venue_data['City'].dropna().unique().tolist()
    states_for_venue = venue_data['State'].dropna().unique().tolist()
    markets_for_venue = venue_data['Market'].dropna().unique().tolist()
    avg_capacity_for_venue = venue_data['Avg. Event Capacity'].mean()

    city = st.selectbox("City", cities_for_venue + ['Other (type manually)'])
    if city == "Other (type manually)":
        city = st.text_input("Enter custom city")

    state = st.selectbox("State", states_for_venue + ['Other (type manually)'])
    if state == "Other (type manually)":
        state = st.text_input("Enter custom state")

    market = st.selectbox("Market", markets_for_venue + ['Other (type manually)'])
    if market == "Other (type manually)":
        market = st.text_input("Enter custom market")

    avg_event_capacity = st.number_input("Avg. Event Capacity", min_value=0.0, step=1.0, value=avg_capacity_for_venue or 0.0)

    # Headliner / Support
    headliner = st.selectbox("Headliner", top_headliners + ['Other (type manually)'])
    if headliner == "Other (type manually)":
        headliner = st.text_input("Enter custom headliner")

    support = st.selectbox("Support Act(s)", top_support + ['Other (type manually)'])
    if support == "Other (type manually)":
        support = st.text_input("Enter custom support act")

    # Other fields
    company_type = st.selectbox("Company Type", top_company_types + ['Other (type manually)'])
    if company_type == "Other (type manually)":
        company_type = st.text_input("Enter custom company type")

    promoter = st.selectbox("Promoter", top_promoters + ['Other (type manually)'])
    if promoter == "Other (type manually)":
        promoter = st.text_input("Enter custom promoter")

    genre = st.selectbox("Genre", top_genres + ['Other (type manually)'])
    if genre == "Other (type manually)":
        genre = st.text_input("Enter custom genre")

    number_of_shows = st.number_input("Number of Shows", min_value=1, step=1)
    avg_capacity_sold = st.number_input("Avg. Capacity Sold (0-1)", min_value=0.0, max_value=1.0, step=0.01)
    ticket_price_min = st.number_input("Ticket Price Min (USD)", min_value=0.0, step=1.0)
    ticket_price_max = st.number_input("Ticket Price Max (USD)", min_value=0.0, step=1.0)

    date = st.date_input("Event Date", value=datetime.today())
    hour = st.slider("Hour of Event (0-23)", min_value=0, max_value=23, value=20)

    submit = st.form_submit_button("🎯 Predict Gross Revenue")

# --- PREDICTION ---
if submit:
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()
    country = "USA"

    # Prepare input data
    input_data = pd.DataFrame([{
        'Number of Shows': number_of_shows,
        'Headliner': headliner,
        'Support': support,
        'Venue': venue,
        'City': city,
        'State': state,
        'Country': country,
        'Market': market,
        'Company Type': company_type,
        'Promoter': promoter,
        'Genre': genre,
        'Avg. Event Capacity': avg_event_capacity,
        'Avg. Capacity Sold': avg_capacity_sold,
        'Ticket Price Min USD': ticket_price_min,
        'Ticket Price Max USD': ticket_price_max,
        'year': year,
        'month': month,
        'day': day,
        'weekday': weekday,
        'hour': hour
    }])

    # Ensure consistent column order
    expected_cols = catboost_model.feature_names_
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = pd.NA

    input_data = input_data[expected_cols]

    # Cast categorical columns to category
    for col in categorical_cols:
        input_data[col] = input_data[col].astype('category')

    # --- Model Predictions ---
    cat_pool = Pool(data=input_data, cat_features=categorical_cols)
    catboost_pred = catboost_model.predict(cat_pool)
    lgb_pred = lgb_model.predict(input_data)

    combined_preds = pd.DataFrame({
        'catboost_pred': catboost_pred,
        'lgb_pred': lgb_pred
    })

    final_prediction = meta_model.predict(combined_preds)[0]

    # Reverse log transformation to original scale
    final_prediction_original = np.expm1(final_prediction)

    st.success(f"💰 Predicted Avg. Gross Revenue: **${final_prediction_original:,.2f} USD**")

    # Optional debug info
    with st.expander("🔍 Show Debug Info"):
        st.write("Input Data")
        st.dataframe(input_data)
        st.write("CatBoost Prediction:", catboost_pred)
        st.write("LightGBM Prediction:", lgb_pred)
        st.write("Final Meta Model Prediction:", final_prediction)
        st.write("Final Prediction on Original Scale:", final_prediction_original)
