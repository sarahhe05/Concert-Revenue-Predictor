import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import numpy as np

# Hide form border using custom CSS
st.markdown("""
    <style>
    div[data-testid="stForm"] {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained models
with open("catboost_model.pkl", "rb") as f:
    catboost_model = pickle.load(f)

with open("lgb_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

with open("meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

# Load reference dataset
data_options = pd.read_csv("concert_data_filtered.csv")

# Define categorical columns
categorical_cols = ['Headliner', 'Support', 'Venue', 'City', 'State', 'Country',
                    'Market', 'Company Type', 'Promoter', 'Genre', 'month', 'weekday']

# Clean Headliner names (strip whitespace and quotes)
headliner_counts = data_options['Headliner'].dropna().astype(str).apply(lambda x: x.strip().strip('"').strip("'"))
top_headliners = headliner_counts.value_counts().head(150).index.tolist()

# Move headliners starting with numbers to the end
alpha_headliners = [h for h in top_headliners if not h[:1].isdigit()]
numeric_headliners = [h for h in top_headliners if h[:1].isdigit()]
top_headliners = sorted(alpha_headliners) + sorted(numeric_headliners)

# Clean and sort other dropdowns (you can apply same cleaning if needed)
top_support = data_options['Support'].dropna().astype(str).apply(lambda x: x.strip().strip('"').strip("'")).value_counts().head(100).index.sort_values(ascending=True).tolist()
top_venue = data_options['Venue'].value_counts().head(100).index.sort_values(ascending=True).tolist()
top_company_types = data_options['Company Type'].dropna().value_counts().head(30).index.sort_values(ascending=True).tolist()
top_promoters = data_options['Promoter'].dropna().value_counts().head(30).index.sort_values(ascending=True).tolist()

# Title
st.title("🎤 Concert Gross Revenue Prediction")

# Venue selection (outside form so City/State updates immediately)
venue_selected = st.selectbox("Venue", top_venue)

# Auto-fill city, state, market, capacity
venue_data = data_options[data_options['Venue'] == venue_selected]
city_value = venue_data['City'].mode()[0] if not venue_data['City'].isna().all() else ""
state_value = venue_data['State'].mode()[0] if not venue_data['State'].isna().all() else ""
market_value = venue_data['Market'].mode()[0] if not venue_data['Market'].isna().all() else ""
avg_event_capacity = venue_data['Avg. Event Capacity'].mean()

# Show auto-filled City and State only (Market is hidden from UI)
st.text_input("City", value=city_value, disabled=True)
st.text_input("State", value=state_value, disabled=True)

# Input form for other fields
with st.form("input_form"):
    headliner = st.selectbox("Headliner", top_headliners)
    support = st.selectbox("Support Act(s)", top_support)
    company_type = st.selectbox("Company Type", top_company_types)
    promoter = st.selectbox("Promoter", top_promoters)

    number_of_shows = st.number_input("Number of Shows", min_value=1, step=1)
    avg_capacity_sold = st.number_input("% Capacity Sold", min_value=0, max_value=100, step=1)
    ticket_price_min = st.number_input("Ticket Price Min (USD)", min_value=0.0, step=1.0)
    ticket_price_max = st.number_input("Ticket Price Max (USD)", min_value=0.0, step=1.0)

    date = st.date_input("Event Date", value=datetime.today())
    hour = st.slider("Hour of Event (0-23)", min_value=0, max_value=23, value=20)

    submit = st.form_submit_button("🎯 Predict Revenue")

# Prediction
if submit:
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()
    country = "United States"

    # Automatically calculate the Genre in the backend
    genre_value = data_options[data_options['Headliner'].astype(str).str.strip().str.strip('"').str.strip("'") == headliner]['Genre'].mode()[0]

    input_data = pd.DataFrame([{
        'Number of Shows': number_of_shows,
        'Headliner': headliner,
        'Support': support,
        'Venue': venue_selected,
        'City': city_value,
        'State': state_value,
        'Country': country,
        'Market': market_value,
        'Company Type': company_type,
        'Promoter': promoter,
        'Genre': genre_value,
        'Avg. Event Capacity': avg_event_capacity,
        '% Capacity Sold': avg_capacity_sold / 100,
        'Ticket Price Min USD': ticket_price_min,
        'Ticket Price Max USD': ticket_price_max,
        'year': year,
        'month': month,
        'day': day,
        'weekday': weekday,
        'hour': hour
    }])

    # Ensure all expected columns
    expected_cols = catboost_model.feature_names_
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = pd.NA
    input_data = input_data[expected_cols]

    # Cast categoricals
    for col in categorical_cols:
        input_data[col] = input_data[col].astype('category')

    # Predictions
    cat_pool = Pool(data=input_data, cat_features=categorical_cols)
    catboost_pred = catboost_model.predict(cat_pool)
    lgb_pred = lgb_model.predict(input_data)

    combined_preds = pd.DataFrame({
        'catboost_pred': catboost_pred,
        'lgb_pred': lgb_pred
    })

    final_prediction = meta_model.predict(combined_preds)[0]
    final_prediction_original = np.expm1(final_prediction)

    st.success(f"💰 Predicted Revenue: **${final_prediction_original:,.2f} USD**")

    with st.expander("🔍 Show Debug Info"):
        st.write("Input Data")
        st.dataframe(input_data)
        st.write("CatBoost Prediction:", catboost_pred)
        st.write("LightGBM Prediction:", lgb_pred)
        st.write("Final Meta Model Prediction:", final_prediction)
        st.write("Final Prediction on Original Scale:", final_prediction_original)