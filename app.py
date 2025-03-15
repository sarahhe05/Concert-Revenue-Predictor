import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

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
with open("models/catboost_model.pkl", "rb") as f:
    catboost_model = pickle.load(f)

with open("models/lgb_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

with open("models/meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

# Load reference dataset
data_options = pd.read_csv("concert_data.csv")

# Define categorical columns
categorical_cols = ['Headliner', 'Support', 'Venue', 'City', 'State', 'Country',
                    'Market', 'Company Type', 'Promoter', 'Genre', 'month', 'weekday']

# Clean categorical columns (remove spaces and quotes)
def clean_text(x):
    return str(x).strip().strip('"').strip("'") if pd.notna(x) else ""

# Extract top 150 headliners and sort them
headliner_counts = data_options['Headliner'].dropna().astype(str).apply(clean_text)
top_headliners = headliner_counts.value_counts().head(150).index.tolist()
alpha_headliners = sorted([h for h in top_headliners if not h[:1].isdigit()])
numeric_headliners = sorted([h for h in top_headliners if h[:1].isdigit()])
top_headliners = alpha_headliners + numeric_headliners

top_support = sorted(data_options['Support'].dropna().astype(str).apply(clean_text).value_counts().head(100).index.tolist())
top_venue = sorted(data_options['Venue'].dropna().value_counts().head(100).index.tolist())
top_company_types = sorted(data_options['Company Type'].dropna().value_counts().head(30).index.tolist())
top_promoters = sorted(data_options['Promoter'].dropna().value_counts().head(30).index.tolist())

st.title("🎤 Concert Gross Revenue Prediction")

venue_selected = st.selectbox("Venue", top_venue)
venue_data = data_options[data_options['Venue'] == venue_selected]
city_value = venue_data['City'].mode()[0] if not venue_data['City'].isna().all() else ""
state_value = venue_data['State'].mode()[0] if not venue_data['State'].isna().all() else ""
market_value = venue_data['Market'].mode()[0] if not venue_data['Market'].isna().all() else ""
avg_event_capacity = venue_data['Avg. Event Capacity'].mean() if len(venue_data) > 0 and not venue_data['Avg. Event Capacity'].isna().all() else 0

st.text_input("City", value=city_value, disabled=True)
st.text_input("State", value=state_value, disabled=True)

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

if submit:
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()
    country = "United States"

    genre_value = data_options[data_options['Headliner'].astype(str).apply(clean_text) == headliner]['Genre'].mode()
    genre_value = genre_value[0] if not genre_value.empty else "Unknown"

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
        'Avg. Event Capacity': float(avg_event_capacity),
        '% Capacity Sold': float(avg_capacity_sold / 100),
        'Ticket Price Min USD': float(ticket_price_min),
        'Ticket Price Max USD': float(ticket_price_max),
        'year': int(year),
        'month': int(month),
        'day': int(day),
        'weekday': int(weekday),
        'hour': int(hour)
    }])

    model_features = catboost_model.feature_names_
    processed_data = pd.DataFrame(index=input_data.index)
    
    for col in model_features:
        if col in categorical_cols:
            processed_data[col] = "Unknown"
        else:
            processed_data[col] = 0.0

    for col in model_features:
        if col in input_data.columns:
            processed_data[col] = input_data[col]
    
    for col in model_features:
        if col in categorical_cols:
            processed_data[col] = processed_data[col].astype('category')
        else:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0.0)

    processed_data = processed_data[model_features]
    
    cat_pool = Pool(data=processed_data, cat_features=categorical_cols)
    catboost_pred = catboost_model.predict(cat_pool)
    lgb_pred = lgb_model.predict(processed_data)

    combined_preds = pd.DataFrame({
        'catboost_pred': catboost_pred,
        'lgb_pred': lgb_pred
    })

    final_prediction = meta_model.predict(combined_preds)[0]
    final_prediction_original = np.expm1(final_prediction)

    st.success(f"💰 Predicted Revenue: **${final_prediction_original:,.2f} USD**")