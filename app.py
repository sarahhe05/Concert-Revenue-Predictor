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
with open("catboost_model.pkl", "rb") as f:
    catboost_model = pickle.load(f)

with open("lgb_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

with open("meta_model.pkl", "rb") as f:
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

# Sort alphabetically, but move numeric ones to the end
alpha_headliners = sorted([h for h in top_headliners if not h[:1].isdigit()])
numeric_headliners = sorted([h for h in top_headliners if h[:1].isdigit()])
top_headliners = alpha_headliners + numeric_headliners

# Other dropdowns
top_support = sorted(data_options['Support'].dropna().astype(str).apply(clean_text).value_counts().head(100).index.tolist())
top_venue = sorted(data_options['Venue'].dropna().value_counts().head(100).index.tolist())
top_company_types = sorted(data_options['Company Type'].dropna().value_counts().head(30).index.tolist())
top_promoters = sorted(data_options['Promoter'].dropna().value_counts().head(30).index.tolist())

# Title
st.title("🎤 Concert Gross Revenue Prediction")

# Debug info
with st.expander("🔍 Model Features"):
    st.write("Expected Model Features:")
    st.write(catboost_model.feature_names_)
    st.write("Categorical Columns:")
    st.write(categorical_cols)

# Venue selection (auto-fills city, state)
venue_selected = st.selectbox("Venue", top_venue)

# Get default values based on venue
venue_data = data_options[data_options['Venue'] == venue_selected]
city_value = venue_data['City'].mode()[0] if not venue_data['City'].isna().all() else ""
state_value = venue_data['State'].mode()[0] if not venue_data['State'].isna().all() else ""
market_value = venue_data['Market'].mode()[0] if not venue_data['Market'].isna().all() else ""
avg_event_capacity = venue_data['Avg. Event Capacity'].mean() if len(venue_data) > 0 and not venue_data['Avg. Event Capacity'].isna().all() else 0

# Show auto-filled values
st.text_input("City", value=city_value, disabled=True)
st.text_input("State", value=state_value, disabled=True)

# Input form
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

    # Get Genre automatically
    genre_value = data_options[data_options['Headliner'].astype(str).apply(clean_text) == headliner]['Genre'].mode()
    genre_value = genre_value[0] if not genre_value.empty else "Unknown"

    # Prepare input data
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
        'Avg. Event Capacity': float(avg_event_capacity),  # Ensure float type
        '% Capacity Sold': float(avg_capacity_sold / 100),  # Ensure float type
        'Ticket Price Min USD': float(ticket_price_min),   # Ensure float type
        'Ticket Price Max USD': float(ticket_price_max),   # Ensure float type
        'year': int(year),                             # Ensure int type
        'month': int(month),                           # Ensure int type
        'day': int(day),                               # Ensure int type
        'weekday': int(weekday),                       # Ensure int type
        'hour': int(hour)                              # Ensure int type
    }])

    # Display current data for debugging
    st.write("Input Data Before Processing:")
    st.dataframe(input_data)
    st.write("Data Types:")
    st.write(input_data.dtypes)

    # Explicitly handle expected columns and their types
    model_features = catboost_model.feature_names_
    processed_data = pd.DataFrame(index=input_data.index)
    
    # Initialize all columns with appropriate default values first
    for col in model_features:
        if col in categorical_cols:
            processed_data[col] = "Unknown"  # Default for categorical
        else:
            processed_data[col] = 0.0  # Default for numerical
    
    # Then copy over the actual data we have
    for col in model_features:
        if col in input_data.columns:
            processed_data[col] = input_data[col]
    
    # Show the processed data
    st.write("Processed Data (Before Type Conversion):")
    st.dataframe(processed_data)
    st.write("Processed Data Types:")
    st.write(processed_data.dtypes)
    
    # Convert data types explicitly
    for col in model_features:
        if col in categorical_cols:
            processed_data[col] = processed_data[col].astype('category')
        else:
            # For numeric columns, ensure they're float
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0.0)
            except Exception as e:
                st.error(f"Error converting column {col}: {str(e)}")
                processed_data[col] = 0.0  # Fallback

    # Show the final processed data
    st.write("Final Processed Data:")
    st.dataframe(processed_data)
    st.write("Final Data Types:")
    st.write(processed_data.dtypes)

    # Make sure indices match expected feature order
    processed_data = processed_data[model_features]
    
    try:
        # CatBoost pool
        cat_pool = Pool(data=processed_data, cat_features=categorical_cols)
        catboost_pred = catboost_model.predict(cat_pool)
        lgb_pred = lgb_model.predict(processed_data)

        # Combine predictions
        combined_preds = pd.DataFrame({
            'catboost_pred': catboost_pred,
            'lgb_pred': lgb_pred
        })

        final_prediction = meta_model.predict(combined_preds)[0]
        final_prediction_original = np.expm1(final_prediction)

        # Display results
        st.success(f"💰 Predicted Revenue: **${final_prediction_original:,.2f} USD**")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Debug information:")
        st.error(f"Error occurred with data shape: {processed_data.shape}")
        
        # Show the first few problematic features
        for i, col in enumerate(model_features):
            st.write(f"Feature {i}: {col} - Type: {processed_data[col].dtype} - First value: {processed_data[col].iloc[0]}")
            if i > 15:  # Limit to avoid too much output
                st.write("...")
                break